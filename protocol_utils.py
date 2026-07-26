"""Start-node selection helpers implementing the HexMaze worksheet protocol.

Rule references in the docstrings point at "Protocol HexMaze worksheet &
start location preparation" (2026 edition), the section that describes how
random start locations have to be picked.
"""

import math
import random
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

import networkx as nx


# Never used as a start or a goal location, because they are hard to reach.
HARD_EXCLUDED_NODES = ('213', '214', '215', '220', '305', '310', '311', '312')

# A start node must be at least this far from the current goal location.
MIN_DISTANCE_FROM_GOAL = 4

# Probe trials: the start node has to be equidistant (< 2 nodes) from the
# current and the previous goal, so the tolerated difference is 0 or 1.
PROBE_EQUIDISTANCE_TOLERANCE = 2

# Start nodes that follow one another *on the same island* should not sit too
# close together ("101...117...120...103" in the protocol). Note that 103 and
# 101 are only 2 apart there, so this constrains successive same-island picks,
# not every pair. Adjacency is forbidden outright by a separate hard rule.
MIN_SAME_ISLAND_SPACING = 3

INFINITY = float('inf')

# --- Maze wiring -----------------------------------------------------------
# Nodes closer than this in the layout are treated as connected.
MAZE_ADJACENCY_THRESHOLD = 65

# Bridges between islands that the distance threshold alone does not catch.
MAZE_MANUAL_EDGES = (
    ('121', '302'), ('324', '401'), ('305', '220'), ('404', '223'), ('201', '124'),
)

# Dead nodes present in the layout file but not part of the maze.
MAZE_REMOVED_NODES = ('501', '502')


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def build_maze_graph(
    node_rows: Iterable[Tuple[object, float, float]],
    *,
    adjacency_threshold: float = MAZE_ADJACENCY_THRESHOLD,
    manual_edges: Iterable[Tuple[str, str]] = MAZE_MANUAL_EDGES,
    remove_nodes: Iterable[str] = MAZE_REMOVED_NODES,
) -> Tuple[nx.Graph, List[Tuple[str, str]]]:
    """Build the maze graph from ``(node_id, x, y)`` rows.

    Island membership follows the node id: 1xx -> island 1, and so on. Returns
    the graph plus any manual edges that could not be added because a node was
    missing from the layout, so the caller can warn about them.
    """
    graph = nx.Graph()
    positions = {}
    for node_id, x, y in node_rows:
        node = str(int(node_id))
        graph.add_node(node, pos=(x, y), group=int(node) // 100)
        positions[node] = (float(x), float(y))

    nodes = list(graph.nodes())
    for i, a in enumerate(nodes):
        ax, ay = positions[a]
        for b in nodes[i + 1:]:
            bx, by = positions[b]
            if math.hypot(ax - bx, ay - by) < adjacency_threshold:
                graph.add_edge(a, b)

    for node in remove_nodes:
        if node in graph:
            graph.remove_node(node)

    missing = []
    for u, v in manual_edges:
        if u in graph and v in graph:
            graph.add_edge(u, v)
        else:
            missing.append((u, v))

    graph.graph['distance_lookup'] = dict(nx.all_pairs_shortest_path_length(graph))
    return graph, missing

def get_shortest_distance(
    graph: nx.Graph,
    source: str,
    target: str,
    cache: Optional[Dict[Tuple[str, str], float]] = None,
) -> float:
    """Return the shortest-path distance between two nodes, using a cache when provided."""
    key = (source, target)
    reverse_key = (target, source)

    if cache is not None and key in cache:
        return cache[key]
    if cache is not None and reverse_key in cache:
        return cache[reverse_key]

    distance = None
    lookup = graph.graph.get('distance_lookup')
    if lookup is not None:
        try:
            distance = lookup[source][target]
        except (KeyError, TypeError):
            distance = None

    if distance is None:
        try:
            distance = nx.shortest_path_length(graph, source=source, target=target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            distance = INFINITY

    if cache is not None:
        cache[key] = distance
        cache[reverse_key] = distance

    return distance


def build_island_graph(graph: nx.Graph) -> nx.Graph:
    """Return (and memoise) the island-level adjacency graph of the maze.

    Two islands are adjacent when at least one maze edge connects a node of one
    island to a node of the other. This is what the protocol means by a start
    island being "directly connected" to a goal island.
    """
    cached = graph.graph.get('island_graph')
    if cached is not None:
        return cached

    island_graph = nx.Graph()
    for _, attrs in graph.nodes(data=True):
        group = attrs.get('group')
        if group is not None:
            island_graph.add_node(group)

    for u, v in graph.edges():
        gu = graph.nodes[u].get('group')
        gv = graph.nodes[v].get('group')
        if gu is not None and gv is not None and gu != gv:
            island_graph.add_edge(gu, gv)

    graph.graph['island_graph'] = island_graph
    return island_graph


# ---------------------------------------------------------------------------
# Island sequence
# ---------------------------------------------------------------------------

def _order_block(
    groups: Sequence[int],
    *,
    hard_ban: Set[Optional[int]],
    soft_ban: Set[Optional[int]],
) -> List[int]:
    """Return a random permutation of ``groups`` whose head respects the bans.

    ``hard_ban`` is honoured whenever any island is left after applying it;
    ``soft_ban`` is only a preference. The tail is always fully shuffled, which
    is what keeps the island order from settling into a fixed cycle.
    """
    block = list(groups)

    allowed = [g for g in block if g not in hard_ban] or block
    preferred = [g for g in allowed if g not in soft_ban] or allowed

    head = random.choice(preferred)
    tail = [g for g in block if g != head]
    random.shuffle(tail)
    return [head] + tail


def build_island_sequence(
    *,
    total_trials: int,
    available_groups: Iterable[int],
    goal_group: int,
    forced_t1_group: Optional[int] = None,
    prev_last_group: Optional[int] = None,
    prev_first_group: Optional[int] = None,
) -> List[int]:
    """Create the per-trial island order required by the protocol.

    - Start island is never the goal island.
    - Randomised in blocks of three trials; each block uses all three remaining
      islands exactly once, so the first three trials use three different
      islands and the totals come out balanced (7:7:6 over 20 trials).
    - Trial 1 avoids the island the previous session ended on, and prefers to
      also avoid the island the previous session started on.
    - Consecutive blocks never repeat an island across the block boundary.
    """
    groups = sorted({g for g in available_groups if g != goal_group})
    if not groups or total_trials <= 0:
        return []

    sequence: List[int] = []
    while len(sequence) < total_trials:
        if not sequence:
            if forced_t1_group is not None and forced_t1_group in groups:
                # Trial 1 was already pinned by the NGL/PT rules.
                tail = [g for g in groups if g != forced_t1_group]
                random.shuffle(tail)
                block = [forced_t1_group] + tail
            else:
                hard_ban = {prev_last_group} if prev_last_group is not None else set()
                soft_ban = set(hard_ban)
                if prev_first_group is not None:
                    soft_ban.add(prev_first_group)
                block = _order_block(groups, hard_ban=hard_ban, soft_ban=soft_ban)
        else:
            ban = {sequence[-1]}
            block = _order_block(groups, hard_ban=ban, soft_ban=ban)

        sequence.extend(block)

    return sequence[:total_trials]


# ---------------------------------------------------------------------------
# Probe-trial topology
# ---------------------------------------------------------------------------

def is_topologically_valid_group(
    graph: nx.Graph,
    target_group: int,
    *,
    prev_goal_group: Optional[int],
    current_goal_group: Optional[int],
) -> bool:
    """Return True if ``target_group`` is a fair probe-trial start island.

    The protocol requires the start island to be directly connected to *both*
    the previous and the current goal island, so that reaching either goal does
    not need an extra intermediate island. The adjacency is read off the maze
    itself rather than hard-coded, so the rule holds for every goal pairing.

    This applies to probe trials only; it is not a constraint on ordinary trials.
    """
    if prev_goal_group is None or current_goal_group is None:
        return True

    if target_group in {prev_goal_group, current_goal_group}:
        return False

    island_graph = build_island_graph(graph)
    if target_group not in island_graph:
        return False

    return (
        island_graph.has_edge(target_group, prev_goal_group)
        and island_graph.has_edge(target_group, current_goal_group)
    )


# ---------------------------------------------------------------------------
# Node selection
# ---------------------------------------------------------------------------

def _independent_set_size(graph: nx.Graph, nodes: Iterable[str]) -> int:
    """How many mutually non-adjacent nodes ``nodes`` holds.

    Greedy, lowest-degree-first. It never over-estimates, and on the sparse maze
    islands it matches the exact maximum independent set.
    """
    remaining = set(nodes)
    count = 0
    while remaining:
        node = min(
            remaining,
            key=lambda n: sum(1 for m in graph.neighbors(n) if m in remaining),
        )
        count += 1
        remaining.discard(node)
        remaining.difference_update(graph.neighbors(node))
    return count


def island_capacity(
    graph: nx.Graph,
    *,
    island: int,
    goal_node: str,
    forbidden_set: Set[str],
    min_distance_from_goal: int = MIN_DISTANCE_FROM_GOAL,
    distance_cache: Optional[Dict[Tuple[str, str], float]] = None,
) -> int:
    """How many start nodes an island can supply for one session.

    That is, the largest set of mutually non-adjacent nodes on the island that
    are all far enough from the goal. Used to explain why a goal node cannot
    support the requested number of trials.
    """
    pool = [
        node for node, attrs in graph.nodes(data=True)
        if attrs.get('group') == island
        and node not in forbidden_set
        and get_shortest_distance(graph, node, goal_node, distance_cache) >= min_distance_from_goal
    ]
    return _independent_set_size(graph, pool)


class SessionAllocation(NamedTuple):
    """One session's start nodes, plus any reused from the previous session."""

    sequence: List[str]
    borrowed: List[str]


def _pick_non_adjacent(
    graph: nx.Graph,
    candidates: List[str],
    count: int,
    blocked: Set[str],
    soft_avoid: Set[str],
    attempts: int,
) -> Optional[List[str]]:
    """Choose ``count`` mutually non-adjacent nodes, none touching ``blocked``.

    Randomised greedy with restarts: on the real maze an island holds only
    10-12 such nodes against a demand of up to 10, so a single pass frequently
    strands itself while a restart succeeds. Nodes in ``soft_avoid`` are tried
    last, which honours the preference without ever failing because of it.
    """
    if count <= 0:
        return []

    # A blocked node is not adjacent to itself, so it has to be excluded by
    # identity as well - otherwise a forced trial-1 pick gets chosen twice.
    pool = [
        n for n in candidates
        if n not in blocked and not any(graph.has_edge(n, b) for b in blocked)
    ]
    if len(pool) < count:
        return None

    for _ in range(attempts):
        order = pool[:]
        random.shuffle(order)
        order.sort(key=lambda n: n in soft_avoid)  # stable: keeps the shuffle within each group

        chosen: List[str] = []
        for node in order:
            if all(not graph.has_edge(node, other) for other in chosen):
                chosen.append(node)
                if len(chosen) == count:
                    return chosen

    return None


def _pick_with_fallback(
    graph: nx.Graph,
    pool: List[str],
    count: int,
    blocked: Set[str],
    soft_avoid: Set[str],
    attempts: int,
    fallback: List[str],
) -> Optional[Tuple[List[str], List[str]]]:
    """Pick ``count`` nodes, topping up from ``fallback`` only if needed.

    Returns ``(picked, borrowed)`` where ``borrowed`` lists just the nodes taken
    from ``fallback`` to cover a shortfall. A fallback node chosen on its own
    merits is not borrowed — it satisfied every rule like any other pick.

    An island holds 10-12 mutually non-adjacent nodes, so a 30-trial session can
    ask for more than the layout can give. The lab's rule for that case is to
    reuse the previous session's start nodes; those are exempt from the spacing
    rules but still have to sit >= 4 from the goal, which ``fallback`` has
    already been filtered for. Borrowing is kept to the minimum: the largest
    number of fresh nodes that still works is used first.
    """
    strict = _pick_non_adjacent(graph, pool, count, blocked, soft_avoid, attempts)
    if strict is not None:
        return strict, []

    if not fallback:
        return None

    # Try holding the fallback nodes in reserve first. They are also valid fresh
    # picks, so left in the pool they get spent on the non-adjacent set and are
    # no longer available to cover the gap they exist for.
    fallback_set = set(fallback)
    reserved_pool = [n for n in pool if n not in fallback_set]

    for candidate_pool in (reserved_pool, pool):
        reachable = [
            n for n in candidate_pool
            if n not in blocked and not any(graph.has_edge(n, b) for b in blocked)
        ]
        # Fewer fresh nodes only means more to borrow, so start from the most
        # this pool can hold and stop at the first count it can actually deliver.
        for fresh_count in range(min(count - 1, _independent_set_size(graph, reachable)), -1, -1):
            fresh = _pick_non_adjacent(
                graph, candidate_pool, fresh_count, blocked, soft_avoid, attempts,
            )
            if fresh is None:
                continue

            taken = set(fresh) | blocked
            borrowed = [n for n in fallback if n not in taken][:count - fresh_count]
            if len(borrowed) == count - fresh_count:
                return fresh + borrowed, borrowed
            break

    return None


def _order_for_spacing(
    graph: nx.Graph,
    nodes: List[str],
    *,
    first: Optional[str] = None,
    preceded_by: Optional[str] = None,
    attempts: int = 60,
    distance_cache: Optional[Dict[Tuple[str, str], float]] = None,
) -> List[str]:
    """Order an island's nodes so successive visits are not bunched together.

    ``first`` pins the opening node (a forced NGL/PT trial-1 pick) and
    ``preceded_by`` is the island's last node from an earlier half of the same
    session. Returns the best ordering found; this rule is a soft one, so a
    perfect ordering is preferred but not required.
    """
    movable = [n for n in nodes if n != first]
    best, best_score = None, None

    for _ in range(attempts):
        random.shuffle(movable)
        order = ([first] if first is not None else []) + movable

        chain = ([preceded_by] if preceded_by else []) + order
        score = sum(
            1 for a, b in zip(chain, chain[1:])
            if get_shortest_distance(graph, a, b, distance_cache) < MIN_SAME_ISLAND_SPACING
        )
        if score == 0:
            return order
        if best_score is None or score < best_score:
            best, best_score = order[:], score

    return best if best is not None else list(nodes)


def select_session_nodes(
    graph: nx.Graph,
    *,
    island_sequence: List[int],
    per_trial_goals: Sequence[str],
    forbidden_set: Set[str],
    soft_avoid: Iterable[str] = (),
    min_distance_from_goal: int = MIN_DISTANCE_FROM_GOAL,
    forced_first_node: Optional[str] = None,
    fallback_nodes: Sequence[str] = (),
    attempts: int = 60,
    distance_cache: Optional[Dict[Tuple[str, str], float]] = None,
) -> Optional['SessionAllocation']:
    """Choose every start node for one session in one pass.

    Allocating whole islands at once instead of trial by trial is what makes the
    tight sessions work: a 30-trial session needs about 10 non-adjacent nodes
    per island against a capacity of 10-12, and a greedy per-trial walk paints
    itself into a corner long before it runs out of nodes.

    Nodes are grouped by ``(island, goal)`` rather than by island alone, because
    an ephys NGL session moves the goal halfway through and the >= 4 rule is
    then measured against a different node for the second half.

    ``fallback_nodes`` are the previous session's start nodes, in their original
    order. When an island cannot supply enough fresh nodes - which happens once
    a session asks for more than 20 trials - the shortfall is taken from this
    list, in order, and only as far down it as needed. Borrowed nodes still have
    to sit >= ``min_distance_from_goal`` from the goal, stay off the goal island
    and off the never-use list, but they are exempt from the non-adjacency and
    spacing rules.

    Returns a ``SessionAllocation`` (the sequence aligned to ``island_sequence``
    plus the nodes that had to be borrowed), or None if the islands cannot
    supply that many nodes even with the fallback.
    """
    soft_avoid = set(soft_avoid)

    keys = list(zip(island_sequence, per_trial_goals))
    needs: Dict[Tuple[int, str], int] = {}
    for key in keys:
        needs[key] = needs.get(key, 0) + 1

    pools = {
        (island, goal): [
            node for node, attrs in graph.nodes(data=True)
            if attrs.get('group') == island
            # The start island is never the goal island. The island sequence
            # already guarantees this, but borrowed fallback nodes are filtered
            # through the same pool, so state it here rather than rely on that.
            and island != graph.nodes[goal].get('group')
            and node not in forbidden_set
            and get_shortest_distance(graph, node, goal, distance_cache)
            >= min_distance_from_goal
        ]
        for island, goal in needs
    }

    # Previous-session nodes that are legal for each group, kept in their
    # original order so borrowing works down the list from the top.
    fallbacks = {
        key: [node for node in fallback_nodes if node in set(pool)]
        for key, pool in pools.items()
    }

    forced_key = keys[0] if forced_first_node else None

    for _ in range(attempts):
        chosen: Dict[Tuple[int, str], List[str]] = {}
        borrowed_all: List[str] = []
        blocked = set()
        if forced_first_node:
            blocked.add(forced_first_node)

        # Serve the groups with the least slack first; they fail fastest.
        for key in sorted(needs, key=lambda k: len(pools[k]) - needs[k]):
            count = needs[key]
            forced = forced_first_node if key == forced_key else None
            if forced:
                count -= 1

            result = _pick_with_fallback(
                graph, pools[key], count, blocked, soft_avoid, attempts, fallbacks[key],
            )
            if result is None:
                break

            picked, borrowed = result
            if forced:
                picked = [forced] + picked
            chosen[key] = picked
            borrowed_all.extend(borrowed)
            blocked.update(picked)
        else:
            # Order within each group so successive visits to an island are not
            # bunched together, carrying the island's tail across a goal switch.
            ordered: Dict[Tuple[int, str], List[str]] = {}
            last_on_island: Dict[int, str] = {}
            for key in keys:
                if key in ordered:
                    continue
                island, _ = key
                ordered[key] = _order_for_spacing(
                    graph, chosen[key],
                    first=forced_first_node if key == forced_key else None,
                    preceded_by=last_on_island.get(island),
                    distance_cache=distance_cache,
                )
                last_on_island[island] = ordered[key][-1]

            cursors = {key: 0 for key in ordered}
            sequence = []
            for key in keys:
                sequence.append(ordered[key][cursors[key]])
                cursors[key] += 1
            return SessionAllocation(sequence, sorted(set(borrowed_all)))

    return None


def choose_first_trial_group(
    graph: nx.Graph,
    *,
    goal_group: int,
    available_groups: Iterable[int],
    prev_goal_group: Optional[int] = None,
    prev_last_group: Optional[int] = None,
    prev_first_group: Optional[int] = None,
) -> Optional[int]:
    """Pick the start island for trial 1 of an NGL session.

    Besides never being the goal island, it must differ from the previous goal
    island, and from the island the previous session ended on. Differing from
    the island the previous session started on is a preference.
    """
    groups = sorted({g for g in available_groups if g != goal_group})
    if not groups:
        return None

    hard_ban = {g for g in (prev_goal_group, prev_last_group) if g is not None}
    allowed = [g for g in groups if g not in hard_ban]
    if not allowed:
        # The previous goal island matters more than the previous last island.
        allowed = [g for g in groups if g != prev_goal_group] or groups

    preferred = [g for g in allowed if g != prev_first_group] or allowed
    return random.choice(preferred)


def select_probe_start_node(
    graph: nx.Graph,
    *,
    goal_node: str,
    prev_goal_node: str,
    forbidden_set: Set[str],
    min_distance_from_goal: int = MIN_DISTANCE_FROM_GOAL,
    soft_avoid: Iterable[str] = (),
    prev_last_group: Optional[int] = None,
    prev_first_group: Optional[int] = None,
    island_demand: int = 0,
    tolerance: int = PROBE_EQUIDISTANCE_TOLERANCE,
    distance_cache: Optional[Dict[Tuple[str, str], float]] = None,
) -> Tuple[Optional[str], dict]:
    """Pick the trial-1 start node for a probe trial.

    The start node must be equidistant (difference < ``tolerance`` nodes) from
    the current and the previous goal, sit at least ``min_distance_from_goal``
    from the current goal, and live on an island directly connected to both goal
    islands. Returns ``(node, info)``; ``info`` carries the distances and notes
    which constraints had to be relaxed.
    """
    soft_avoid = set(soft_avoid)
    goal_group = graph.nodes[goal_node].get('group')
    prev_goal_group = graph.nodes[prev_goal_node].get('group')
    info: dict = {'relaxed': []}

    def collect(group_filter) -> List[dict]:
        found = []
        for node, attrs in graph.nodes(data=True):
            if node in forbidden_set:
                continue
            group = attrs.get('group')
            if group is None or not group_filter(group):
                continue

            d_new = get_shortest_distance(graph, node, goal_node, distance_cache)
            if d_new == INFINITY or d_new < min_distance_from_goal:
                continue

            d_old = get_shortest_distance(graph, node, prev_goal_node, distance_cache)
            if d_old == INFINITY:
                continue

            found.append({
                'node': node,
                'group': group,
                'dist_new': d_new,
                'dist_old': d_old,
                'diff': abs(d_new - d_old),
            })
        return found

    def topological(group):
        return is_topologically_valid_group(
            graph, group, prev_goal_group=prev_goal_group, current_goal_group=goal_group
        )

    candidates = collect(topological)
    if not candidates:
        # No island is directly connected to both goals; fall back to any
        # non-goal island so the session can still be prepared, and say so.
        info['relaxed'].append('start_island_topology')
        candidates = collect(lambda g: g not in {goal_group, prev_goal_group})

    if not candidates:
        info['reason'] = 'no candidate island available'
        return None, info

    # Trial 1 must not reuse the island the previous session ended on. This is
    # applied before the equidistance filter, otherwise a pool that happens to
    # be all on that island would force a violation.
    if prev_last_group is not None:
        without_prev_last = [c for c in candidates if c['group'] != prev_last_group]
        if without_prev_last:
            candidates = without_prev_last
        else:
            info['relaxed'].append('first_trial_island_differs_from_prev_last')

    pool: List[dict] = []
    for max_diff in range(0, tolerance):
        pool = [c for c in candidates if c['diff'] <= max_diff]
        if pool:
            break

    if not pool:
        info['relaxed'].append('equidistance')
        best = min(c['diff'] for c in candidates)
        pool = [c for c in candidates if c['diff'] == best]

    # Pinning trial 1 costs its island one node. On a 30-trial session an island
    # needs ~10 non-adjacent nodes against a capacity of 10-12, so a pick that
    # is not part of a maximum independent set makes the rest unsatisfiable.
    if island_demand > 1:
        viable = []
        for candidate in pool:
            rest = [
                node for node, attrs in graph.nodes(data=True)
                if attrs.get('group') == candidate['group']
                and node not in forbidden_set
                and node != candidate['node']
                and not graph.has_edge(node, candidate['node'])
                and get_shortest_distance(graph, node, goal_node, distance_cache)
                >= min_distance_from_goal
            ]
            if _independent_set_size(graph, rest) >= island_demand - 1:
                viable.append(candidate)
        if viable:
            pool = viable

    for tier in (
        [c for c in pool if c['node'] not in soft_avoid and c['group'] != prev_first_group],
        [c for c in pool if c['node'] not in soft_avoid],
        [c for c in pool if c['group'] != prev_first_group],
        pool,
    ):
        if tier:
            chosen = random.choice(tier)
            info.update(chosen)
            return chosen['node'], info

    return None, info


# ---------------------------------------------------------------------------
# Compliance report
# ---------------------------------------------------------------------------

def _check(name: str, passed: bool, details, severity: str = 'hard') -> dict:
    return {'name': name, 'passed': bool(passed), 'severity': severity, 'details': details}


def _goal_segments(per_trial_goals: Sequence[str]) -> List[Tuple[int, int]]:
    """Split a session into ``(start, stop)`` runs that share one goal node.

    An ephys NGL session runs 15 trials on the old goal and 15 on the new one;
    rules about island usage apply within each run, not across the switch.
    """
    segments = []
    start = 0
    for i in range(1, len(per_trial_goals) + 1):
        if i == len(per_trial_goals) or per_trial_goals[i] != per_trial_goals[start]:
            segments.append((start, i))
            start = i
    return segments


def evaluate_protocol_compliance(
    graph: nx.Graph,
    sequence: List[str],
    *,
    per_trial_goals: Sequence[str],
    session_kind: str = 'Normal',
    prev_goal_node: Optional[str] = None,
    probe_reference_goal: Optional[str] = None,
    prev_used_nodes: Iterable[str] = (),
    reused_nodes: Iterable[str] = (),
    prev_first_group: Optional[int] = None,
    prev_last_group: Optional[int] = None,
    min_distance_from_goal: int = MIN_DISTANCE_FROM_GOAL,
    distance_cache: Optional[Dict[Tuple[str, str], float]] = None,
) -> List[dict]:
    """Check a generated sequence against every start-location rule.

    Each entry carries a ``severity``: ``hard`` rules must never fail, ``soft``
    rules are the ones the protocol qualifies with "if possible" / "try to avoid".

    ``reused_nodes`` are previous-session start nodes deliberately brought back
    to fill a shortfall beyond 20 trials. They are exempt from the spacing rules
    by design, so they are excluded from those checks and reported separately.
    """
    checks: List[dict] = []
    if not sequence:
        return checks

    prev_used = set(prev_used_nodes)
    reused = set(reused_nodes) & set(sequence)
    groups = [graph.nodes[node].get('group') for node in sequence]
    goal_groups = [graph.nodes[node].get('group') for node in per_trial_goals]

    # --- Excluded nodes -----------------------------------------------------
    used_excluded = sorted(set(sequence) & set(HARD_EXCLUDED_NODES))
    checks.append(_check(
        'no_hard_excluded_nodes', not used_excluded,
        f'excluded nodes used: {used_excluded}' if used_excluded else 'none used',
    ))

    # --- Start island never the goal island ---------------------------------
    clashes = [
        (i + 1, node) for i, node in enumerate(sequence)
        if groups[i] == goal_groups[i]
    ]
    checks.append(_check(
        'start_island_never_goal_island', not clashes,
        f'trials on the goal island: {clashes}' if clashes else 'ok',
    ))

    # --- First three trials use three different islands ---------------------
    first_three = groups[:3]
    checks.append(_check(
        'first_three_trials_use_different_islands',
        len(set(first_three)) == min(3, len(first_three)),
        first_three,
    ))

    # --- Block structure ----------------------------------------------------
    bad_blocks = []
    for start in range(0, len(groups), 3):
        block = groups[start:start + 3]
        if len(set(block)) != len(block):
            bad_blocks.append((start + 1, block))
    checks.append(_check(
        'blocks_of_three_use_different_islands', not bad_blocks,
        f'repeating islands in blocks starting at trial {bad_blocks}' if bad_blocks else 'ok',
    ))

    # --- Island balance (7:7:6 over 20 trials) ------------------------------
    # An ephys NGL session swaps the goal halfway, which legitimately swaps the
    # set of usable islands, so each goal segment is balanced on its own.
    balanced = True
    detail = []
    for start, stop in _goal_segments(per_trial_goals):
        counts: Dict[int, int] = {}
        for group in groups[start:stop]:
            counts[group] = counts.get(group, 0) + 1
        length = stop - start
        islands_used = len(counts) or 1
        low, high = length // islands_used, math.ceil(length / islands_used)
        if not all(low <= c <= high for c in counts.values()):
            balanced = False
        detail.append(
            f'trials {start + 1}-{stop}: {dict(sorted(counts.items()))} '
            f'(expected each in {low}..{high})'
        )
    checks.append(_check('island_counts_balanced', balanced, '; '.join(detail)))

    # --- Distance to the goal of that trial ---------------------------------
    too_close = []
    for i, node in enumerate(sequence):
        distance = get_shortest_distance(graph, node, per_trial_goals[i], distance_cache)
        if distance == INFINITY or distance < min_distance_from_goal:
            too_close.append((i + 1, node, distance))
    checks.append(_check(
        'distance_to_goal_at_least_min', not too_close,
        f'trials closer than {min_distance_from_goal}: {too_close}' if too_close else
        f'all >= {min_distance_from_goal}',
    ))

    # --- No repeats within the session --------------------------------------
    repeated = sorted({n for n in sequence if sequence.count(n) > 1})
    checks.append(_check(
        'no_repeated_start_nodes', not repeated,
        f'repeated: {repeated}' if repeated else 'ok',
    ))

    # --- Dispersion: no two start nodes adjacent ----------------------------
    adjacent = [
        (a, b) for i, a in enumerate(sequence) for b in sequence[i + 1:]
        if graph.has_edge(a, b) and a not in reused and b not in reused
    ]
    checks.append(_check(
        'start_nodes_not_adjacent', not adjacent,
        f'adjacent pairs: {adjacent}' if adjacent else
        ('ok (reused nodes exempt)' if reused else 'ok'),
    ))

    # --- Successive same-island start nodes kept apart (soft) ---------------
    # The protocol's own example (101...117...120...103) lets a later trial come
    # back close to an earlier one, so only consecutive visits to an island are
    # compared.
    last_seen: Dict[int, str] = {}
    crowded = []
    for node in sequence:
        group = graph.nodes[node].get('group')
        previous = last_seen.get(group)
        if previous is not None and node not in reused and previous not in reused:
            distance = get_shortest_distance(graph, previous, node, distance_cache)
            if distance < MIN_SAME_ISLAND_SPACING:
                crowded.append((previous, node, distance))
        last_seen[group] = node
    checks.append(_check(
        'successive_same_island_nodes_dispersed', not crowded,
        f'closer than {MIN_SAME_ISLAND_SPACING}: {crowded}' if crowded else 'ok',
        severity='soft',
    ))

    # --- Previous session's start nodes (soft) ------------------------------
    # Nodes deliberately borrowed to fill a shortfall are reported on their own
    # line, so this check only flags unintended overlap.
    overlap = sorted((set(sequence) & prev_used) - reused)
    checks.append(_check(
        'avoid_previous_session_start_nodes', not overlap,
        f'reused from last session: {overlap}' if overlap else 'none reused',
        severity='soft',
    ))

    if reused:
        borrowed = [(i + 1, n) for i, n in enumerate(sequence) if n in reused]
        checks.append(_check(
            'reused_previous_nodes_to_fill_shortfall', False,
            f'{len(borrowed)} trial(s) reuse a previous start node because the islands '
            f'cannot supply enough fresh ones: {borrowed}. These are exempt from the '
            f'spacing rules by design - check them by hand.',
            severity='soft',
        ))

    # --- Previous goal node (soft) ------------------------------------------
    if prev_goal_node:
        hits = [i + 1 for i, node in enumerate(sequence) if node == prev_goal_node]
        checks.append(_check(
            'avoid_previous_goal_node', not hits,
            f'previous goal {prev_goal_node} used on trials {hits}' if hits else 'ok',
            severity='soft',
        ))

    # --- Trial 1 vs. previous session ---------------------------------------
    if prev_last_group is not None:
        checks.append(_check(
            'first_trial_island_differs_from_prev_last',
            groups[0] != prev_last_group,
            f'trial 1 island {groups[0]}, previous session last island {prev_last_group}',
        ))

    if prev_first_group is not None:
        checks.append(_check(
            'first_trial_island_differs_from_prev_first',
            groups[0] != prev_first_group,
            f'trial 1 island {groups[0]}, previous session first island {prev_first_group}',
            severity='soft',
        ))

    # --- NGL / PT trial-1 rules ---------------------------------------------
    if session_kind in ('NGL', 'PT') and prev_goal_node:
        prev_goal_group = graph.nodes[prev_goal_node].get('group')
        checks.append(_check(
            'ngl_pt_first_trial_island_differs_from_prev_goal',
            groups[0] != prev_goal_group,
            f'trial 1 island {groups[0]}, previous goal island {prev_goal_group}',
        ))

    if session_kind == 'PT' and probe_reference_goal:
        start = sequence[0]
        d_new = get_shortest_distance(graph, start, per_trial_goals[0], distance_cache)
        d_old = get_shortest_distance(graph, start, probe_reference_goal, distance_cache)
        diff = abs(d_new - d_old)
        checks.append(_check(
            'probe_start_equidistant_from_both_goals',
            diff < PROBE_EQUIDISTANCE_TOLERANCE,
            f'dist(new)={d_new}, dist(old)={d_old}, diff={diff} '
            f'(must be < {PROBE_EQUIDISTANCE_TOLERANCE})',
        ))
        checks.append(_check(
            'probe_start_island_connected_to_both_goals',
            is_topologically_valid_group(
                graph, groups[0],
                prev_goal_group=graph.nodes[probe_reference_goal].get('group'),
                current_goal_group=goal_groups[0],
            ),
            f'start island {groups[0]}, goal islands '
            f'{graph.nodes[probe_reference_goal].get("group")} / {goal_groups[0]}',
        ))

    return checks
