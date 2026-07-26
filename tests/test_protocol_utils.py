import random
from collections import Counter
from pathlib import Path

import networkx as nx
import pytest

from protocol_utils import (
    HARD_EXCLUDED_NODES,
    MIN_DISTANCE_FROM_GOAL,
    build_island_graph,
    build_island_sequence,
    build_maze_graph,
    choose_first_trial_group,
    evaluate_protocol_compliance,
    get_shortest_distance,
    is_topologically_valid_group,
    island_capacity,
    select_probe_start_node,
    select_session_nodes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_maze():
    """A four-island stand-in maze with the same shape as the real one.

    Each island is a path of 24 nodes, as in node_list_new.csv. Island
    adjacency comes out as 1-2, 1-3, 2-3, 2-4, 3-4 — islands 1 and 4 are NOT
    directly connected, which is what the real maze produces and what the
    probe-trial topology rule hinges on.
    """
    graph = nx.Graph()
    for island in (1, 2, 3, 4):
        for i in range(1, 25):
            graph.add_node(f'{island}{i:02d}', group=island)
        for i in range(1, 24):
            graph.add_edge(f'{island}{i:02d}', f'{island}{i + 1:02d}')

    for a, b in [('124', '201'), ('101', '301'), ('224', '324'), ('212', '401'), ('313', '424')]:
        graph.add_edge(a, b)
    return graph


@pytest.fixture
def maze():
    return make_maze()


@pytest.fixture(scope='module')
def real_maze():
    """The actual maze from node_list_new.csv, so capacity-sensitive rules are
    exercised against the layout the lab really uses."""
    csv = Path(__file__).resolve().parent.parent / 'node_list_new.csv'
    if not csv.exists():
        pytest.skip('node_list_new.csv not available')

    rows = []
    with csv.open() as handle:
        for line in handle:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                rows.append((int(parts[0]), float(parts[1]), float(parts[2])))

    graph, missing = build_maze_graph(rows)
    assert missing == []
    return graph


# ---------------------------------------------------------------------------
# build_island_sequence
# ---------------------------------------------------------------------------

def test_island_sequence_uses_three_island_blocks_and_balances_counts():
    for seed in range(30):
        random.seed(seed)
        sequence = build_island_sequence(
            total_trials=20, available_groups=[1, 2, 3, 4], goal_group=4,
        )

        assert len(sequence) == 20
        assert 4 not in sequence
        assert len(set(sequence[:3])) == 3

        for start in range(0, len(sequence), 3):
            block = sequence[start:start + 3]
            assert len(set(block)) == len(block)

        assert sorted(Counter(sequence).values()) == [6, 7, 7]


def test_island_sequence_with_forced_first_island_terminates_and_stays_balanced():
    """Regression: the forced trial-1 island used to be dropped from every later
    block, which emptied the block and spun the while loop forever."""
    for seed in range(30):
        random.seed(seed)
        sequence = build_island_sequence(
            total_trials=20, available_groups=[1, 2, 3, 4], goal_group=4,
            forced_t1_group=1, prev_last_group=2,
        )

        assert len(sequence) == 20
        assert sequence[0] == 1
        # The forced island must keep appearing later on, not be used up.
        assert Counter(sequence)[1] >= 6
        assert sorted(Counter(sequence).values()) == [6, 7, 7]


def test_island_sequence_never_repeats_an_island_across_a_block_boundary():
    for seed in range(30):
        random.seed(seed)
        sequence = build_island_sequence(
            total_trials=30, available_groups=[1, 2, 3, 4], goal_group=2,
        )
        assert all(a != b for a, b in zip(sequence, sequence[1:]))


def test_island_sequence_first_trial_avoids_previous_session_islands():
    heads = set()
    for seed in range(40):
        random.seed(seed)
        sequence = build_island_sequence(
            total_trials=20, available_groups=[1, 2, 3, 4], goal_group=4,
            prev_last_group=1, prev_first_group=2,
        )
        heads.add(sequence[0])

    # Island 1 is banned outright; island 2 only when island 3 is available.
    assert 1 not in heads
    assert heads == {3}


def test_island_sequence_orders_are_not_a_fixed_cycle():
    orders = set()
    for seed in range(40):
        random.seed(seed)
        orders.add(tuple(build_island_sequence(
            total_trials=9, available_groups=[1, 2, 3, 4], goal_group=4,
        )))
    assert len(orders) > 5


# ---------------------------------------------------------------------------
# Island topology (probe trials)
# ---------------------------------------------------------------------------

def test_island_graph_matches_maze_connectivity(maze):
    island_graph = build_island_graph(maze)
    assert {tuple(sorted(e)) for e in island_graph.edges()} == {(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)}


def test_topological_rule_matches_the_protocol_example(maze):
    """Protocol: prev goal on island 2, new goal on island 1 -> start island 3,
    not island 4, since island 4 does not touch island 1."""
    assert is_topologically_valid_group(maze, 3, prev_goal_group=2, current_goal_group=1)
    assert not is_topologically_valid_group(maze, 4, prev_goal_group=2, current_goal_group=1)


def test_topological_rule_is_derived_from_the_graph_not_hardcoded(maze):
    """Regression: the {1,2} -> island 3 example used to be hard-coded, so every
    other goal pairing was unchecked."""
    # prev goal island 1, current goal island 4: only islands 2 and 3 touch both.
    assert is_topologically_valid_group(maze, 2, prev_goal_group=1, current_goal_group=4)
    assert is_topologically_valid_group(maze, 3, prev_goal_group=1, current_goal_group=4)

    # A start island can never be one of the two goal islands.
    assert not is_topologically_valid_group(maze, 1, prev_goal_group=1, current_goal_group=4)
    assert not is_topologically_valid_group(maze, 4, prev_goal_group=1, current_goal_group=4)


def test_topological_rule_rejects_island_missing_one_connection():
    graph = nx.Graph()
    for island in (1, 2, 3, 4):
        graph.add_node(f'{island}01', group=island)
    # Island 4 touches island 2 only; island 3 touches both 1 and 2.
    graph.add_edges_from([('101', '301'), ('201', '301'), ('201', '401')])

    assert is_topologically_valid_group(graph, 3, prev_goal_group=2, current_goal_group=1)
    assert not is_topologically_valid_group(graph, 4, prev_goal_group=2, current_goal_group=1)


# ---------------------------------------------------------------------------
# select_session_nodes (whole-session allocation)
# ---------------------------------------------------------------------------

def allocate(maze, goal_segments, *, forced_first_node=None, soft_avoid=(),
             fallback_nodes=(), seed=0):
    islands, per_trial_goals = [], []
    for index, (goal, count) in enumerate(goal_segments):
        islands.extend(build_island_sequence(
            total_trials=count, available_groups=[1, 2, 3, 4],
            goal_group=maze.nodes[goal]['group'],
            forced_t1_group=(
                maze.nodes[forced_first_node]['group']
                if forced_first_node and index == 0 else None
            ),
            prev_last_group=None if index == 0 else islands[-1],
        ))
        per_trial_goals.extend([goal] * count)

    random.seed(seed)
    allocation = select_session_nodes(
        maze, island_sequence=islands, per_trial_goals=per_trial_goals,
        forbidden_set=set(HARD_EXCLUDED_NODES) | {g for g, _ in goal_segments},
        forced_first_node=forced_first_node, soft_avoid=soft_avoid,
        fallback_nodes=fallback_nodes,
    )
    return allocation, islands, per_trial_goals


def test_session_allocation_fills_a_30_trial_session(real_maze):
    for seed in range(20):
        allocation, islands, goals = allocate(real_maze, [('118', 30)], seed=seed)
        assert allocation is not None, seed
        sequence = allocation.sequence
        assert len(sequence) == 30
        assert len(set(sequence)) == 30
        assert [real_maze.nodes[n]['group'] for n in sequence] == islands


def test_session_allocation_respects_hard_rules(real_maze):
    for seed in range(20):
        allocation, _, goals = allocate(real_maze, [('118', 20)], seed=seed)
        assert allocation is not None
        sequence = allocation.sequence
        for node, goal in zip(sequence, goals):
            assert node not in HARD_EXCLUDED_NODES
            assert get_shortest_distance(real_maze, node, goal) >= MIN_DISTANCE_FROM_GOAL
        for i, a in enumerate(sequence):
            for b in sequence[i + 1:]:
                assert not real_maze.has_edge(a, b), f'{a} and {b} are adjacent'


def test_session_allocation_uses_the_forced_first_node_exactly_once(real_maze):
    """Regression: a blocked node is not adjacent to itself, so the forced
    trial-1 pick used to be selected a second time and shorten the sequence."""
    for seed in range(20):
        allocation, _, _ = allocate(
            real_maze, [('118', 30)], forced_first_node='210', seed=seed,
        )
        assert allocation is not None
        sequence = allocation.sequence
        assert len(sequence) == 30
        assert sequence[0] == '210'
        assert sequence.count('210') == 1


def test_session_allocation_handles_an_ephys_ngl_split(real_maze):
    """Both halves are allocated in one pass, so the second half cannot be
    stranded by nodes the first half consumed, and each half's distance rule is
    measured against its own goal."""
    for seed in range(20):
        allocation, islands, goals = allocate(real_maze, [('118', 15), ('205', 15)], seed=seed)
        assert allocation is not None, seed
        sequence = allocation.sequence
        assert len(set(sequence)) == 30

        for node, goal in zip(sequence, goals):
            assert get_shortest_distance(real_maze, node, goal) >= MIN_DISTANCE_FROM_GOAL
            assert real_maze.nodes[node]['group'] != real_maze.nodes[goal]['group']

        for i, a in enumerate(sequence):
            for b in sequence[i + 1:]:
                assert not real_maze.has_edge(a, b)


def test_session_allocation_prefers_avoiding_previous_session_nodes(real_maze):
    """Soft rule: honoured when possible, never a reason to fail."""
    avoid = {n for n in real_maze if real_maze.nodes[n]['group'] == 2}
    allocation, _, _ = allocate(real_maze, [('118', 20)], soft_avoid=avoid, seed=1)
    assert allocation is not None
    sequence = allocation.sequence

    reused = [n for n in sequence if n in avoid]
    island_2_trials = [n for n in sequence if real_maze.nodes[n]['group'] == 2]
    # Island 2 is still needed, so some overlap is unavoidable - but nothing
    # outside island 2 should have been dragged in.
    assert reused == island_2_trials


def test_session_allocation_reports_failure_instead_of_a_short_sequence(real_maze):
    """Goal 124 leaves island 2 with only 7 usable nodes, so 30 trials is
    impossible without help; the allocator must return None rather than a
    partial answer."""
    allocation, _, _ = allocate(real_maze, [('124', 30)], seed=0)
    assert allocation is None


# ---------------------------------------------------------------------------
# Reusing previous-session start nodes past 20 trials
# ---------------------------------------------------------------------------

def previous_session(real_maze, goal, count=20, seed=7):
    """A plausible previous session's start nodes, in trial order."""
    allocation, _, _ = allocate(real_maze, [(goal, 20)], seed=seed)
    assert allocation is not None
    return allocation.sequence[:count]


def legal_nodes_on(real_maze, island, goal):
    return [
        n for n in sorted(real_maze)
        if real_maze.nodes[n]['group'] == island
        and n not in HARD_EXCLUDED_NODES
        and n != goal
        and get_shortest_distance(real_maze, n, goal) >= MIN_DISTANCE_FROM_GOAL
    ]


def test_fallback_rescues_a_goal_that_cannot_supply_30_fresh_nodes(real_maze):
    """Goal 124 caps island 2 at 7 fresh non-adjacent nodes against a demand of
    10. Reused nodes are exempt from the spacing rules, so they cover the gap."""
    goal = '124'
    assert allocate(real_maze, [(goal, 30)], seed=0)[0] is None, \
        'expected this goal to be infeasible without help'

    fallback = legal_nodes_on(real_maze, 2, goal)[:10]

    for seed in range(8):
        allocation, _, _ = allocate(
            real_maze, [(goal, 30)], fallback_nodes=fallback, seed=seed,
        )
        assert allocation is not None, seed
        sequence = allocation.sequence
        assert len(sequence) == 30
        assert len(set(sequence)) == 30
        assert allocation.borrowed, 'expected some nodes to be borrowed'


def shortfall_islands(real_maze, goal, needed=10):
    """Islands that cannot supply ``needed`` fresh start nodes for this goal."""
    forbidden = set(HARD_EXCLUDED_NODES) | {goal}
    goal_island = real_maze.nodes[goal]['group']
    return {
        island for island in (1, 2, 3, 4)
        if island != goal_island
        and island_capacity(
            real_maze, island=island, goal_node=goal, forbidden_set=forbidden,
        ) < needed
    }


def test_a_real_previous_session_covers_the_bridge_goals(real_maze):
    """The shortfall lands on one specific island, and only about a quarter of a
    previous session's start nodes sit there - which is why the whole list is
    offered rather than just the first ten."""
    checked = 0
    for prev_goal in ('118', '205', '315', '410'):
        fallback = previous_session(real_maze, prev_goal)
        prev_island = real_maze.nodes[prev_goal]['group']

        for goal in ('121', '124', '401', '404'):
            # A session never starts on its own goal island, so it has no nodes
            # to offer a goal whose shortfall is on that same island.
            if prev_island in shortfall_islands(real_maze, goal):
                continue

            allocation, _, _ = allocate(
                real_maze, [(goal, 30)], fallback_nodes=fallback, seed=0,
            )
            assert allocation is not None, f'{goal} after a session on {prev_goal}'
            assert allocation.borrowed
            checked += 1

    assert checked >= 8, 'expected most pairings to be usable'


def test_fallback_is_only_used_when_fresh_nodes_run_out(real_maze):
    """A 20-trial session is always satisfiable, so nothing should be borrowed
    even when a fallback list is available."""
    fallback = previous_session(real_maze, '205')

    for seed in range(15):
        allocation, _, _ = allocate(real_maze, [('118', 20)], fallback_nodes=fallback, seed=seed)
        assert allocation is not None
        assert not allocation.borrowed, f'seed {seed} borrowed unnecessarily'


def test_borrowed_nodes_still_respect_distance_and_goal_island(real_maze):
    """Borrowed nodes are exempt from the spacing rules, but never from the
    >= 4 distance rule, the never-use list, or the goal-island rule."""
    goal = '124'
    fallback = legal_nodes_on(real_maze, 2, goal)[:10]

    for seed in range(8):
        allocation, _, goals = allocate(
            real_maze, [(goal, 30)], fallback_nodes=fallback, seed=seed,
        )
        assert allocation is not None
        sequence = allocation.sequence
        goal_island = real_maze.nodes[goal]['group']
        for node, trial_goal in zip(sequence, goals):
            assert node not in HARD_EXCLUDED_NODES
            assert real_maze.nodes[node]['group'] != goal_island
            assert get_shortest_distance(real_maze, node, trial_goal) >= MIN_DISTANCE_FROM_GOAL


def test_fallback_takes_nodes_from_the_top_of_the_list(real_maze):
    """Borrowing works down the previous session's list from the top, rather
    than picking freely."""
    goal = '124'
    fallback = legal_nodes_on(real_maze, 2, goal)[:10]

    allocation, _, _ = allocate(
        real_maze, [(goal, 30)], fallback_nodes=fallback, seed=0,
    )
    assert allocation is not None
    # Borrowing follows the fallback list's own order.
    assert allocation.borrowed == [n for n in fallback if n in set(allocation.borrowed)]


def test_compliance_exempts_borrowed_nodes_from_spacing_rules(real_maze):
    """Borrowed nodes sit wherever the previous session put them, so they must
    not be reported as hard adjacency failures."""
    goal = '124'
    fallback = legal_nodes_on(real_maze, 2, goal)[:10]
    allocation, _, goals = allocate(
        real_maze, [(goal, 30)], fallback_nodes=fallback, seed=0,
    )
    assert allocation is not None
    sequence = allocation.sequence
    reused = allocation.borrowed
    assert reused

    results = evaluate_protocol_compliance(
        real_maze, sequence, per_trial_goals=goals,
        prev_used_nodes=fallback, reused_nodes=reused,
        min_distance_from_goal=MIN_DISTANCE_FROM_GOAL,
    )

    hard_failures = [i['name'] for i in results if not i['passed'] and i['severity'] == 'hard']
    assert hard_failures == []

    # ... but the borrowing is surfaced as a soft finding to check by hand.
    flagged = [i for i in results if i['name'] == 'reused_previous_nodes_to_fill_shortfall']
    assert flagged and not flagged[0]['passed']
    assert flagged[0]['severity'] == 'soft'


# ---------------------------------------------------------------------------
# Probe / NGL trial 1
# ---------------------------------------------------------------------------

def test_probe_start_node_is_equidistant_and_on_a_connected_island(maze):
    for seed in range(30):
        random.seed(seed)
        node, info = select_probe_start_node(
            maze, goal_node='105', prev_goal_node='205',
            forbidden_set=set(HARD_EXCLUDED_NODES),
        )
        assert node is not None
        assert info['relaxed'] == []
        assert abs(info['dist_new'] - info['dist_old']) < 2
        assert info['dist_new'] >= MIN_DISTANCE_FROM_GOAL
        # Islands 1 and 2 are the goal islands; only island 3 touches both.
        assert maze.nodes[node]['group'] == 3


def test_probe_start_island_differs_from_previous_session_last_island(maze):
    """The trial-1 island rule is applied before the equidistance filter, so a
    pool that happens to sit on the previous last island cannot force a break."""
    for seed in range(30):
        random.seed(seed)
        node, info = select_probe_start_node(
            maze, goal_node='105', prev_goal_node='405',
            forbidden_set=set(HARD_EXCLUDED_NODES), prev_last_group=2,
        )
        assert node is not None
        assert maze.nodes[node]['group'] != 2
        assert 'first_trial_island_differs_from_prev_last' not in info['relaxed']


def test_probe_reports_when_a_rule_had_to_be_relaxed(maze):
    """Islands 2 and 3 are the only valid ones here; banning both leaves no
    choice, and the caller must be told rather than silently misled."""
    random.seed(0)
    node, info = select_probe_start_node(
        maze, goal_node='105', prev_goal_node='405',
        forbidden_set=set(HARD_EXCLUDED_NODES) | {f'2{i:02d}' for i in range(1, 25)},
        prev_last_group=3,
    )
    assert node is not None
    assert maze.nodes[node]['group'] == 3
    assert 'first_trial_island_differs_from_prev_last' in info['relaxed']


def test_probe_start_node_never_lands_on_a_goal_island(maze):
    for seed in range(20):
        random.seed(seed)
        node, _ = select_probe_start_node(
            maze, goal_node='105', prev_goal_node='405',
            forbidden_set=set(HARD_EXCLUDED_NODES),
        )
        assert node is not None
        # Goal islands are 1 and 4; islands 2 and 3 both touch each of them.
        assert maze.nodes[node]['group'] in (2, 3)


def test_ngl_first_trial_island_avoids_goal_and_previous_goal_islands(maze):
    for seed in range(40):
        random.seed(seed)
        group = choose_first_trial_group(
            maze, goal_group=3, available_groups=[1, 2, 3, 4],
            prev_goal_group=1, prev_last_group=2,
        )
        assert group == 4


def test_ngl_first_trial_prefers_avoiding_previous_first_island(maze):
    groups = set()
    for seed in range(40):
        random.seed(seed)
        groups.add(choose_first_trial_group(
            maze, goal_group=4, available_groups=[1, 2, 3, 4],
            prev_goal_group=1, prev_last_group=None, prev_first_group=2,
        ))
    assert groups == {3}


# ---------------------------------------------------------------------------
# get_shortest_distance
# ---------------------------------------------------------------------------

def test_get_shortest_distance_uses_cache_for_repeated_lookup():
    graph = nx.Graph()
    graph.add_edges_from([('101', '102'), ('102', '103')])

    cache = {}
    assert get_shortest_distance(graph, '101', '103', cache) == 2
    assert get_shortest_distance(graph, '101', '103', cache) == 2
    assert cache[('101', '103')] == 2
    assert cache[('103', '101')] == 2


def test_get_shortest_distance_reports_infinity_for_unreachable_nodes():
    graph = nx.Graph()
    graph.add_edges_from([('101', '102')])
    graph.add_node('999')

    assert get_shortest_distance(graph, '101', '999') == float('inf')
    assert get_shortest_distance(graph, '101', 'missing') == float('inf')


# ---------------------------------------------------------------------------
# Compliance report
# ---------------------------------------------------------------------------

def passed(results, name):
    return next(item['passed'] for item in results if item['name'] == name)


def names(results):
    return {item['name'] for item in results}


def test_compliance_accepts_a_clean_20_trial_session(real_maze):
    """End-to-end: what the generator produces passes every hard rule."""
    for seed in range(20):
        allocation, _, goals = allocate(real_maze, [('118', 20)], seed=seed)
        assert allocation is not None
        sequence = allocation.sequence

        results = evaluate_protocol_compliance(
            real_maze, sequence, per_trial_goals=goals,
            min_distance_from_goal=MIN_DISTANCE_FROM_GOAL,
        )
        hard_failures = [i['name'] for i in results if not i['passed'] and i['severity'] == 'hard']
        assert hard_failures == [], f'seed {seed}: {hard_failures}'


def test_compliance_accepts_a_clean_ephys_ngl_session(real_maze):
    """The 15/15 goal switch must not trip the balance or distance checks."""
    for seed in range(20):
        allocation, _, goals = allocate(real_maze, [('118', 15), ('205', 15)], seed=seed)
        assert allocation is not None
        sequence = allocation.sequence

        results = evaluate_protocol_compliance(
            real_maze, sequence, per_trial_goals=goals,
            min_distance_from_goal=MIN_DISTANCE_FROM_GOAL,
        )
        hard_failures = [i['name'] for i in results if not i['passed'] and i['severity'] == 'hard']
        assert hard_failures == [], f'seed {seed}: {hard_failures}'


def test_every_legal_goal_supports_a_20_trial_session(real_maze):
    """20 trials is what the protocol asks for, so no legal goal may fail."""
    goals = [n for n in sorted(real_maze) if n not in HARD_EXCLUDED_NODES]
    for goal in goals:
        allocation, _, _ = allocate(real_maze, [(goal, 20)], seed=int(goal))
        assert allocation is not None, goal
        assert not allocation.borrowed, f'{goal} should not need to borrow at 20 trials'


def test_island_capacity_explains_an_impossible_goal(real_maze):
    """Goal 124 is a bridge node: island 2 can only supply 7 legal start nodes,
    so 30 trials there is impossible while 20 is still fine."""
    forbidden = set(HARD_EXCLUDED_NODES) | {'124'}
    capacities = {
        island: island_capacity(
            real_maze, island=island, goal_node='124', forbidden_set=forbidden,
        )
        for island in (2, 3, 4)
    }
    assert min(capacities.values()) < 10   # 30 trials needs 10 per island
    assert min(capacities.values()) >= 7   # 20 trials needs 7, still fine


def test_compliance_flags_excluded_nodes_and_short_distances(maze):
    goal = '108'
    # 213 is on the protocol's never-use list; 301/302 are adjacent to each
    # other and both sit closer than 10 steps from the goal.
    sequence = ['301', '302', '213']

    results = evaluate_protocol_compliance(
        maze, sequence, per_trial_goals=[goal] * 3, min_distance_from_goal=10,
    )

    assert not passed(results, 'no_hard_excluded_nodes')
    assert not passed(results, 'distance_to_goal_at_least_min')
    assert not passed(results, 'start_nodes_not_adjacent')


def test_compliance_flags_goal_island_and_block_and_balance(maze):
    goal = '424'
    sequence = ['101', '103', '105', '401', '205', '305']

    results = evaluate_protocol_compliance(
        maze, sequence, per_trial_goals=[goal] * 6, min_distance_from_goal=1,
    )

    assert not passed(results, 'first_three_trials_use_different_islands')
    assert not passed(results, 'blocks_of_three_use_different_islands')
    assert not passed(results, 'island_counts_balanced')
    assert not passed(results, 'start_island_never_goal_island')


def test_compliance_uses_the_per_trial_goal_for_split_ngl_sessions(maze):
    """Regression: an ephys NGL session has two goals, but every trial used to be
    scored against the first one."""
    sequence = ['301', '303', '307']
    # Island 3 sits ~8-14 steps from island 1 but ~32-34 steps from island 2.
    correct = evaluate_protocol_compliance(
        maze, sequence, per_trial_goals=['208'] * 3, min_distance_from_goal=20,
    )
    scored_against_wrong_goal = evaluate_protocol_compliance(
        maze, sequence, per_trial_goals=['108'] * 3, min_distance_from_goal=20,
    )

    assert passed(correct, 'distance_to_goal_at_least_min')
    assert not passed(scored_against_wrong_goal, 'distance_to_goal_at_least_min')


def test_compliance_marks_previous_session_rules_as_soft(maze):
    goal = '424'
    sequence = ['105', '205', '320']

    results = evaluate_protocol_compliance(
        maze, sequence, per_trial_goals=[goal] * 3, min_distance_from_goal=1,
        prev_used_nodes=['105'], prev_goal_node='205',
    )

    severities = {item['name']: item['severity'] for item in results}
    assert severities['avoid_previous_session_start_nodes'] == 'soft'
    assert severities['avoid_previous_goal_node'] == 'soft'
    assert not passed(results, 'avoid_previous_session_start_nodes')
    assert not passed(results, 'avoid_previous_goal_node')


def test_compliance_only_compares_successive_visits_to_an_island(maze):
    """The protocol's example 101...117...120...103 ends 2 nodes from where it
    started, so a later return is fine; only back-to-back visits are checked."""
    goal = '424'

    # Island 1 visited at trials 1 and 4, far apart each time it comes round.
    ok = ['101', '201', '301', '117', '205', '305']
    assert passed(
        evaluate_protocol_compliance(
            maze, ok, per_trial_goals=[goal] * 6, min_distance_from_goal=1,
        ),
        'successive_same_island_nodes_dispersed',
    )

    # Same island twice in a row, only 2 nodes apart.
    crowded = ['101', '201', '301', '103', '205', '305']
    assert not passed(
        evaluate_protocol_compliance(
            maze, crowded, per_trial_goals=[goal] * 6, min_distance_from_goal=1,
        ),
        'successive_same_island_nodes_dispersed',
    )


def test_compliance_checks_probe_specific_rules_only_for_probe_sessions(maze):
    goal = '105'
    sequence = ['320', '322', '324']

    probe = evaluate_protocol_compliance(
        maze, sequence, per_trial_goals=[goal] * 3, session_kind='PT',
        prev_goal_node='205', probe_reference_goal='205', min_distance_from_goal=1,
    )
    normal = evaluate_protocol_compliance(
        maze, sequence, per_trial_goals=[goal] * 3, session_kind='Normal',
        prev_goal_node='205', min_distance_from_goal=1,
    )

    assert 'probe_start_equidistant_from_both_goals' in names(probe)
    assert 'probe_start_island_connected_to_both_goals' in names(probe)
    assert passed(probe, 'probe_start_equidistant_from_both_goals')
    assert passed(probe, 'probe_start_island_connected_to_both_goals')

    assert 'probe_start_equidistant_from_both_goals' not in names(normal)
    assert 'ngl_pt_first_trial_island_differs_from_prev_goal' not in names(normal)


def test_compliance_checks_first_trial_island_against_previous_session(maze):
    goal = '424'
    sequence = ['105', '205', '320']

    results = evaluate_protocol_compliance(
        maze, sequence, per_trial_goals=[goal] * 3, min_distance_from_goal=1,
        prev_last_group=1, prev_first_group=1,
    )

    assert not passed(results, 'first_trial_island_differs_from_prev_last')
    assert not passed(results, 'first_trial_island_differs_from_prev_first')
    severities = {item['name']: item['severity'] for item in results}
    assert severities['first_trial_island_differs_from_prev_last'] == 'hard'
    assert severities['first_trial_island_differs_from_prev_first'] == 'soft'
