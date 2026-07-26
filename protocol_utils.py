import random
from typing import List, Optional

import networkx as nx


def build_island_sequence(
    *,
    total_trials: int,
    available_groups: List[int],
    goal_group: int,
    forced_t1_group: Optional[int] = None,
    prev_last_group: Optional[int] = None,
    is_ngl_pt: bool = False,
) -> List[int]:
    """Create an island sequence that follows the protocol's block structure.

    The protocol calls for:
    - first three trials to use different islands
    - block size of three, each block using three different islands
    - start island should avoid the goal island and respect previous-session history
    - counts across the three non-goal islands should be balanced (about 7/7/6)
    """
    if not available_groups:
        return []

    groups = [g for g in available_groups if g != goal_group]
    if not groups:
        return []

    island_sequence: List[int] = []

    # Trial 1 is handled separately when a forced start island exists.
    if forced_t1_group is not None:
        island_sequence.append(forced_t1_group)
        groups = [g for g in groups if g != forced_t1_group]

    # Build first three trials with distinct islands.
    first_block = []
    pool = list(groups)
    random.shuffle(pool)

    if not is_ngl_pt and prev_last_group is not None:
        for _ in range(50):
            if pool and pool[0] != prev_last_group:
                break
            random.shuffle(pool)

    for g in pool:
        if len(first_block) >= 3:
            break
        if g not in first_block:
            first_block.append(g)

    if len(first_block) < 3:
        first_block = [g for g in groups if g not in first_block]

    island_sequence.extend(first_block)

    while len(island_sequence) < total_trials:
        block = []
        pool = list(groups)
        random.shuffle(pool)
        for g in pool:
            if len(block) >= 3:
                break
            if g not in block:
                block.append(g)
        if len(block) < 3:
            block = [g for g in groups if g not in block]
        island_sequence.extend(block)

    return island_sequence[:total_trials]


def is_topologically_valid_group(graph: nx.Graph, target_group: int, *, prev_goal_group: Optional[int], current_goal_group: int) -> bool:
    """Return True if the target island is a suitable bridge for the goal transition.

    In this maze layout, the protocol prefers the middle island (3) when the goal
    switches between islands 1 and 2, because it keeps the start island equally distant
    from both goals and avoids introducing an extra topological hop.
    """
    if prev_goal_group is None or current_goal_group is None:
        return True

    if target_group in {prev_goal_group, current_goal_group}:
        return False

    if {prev_goal_group, current_goal_group} == {1, 2}:
        return target_group == 3

    return True


def select_start_node(
    graph: nx.Graph,
    *,
    target_group: int,
    current_selected: set,
    forbidden_set: set,
    goal_node: str,
    prev_goal_node: Optional[str],
    prev_goal_group: Optional[int],
    current_goal_group: int,
    min_distance_from_goal: int,
) -> Optional[str]:
    """Pick the most protocol-compliant node from the target island."""
    candidates = [
        n for n, attrs in graph.nodes(data=True)
        if attrs.get('group') == target_group and n not in forbidden_set and n not in current_selected
    ]

    if not candidates:
        return None

    valid = []
    for node in candidates:
        try:
            if nx.shortest_path_length(graph, source=node, target=goal_node) < min_distance_from_goal:
                continue
        except nx.NetworkXNoPath:
            continue

        if prev_goal_node:
            try:
                prev_dist = nx.shortest_path_length(graph, source=node, target=prev_goal_node)
                curr_dist = nx.shortest_path_length(graph, source=node, target=goal_node)
            except nx.NetworkXNoPath:
                prev_dist = curr_dist = float('inf')
            if prev_dist != float('inf') and curr_dist != float('inf'):
                if abs(prev_dist - curr_dist) > 2:
                    continue
        
        if not is_topologically_valid_group(graph, target_group, prev_goal_group=prev_goal_group, current_goal_group=current_goal_group):
            continue

        neighbours = set(graph.neighbors(node))
        if any(neighbor in current_selected for neighbor in neighbours):
            continue

        valid.append(node)

    if not valid:
        valid = candidates

    # Prefer nodes that are farther from existing selected nodes within the same island.
    if valid:
        scored = []
        for node in valid:
            dist_to_selected = []
            for selected in current_selected:
                try:
                    dist_to_selected.append(nx.shortest_path_length(graph, source=node, target=selected))
                except nx.NetworkXNoPath:
                    dist_to_selected.append(10)
            avg_distance = sum(dist_to_selected) / max(1, len(dist_to_selected))
            node_num = int(node[1:]) if node[1:].isdigit() else 0
            scored.append((avg_distance, -node_num, node))
        scored.sort(reverse=True)
        best_score = scored[0][0]
        best_nodes = [node for score, _, node in scored if score == best_score]
        return random.choice(best_nodes)

    return None


def evaluate_protocol_compliance(
    graph: nx.Graph,
    sequence: List[str],
    *,
    goal_node: str,
    prev_goal_node: Optional[str],
    prev_goal_group: Optional[int],
    current_goal_group: int,
    prev_used_nodes: List[str],
    goal_group: int,
    is_ngl_pt: bool,
    prev_last_group: Optional[int],
    min_distance_from_goal: int,
) -> List[dict]:
    """Return a list of protocol checks with pass/fail status for a generated sequence."""
    checks = []

    if not sequence:
        return checks

    # 1. Start island should not be the goal island.
    start_group = graph.nodes[sequence[0]]['group']
    passed = start_group != goal_group
    checks.append({'name': 'start_island_not_goal', 'passed': passed, 'details': f'start island {start_group} vs goal island {goal_group}'})

    # 2. First three trials should use different islands.
    first_three_groups = [graph.nodes[node]['group'] for node in sequence[:3]]
    passed = len(set(first_three_groups)) == 3
    checks.append({'name': 'first_three_trials_use_different_islands', 'passed': passed, 'details': first_three_groups})

    # 3. Each trial should be at least min_distance_from_goal away from current goal.
    distance_checks = []
    for node in sequence:
        try:
            dist = nx.shortest_path_length(graph, source=node, target=goal_node)
        except nx.NetworkXNoPath:
            dist = float('inf')
        distance_checks.append(dist)
    passed = all(dist >= min_distance_from_goal for dist in distance_checks if dist != float('inf'))
    checks.append({'name': 'distance_to_goal', 'passed': passed, 'details': distance_checks})

    # 4. Avoid previous session start nodes and the previous goal node if possible.
    previous_conflicts = [node for node in sequence if node in prev_used_nodes or node == prev_goal_node]
    passed = not previous_conflicts
    checks.append({'name': 'avoid_previous_session_nodes', 'passed': passed, 'details': f'conflicts={previous_conflicts}'})

    # 5. Start island should differ from the previous session's last island when possible.
    if prev_last_group is not None and not is_ngl_pt:
        passed = start_group != prev_last_group
        checks.append({'name': 'start_island_differs_from_prev_last', 'passed': passed, 'details': f'start_group={start_group}, prev_last_group={prev_last_group}'})

    return checks
