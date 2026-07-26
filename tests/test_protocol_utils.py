import random
from collections import Counter

import networkx as nx

from protocol_utils import (
    build_island_sequence,
    evaluate_protocol_compliance,
    get_shortest_distance,
    is_topologically_valid_group,
    select_start_node,
)


def test_build_island_sequence_uses_three_island_blocks_and_balances_counts():
    random.seed(7)
    sequence = build_island_sequence(
        total_trials=20,
        available_groups=[1, 2, 3],
        goal_group=4,
        forced_t1_group=None,
        prev_last_group=None,
        is_ngl_pt=False,
    )

    assert len(sequence) == 20
    assert len(set(sequence[:3])) == 3

    for start in range(0, len(sequence), 3):
        block = sequence[start:start + 3]
        if len(block) == 3:
            assert len(set(block)) == 3

    counts = Counter(sequence)
    assert counts[1] in {6, 7}
    assert counts[2] in {6, 7}
    assert counts[3] in {6, 7}


def test_is_topologically_valid_group_prefers_equal_distance_islands():
    graph = nx.Graph()
    graph.add_node('a', group=1)
    graph.add_node('b', group=2)
    graph.add_node('c', group=3)
    graph.add_node('d', group=4)
    graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd')])

    assert is_topologically_valid_group(graph, 3, prev_goal_group=2, current_goal_group=1)
    assert not is_topologically_valid_group(graph, 4, prev_goal_group=2, current_goal_group=1)


def test_select_start_node_prefers_dispersion_and_distance():
    graph = nx.Graph()
    for node_id, group in [('101', 1), ('102', 1), ('103', 1), ('201', 2), ('202', 2), ('203', 2)]:
        graph.add_node(node_id, group=group)
    graph.add_edges_from([('101', '102'), ('102', '103'), ('201', '202'), ('202', '203')])
    graph.add_edge('102', '201')

    selected = select_start_node(
        graph,
        target_group=1,
        current_selected={'201'},
        forbidden_set=set(),
        goal_node='203',
        prev_goal_node='201',
        prev_goal_group=2,
        current_goal_group=2,
        min_distance_from_goal=2,
    )

    assert selected == '101'


def test_get_shortest_distance_uses_cache_for_repeated_lookup():
    graph = nx.Graph()
    graph.add_edges_from([('101', '102'), ('102', '103')])

    distance_cache = {}
    assert get_shortest_distance(graph, '101', '103', distance_cache) == 2
    assert get_shortest_distance(graph, '101', '103', distance_cache) == 2
    assert distance_cache[('101', '103')] == 2


def test_evaluate_protocol_compliance_reports_sequence_checks():
    graph = nx.Graph()
    for node_id, group in [('101', 1), ('102', 1), ('103', 1), ('201', 2), ('202', 2), ('203', 2), ('301', 3), ('302', 3), ('303', 3)]:
        graph.add_node(node_id, group=group)
    graph.add_edges_from([('101', '102'), ('102', '103'), ('201', '202'), ('202', '203'), ('301', '302'), ('302', '303')])

    sequence = ['101', '201', '301', '103', '202', '303']
    results = evaluate_protocol_compliance(
        graph,
        sequence,
        goal_node='203',
        prev_goal_node='201',
        prev_goal_group=2,
        current_goal_group=2,
        prev_used_nodes=[],
        goal_group=2,
        is_ngl_pt=False,
        prev_last_group=1,
        min_distance_from_goal=2,
    )

    assert any(item['name'] == 'start_island_not_goal' and item['passed'] for item in results)
    assert any(item['name'] == 'first_three_trials_use_different_islands' and item['passed'] for item in results)
    assert any(item['name'] == 'distance_to_goal' and not item['passed'] for item in results)
