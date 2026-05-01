from keyframe.scoring import (
    allocate_clusters_by_novelty,
    candidate_budget_for_scenes,
    coalesce_tiny_scenes,
)


def test_static_scene_gets_floor_eventful_scene_gets_surplus():
    scenes = [(0, 2), (3, 6)]
    dhashes = [7, 7, 7, 0, 255, 0, 255]

    allocs = allocate_clusters_by_novelty(scenes, 5, dhashes, floor=1)

    assert sum(allocs) == 5
    assert allocs[0] == 1
    assert allocs[1] == 4


def test_no_scene_below_floor_when_budget_allows():
    scenes = [(0, 1), (2, 3), (4, 5)]
    dhashes = [0, 1, 10, 11, 20, 21]

    allocs = allocate_clusters_by_novelty(scenes, 6, dhashes, floor=1)

    assert sum(allocs) == 6
    assert all(alloc >= 1 for alloc in allocs)


def test_tiny_scenes_coalesce_when_boundary_jump_is_small():
    scenes = [(0, 5), (6, 7), (8, 12)]
    timestamps = [float(i) for i in range(13)]
    dhashes = [0] * 13

    merged = coalesce_tiny_scenes(scenes, timestamps, dhashes)

    assert merged == [(0, 7), (8, 12)]


def test_tiny_scenes_stay_separate_when_boundary_jump_is_large():
    scenes = [(0, 5), (6, 7), (8, 12)]
    timestamps = [float(i) for i in range(13)]
    dhashes = [0] * 13
    dhashes[6] = (1 << 20) - 1

    merged = coalesce_tiny_scenes(scenes, timestamps, dhashes, boundary_hamming_threshold=18)

    assert merged == scenes


def test_candidate_budget_allows_bounded_scene_heavy_surplus():
    assert candidate_budget_for_scenes(15, 49) == 30
    assert candidate_budget_for_scenes(15, 20) == 20
    assert candidate_budget_for_scenes(15, 10) == 15
