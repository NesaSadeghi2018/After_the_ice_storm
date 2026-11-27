
import numpy as np
import pytest

from ice_storm.optimization.algorithms.multiobjective_utils import (
    dominates,
    fast_non_dominated_sort,
    crowding_distance,
    environmental_selection,
)


def test_dominates_identifies_pareto_dominance():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 3.0, 4.0])

    assert dominates(a, b)
    assert not dominates(b, a)


def test_dominates_rejects_tradeoff():
    a = np.array([1.0, 4.0, 3.0])
    b = np.array([2.0, 3.0, 4.0])

    assert not dominates(a, b)
    assert not dominates(b, a)


def test_equal_objectives_do_not_dominate_each_other():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])

    assert not dominates(a, b)
    assert not dominates(b, a)


def test_dominates_rejects_incompatible_shapes():
    with pytest.raises(ValueError):
        dominates(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0, 3.0]),
        )


def test_fast_non_dominated_sort_identifies_fronts():
    objectives = np.array(
        [
            [1.0, 4.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ]
    )

    fronts = fast_non_dominated_sort(objectives)

    assert set(fronts[0]) == {0, 1, 2}
    assert set(fronts[1]) == {3}
    assert set(fronts[2]) == {4}


def test_fast_non_dominated_sort_handles_empty_input():
    objectives = np.empty((0, 3))

    assert fast_non_dominated_sort(objectives) == []


def test_crowding_distance_assigns_infinite_distance_to_boundaries():
    objectives = np.array(
        [
            [1.0, 5.0],
            [2.0, 4.0],
            [3.0, 3.0],
            [4.0, 2.0],
            [5.0, 1.0],
        ]
    )

    front = [0, 1, 2, 3, 4]

    distances = crowding_distance(objectives, front)

    assert np.isinf(distances[0])
    assert np.isinf(distances[-1])

    assert np.all(np.isfinite(distances[1:-1]))
    assert np.all(distances[1:-1] > 0.0)


def test_crowding_distance_handles_front_with_two_solutions():
    objectives = np.array(
        [
            [1.0, 3.0],
            [2.0, 2.0],
        ]
    )

    distances = crowding_distance(
        objectives,
        [0, 1],
    )

    assert np.all(np.isinf(distances))


def test_environmental_selection_preserves_population_size():
    population = np.arange(20, dtype=float).reshape(10, 2)

    objectives = np.array(
        [
            [1.0, 5.0],
            [2.0, 4.0],
            [3.0, 3.0],
            [4.0, 2.0],
            [5.0, 1.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 7.0],
            [8.0, 8.0],
        ]
    )

    selected_population, selected_objectives, indices = (
        environmental_selection(
            population,
            objectives,
            population_size=5,
        )
    )

    assert selected_population.shape == (5, 2)
    assert selected_objectives.shape == (5, 2)
    assert len(indices) == 5
    assert len(np.unique(indices)) == 5

    # All five members of the first Pareto front must be retained.
    assert set(indices) == {0, 1, 2, 3, 4}


def test_environmental_selection_rejects_oversized_population():
    population = np.zeros((3, 2))
    objectives = np.zeros((3, 3))

    with pytest.raises(ValueError):
        environmental_selection(
            population,
            objectives,
            population_size=4,
        )

def test_crowding_distance_ignores_objective_with_zero_range():
    objectives = np.array(
        [
            [1.0, 3.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 5.0],
            [1.0, 4.0],
        ]
    )

    front = [0, 1, 2, 3, 4]

    distances = crowding_distance(objectives, front)

    # Only the boundaries of the varying second objective
    # should receive infinite crowding distance.
    assert np.isinf(distances[1])
    assert np.isinf(distances[3])

    # The constant first objective must not create
    # additional boundary solutions.
    assert np.all(np.isfinite(distances[[0, 2, 4]]))

    assert np.allclose(
        distances[[0, 2, 4]],
        [0.5, 0.5, 0.5],
    )
def test_environmental_selection_handles_duplicate_objectives():
    population = np.array(
        [
            [10.0, 1.0],
            [20.0, 2.0],
            [30.0, 3.0],
            [40.0, 4.0],
            [50.0, 5.0],
        ]
    )

    objectives = np.array(
        [
            [1.0, 5.0],
            [1.0, 5.0],
            [2.0, 4.0],
            [3.0, 3.0],
            [4.0, 2.0],
        ]
    )

    selected_population, selected_objectives, indices = (
        environmental_selection(
            population,
            objectives,
            population_size=4,
        )
    )

    assert len(indices) == 4
    assert len(np.unique(indices)) == 4

    assert selected_population.shape == (4, 2)
    assert selected_objectives.shape == (4, 2)

    # Identical objective vectors must not dominate each other.
    assert not dominates(
        objectives[0],
        objectives[1],
    )

    assert not dominates(
        objectives[1],
        objectives[0],
    )

    # Both duplicate-objective candidates belong to the first front.
    assert 0 in indices
    assert 1 in indices