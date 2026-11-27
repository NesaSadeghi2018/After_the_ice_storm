
import sys

import frostline_runner


def run_main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["frostline_runner.py", *args])
    frostline_runner.main()


def test_assignment_a_is_dispatched(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        frostline_runner,
        "run_assignment_a",
        lambda **kwargs: calls.append(("A", kwargs)),
    )

    run_main(
        monkeypatch,
        "--assignment", "A",
        "--runs", "2",
        "--max-evaluations", "500",
        "--output-dir", str(tmp_path),
    )

    assert len(calls) == 1
    assert calls[0][0] == "A"
    assert calls[0][1]["runs"] == 2
    assert calls[0][1]["max_evaluations"] == 500


def test_assignment_b_is_dispatched(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        frostline_runner,
        "run_assignment_b",
        lambda **kwargs: calls.append(("B", kwargs)),
    )

    run_main(
        monkeypatch,
        "--assignment", "B",
        "--output-dir", str(tmp_path),
    )

    assert len(calls) == 1
    assert calls[0][0] == "B"


def test_assignment_c_is_dispatched(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        frostline_runner,
        "run_assignment_c",
        lambda **kwargs: calls.append(("C", kwargs)),
    )

    run_main(
        monkeypatch,
        "--assignment", "C",
        "--output-dir", str(tmp_path),
    )

    assert len(calls) == 1
    assert calls[0][0] == "C"



def test_all_assignments_are_dispatched_in_order(monkeypatch, tmp_path):
    calls = []

    for assignment in ("a", "b", "c"):
        monkeypatch.setattr(
            frostline_runner,
            f"run_assignment_{assignment}",
            lambda assignment=assignment, **kwargs: calls.append(
                assignment.upper()
            ),
        )

    run_main(
        monkeypatch,
        "--assignment", "all",
        "--output-dir", str(tmp_path),
    )

    assert calls == ["A", "B", "C"]



def test_dry_run_does_not_dispatch_experiments(monkeypatch, tmp_path):
    calls = []

    for assignment in ("a", "b", "c"):
        monkeypatch.setattr(
            frostline_runner,
            f"run_assignment_{assignment}",
            lambda **kwargs: calls.append(assignment.upper()),
        )

    run_main(
        monkeypatch,
        "--assignment", "all",
        "--dry-run",
        "--output-dir", str(tmp_path),
    )

    assert calls == []
