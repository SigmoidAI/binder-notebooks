"""
Unit tests for the Matplotlib Fundamentals assignment.

This module contains test functions for each exercise that validate
the learner's implementation against expected outputs.
"""

from types import FunctionType
from typing import List

from unittests_utils import (
    CreateSubplotsBattery,
    AddLineChartBattery,
    AddAnnotationBattery,
    ApplyStylingBattery,
    SaveFigureBattery,
)


class test_case:
    """Simple container for a single test result."""

    def __init__(self):
        self.failed = False
        self.msg = ""
        self.want = ""
        self.got = ""


def print_feedback(cases: List[test_case]) -> None:
    """Print colour-coded pass / fail feedback for a list of test cases."""
    failed_cases = [c for c in cases if c.failed]

    if not failed_cases:
        print("\033[92m" + "All tests passed!" + "\033[0m")
    else:
        print(
            "\033[91m"
            + f"Failed {len(failed_cases)} of {len(cases)} tests:"
            + "\033[0m"
        )
        for case in failed_cases:
            print(f"\n  - {case.msg}")
            if case.want:
                print(f"     Expected: {case.want}")
            if case.got:
                print(f"     Got:      {case.got}")


# ---------------------------------------------------------------------------
# Exercise 1 – create_subplots
# ---------------------------------------------------------------------------

def exercise_1(learner_func):
    """Test the create_subplots function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "create_subplots must be a function"
            return [t]
        cases.append(t)

        try:
            battery = CreateSubplotsBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_tuple()
            if failed:
                t.failed = True
                t.msg = "create_subplots must return a tuple (fig, axes)"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.result_has_figure()
            if failed:
                t.failed = True
                t.msg = "First element of the returned tuple must be a matplotlib Figure"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.axes_has_correct_length()
            if failed:
                t.failed = True
                t.msg = "Second element (axes) must have the correct number of entries"
                t.want = f"{want} axes"
                t.got = f"{got} axes"
            cases.append(t)

            battery.cleanup()

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


# ---------------------------------------------------------------------------
# Exercise 2 – add_line_chart
# ---------------------------------------------------------------------------

def exercise_2(learner_func):
    """Test the add_line_chart function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "add_line_chart must be a function"
            return [t]
        cases.append(t)

        try:
            battery = AddLineChartBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_axes()
            if failed:
                t.failed = True
                t.msg = "add_line_chart must return the Axes object"
                t.got = f"Got type: {type(battery.result)}"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.ax_has_line()
            if failed:
                t.failed = True
                t.msg = "The Axes must contain at least one plotted line"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.title_is_set()
            if failed:
                t.failed = True
                t.msg = "The Axes title was not set correctly"
                t.want = battery.title
                t.got = str(battery.ax.get_title())
            cases.append(t)

            t = test_case()
            got, want, failed = battery.xlabel_is_set()
            if failed:
                t.failed = True
                t.msg = "The x-axis label was not set correctly"
                t.want = battery.xlabel
                t.got = str(battery.ax.get_xlabel())
            cases.append(t)

            battery.cleanup()

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


# ---------------------------------------------------------------------------
# Exercise 3 – add_peak_annotation
# ---------------------------------------------------------------------------

def exercise_3(learner_func):
    """Test the add_peak_annotation function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "add_peak_annotation must be a function"
            return [t]
        cases.append(t)

        try:
            battery = AddAnnotationBattery(learner_func)

            t = test_case()
            got, want, failed = battery.ax_has_annotation()
            if failed:
                t.failed = True
                t.msg = (
                    "add_peak_annotation must add at least one annotation "
                    "(use ax.annotate(...))"
                )
            cases.append(t)

            battery.cleanup()

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


# ---------------------------------------------------------------------------
# Exercise 4 – apply_clean_styling
# ---------------------------------------------------------------------------

def exercise_4(learner_func):
    """Test the apply_clean_styling function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "apply_clean_styling must be a function"
            return [t]
        cases.append(t)

        try:
            battery = ApplyStylingBattery(learner_func)

            t = test_case()
            got, want, failed = battery.right_spine_hidden()
            if failed:
                t.failed = True
                t.msg = "The right spine should be hidden after apply_clean_styling(ax)"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.top_spine_hidden()
            if failed:
                t.failed = True
                t.msg = "The top spine should be hidden after apply_clean_styling(ax)"
            cases.append(t)

            battery.cleanup()

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


# ---------------------------------------------------------------------------
# Exercise 5 – save_figure
# ---------------------------------------------------------------------------

def exercise_5(learner_func):
    """Test the save_figure function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "save_figure must be a function"
            return [t]
        cases.append(t)

        try:
            battery = SaveFigureBattery(learner_func)

            t = test_case()
            got, want, failed = battery.file_exists()
            if failed:
                t.failed = True
                t.msg = (
                    f"save_figure did not create the output file at '{battery.path}'"
                )
            cases.append(t)

            battery.cleanup()

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)
