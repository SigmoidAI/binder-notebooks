"""
Unit tests for the Interactive Visualization (Plotly) assignment.

This module contains test functions for each exercise that validate
the learner's implementation against expected outputs.
"""

from types import FunctionType
from typing import List

from unittests_utils import (
    CreateScatterBattery,
    CreateTimeSeriesBattery,
    CreateGroupedBarBattery,
    CustomizeLayoutBattery,
    SaveFigureHtmlBattery,
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
# Exercise 1 – create_scatter
# ---------------------------------------------------------------------------

def exercise_1(learner_func):
    """Test the create_scatter function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "create_scatter must be a function"
            return [t]
        cases.append(t)

        try:
            battery = CreateScatterBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_figure()
            if failed:
                t.failed = True
                t.msg = "create_scatter must return a plotly.graph_objects.Figure"
                t.got = f"Got type: {type(battery.result)}"
                t.want = "plotly.graph_objects.Figure"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_traces()
            if failed:
                t.failed = True
                t.msg = (
                    "The figure must contain at least one trace — "
                    "check that px.scatter is called and returns a figure with data"
                )
            cases.append(t)

            t = test_case()
            got, want, failed = battery.title_is_set()
            if failed:
                t.failed = True
                t.msg = (
                    "The figure title must be set (e.g. 'amount vs units') — "
                    "pass a 'title' argument to px.scatter"
                )
                try:
                    t.got = f"Got title: '{battery.result.layout.title.text}'"
                except Exception:
                    t.got = "Could not read title"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


# ---------------------------------------------------------------------------
# Exercise 2 – create_time_series
# ---------------------------------------------------------------------------

def exercise_2(learner_func):
    """Test the create_time_series function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "create_time_series must be a function"
            return [t]
        cases.append(t)

        try:
            battery = CreateTimeSeriesBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_figure()
            if failed:
                t.failed = True
                t.msg = "create_time_series must return a plotly.graph_objects.Figure"
                t.got = f"Got type: {type(battery.result)}"
                t.want = "plotly.graph_objects.Figure"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_traces()
            if failed:
                t.failed = True
                t.msg = (
                    "The figure must contain at least one trace — "
                    "check that px.line is called with the correct data"
                )
            cases.append(t)

            t = test_case()
            got, want, failed = battery.xaxis_title_is_set()
            if failed:
                t.failed = True
                t.msg = (
                    "The x-axis title must be set — "
                    "use fig.update_layout(xaxis_title=date_col)"
                )
                try:
                    t.got = f"Got xaxis title: '{battery.result.layout.xaxis.title.text}'"
                except Exception:
                    t.got = "Could not read xaxis title"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


# ---------------------------------------------------------------------------
# Exercise 3 – create_grouped_bar
# ---------------------------------------------------------------------------

def exercise_3(learner_func):
    """Test the create_grouped_bar function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "create_grouped_bar must be a function"
            return [t]
        cases.append(t)

        try:
            battery = CreateGroupedBarBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_figure()
            if failed:
                t.failed = True
                t.msg = "create_grouped_bar must return a plotly.graph_objects.Figure"
                t.got = f"Got type: {type(battery.result)}"
                t.want = "plotly.graph_objects.Figure"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.has_traces()
            if failed:
                t.failed = True
                t.msg = (
                    "The figure must contain at least one trace — "
                    "check that px.bar is called correctly"
                )
            cases.append(t)

            t = test_case()
            got, want, failed = battery.barmode_is_group()
            if failed:
                t.failed = True
                t.msg = (
                    "The figure barmode must be 'group' — "
                    "pass barmode='group' to px.bar or call fig.update_layout(barmode='group')"
                )
                try:
                    t.got = f"Got barmode: '{battery.result.layout.barmode}'"
                except Exception:
                    t.got = "Could not read barmode"
                t.want = "'group'"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


# ---------------------------------------------------------------------------
# Exercise 4 – customize_layout
# ---------------------------------------------------------------------------

def exercise_4(learner_func):
    """Test the customize_layout function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "customize_layout must be a function"
            return [t]
        cases.append(t)

        try:
            battery = CustomizeLayoutBattery(learner_func)

            t = test_case()
            got, want, failed = battery.result_is_figure()
            if failed:
                t.failed = True
                t.msg = "customize_layout must return a plotly.graph_objects.Figure"
                t.got = f"Got type: {type(battery.result)}"
                t.want = "plotly.graph_objects.Figure"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.title_matches()
            if failed:
                t.failed = True
                t.msg = (
                    f"The figure title must match the provided title argument — "
                    f"use fig.update_layout(title=title)"
                )
                try:
                    t.got = f"Got title: '{battery.result.layout.title.text}'"
                except Exception:
                    t.got = "Could not read title"
                t.want = f"'{battery.title}'"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.xaxis_title_matches()
            if failed:
                t.failed = True
                t.msg = (
                    "The x-axis title must match xaxis_title — "
                    "use fig.update_layout(xaxis_title=xaxis_title)"
                )
                try:
                    t.got = f"Got xaxis title: '{battery.result.layout.xaxis.title.text}'"
                except Exception:
                    t.got = "Could not read xaxis title"
                t.want = f"'{battery.xaxis_title}'"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.yaxis_title_matches()
            if failed:
                t.failed = True
                t.msg = (
                    "The y-axis title must match yaxis_title — "
                    "use fig.update_layout(yaxis_title=yaxis_title)"
                )
                try:
                    t.got = f"Got yaxis title: '{battery.result.layout.yaxis.title.text}'"
                except Exception:
                    t.got = "Could not read yaxis title"
                t.want = f"'{battery.yaxis_title}'"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)


# ---------------------------------------------------------------------------
# Exercise 5 – save_figure_html
# ---------------------------------------------------------------------------

def exercise_5(learner_func):
    """Test the save_figure_html function."""

    def g():
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "save_figure_html must be a function"
            return [t]
        cases.append(t)

        try:
            battery = SaveFigureHtmlBattery(learner_func)

            t = test_case()
            got, want, failed = battery.file_exists()
            if failed:
                t.failed = True
                t.msg = (
                    f"The HTML file was not created at '{battery.path}' — "
                    "use fig.write_html(path) to save the figure"
                )
                t.want = f"File exists at {battery.path}"
                t.got = "File not found"
            cases.append(t)

            t = test_case()
            got, want, failed = battery.file_is_not_empty()
            if failed:
                t.failed = True
                t.msg = "The saved HTML file is empty — ensure fig.write_html writes content"
                t.want = "Non-empty HTML file"
                t.got = "Empty file"
            cases.append(t)

        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"Exception raised: {e}"
            return cases + [t]

        return cases

    cases = g()
    print_feedback(cases)
