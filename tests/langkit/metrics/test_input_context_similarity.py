from typing import List

import pandas as pd
import pytest

from langkit.core.workflow import InputContext, Workflow, is_input_context
from langkit.metrics.library import lib


def test_similarity():
    wf = Workflow(metrics=[lib.prompt.similarity.context()])

    context: InputContext = {
        "entries": [
            {"content": "Some source 1", "metadata": {"source": "https://internal.com/foo"}},
            {"content": "Some source 2", "metadata": {"source": "https://internal.com/bar"}},
        ]
    }

    df = pd.DataFrame({"prompt": ["Some source"], "context": [context]})

    result = wf.run(df)

    metrics = result.metrics

    metric_names: List[str] = metrics.columns.tolist()  # pyright: ignore[reportUnknownMemberType]

    assert metric_names == ["prompt.similarity.context", "id"]
    assert metrics["prompt.similarity.context"][0] == pytest.approx(0.7447172999382019)  # pyright: ignore[reportUnknownMemberType]


def test_similarity_repoonse():
    wf = Workflow(metrics=[lib.response.similarity.context()])

    context: InputContext = {
        "entries": [
            {"content": "Some source 1", "metadata": {"source": "https://internal.com/foo"}},
            {"content": "Some source 2", "metadata": {"source": "https://internal.com/bar"}},
        ]
    }

    df = pd.DataFrame({"response": ["Some source"], "context": [context]})

    result = wf.run(df)

    metrics = result.metrics

    metric_names: List[str] = metrics.columns.tolist()  # pyright: ignore[reportUnknownMemberType]

    assert metric_names == ["response.similarity.context", "id"]
    assert metrics["response.similarity.context"][0] == pytest.approx(0.7447172999382019)  # pyright: ignore[reportUnknownMemberType]


def test_similarity_missing_context():
    # The metric should not be run in this case since the context is missing
    wf = Workflow(metrics=[lib.prompt.similarity.context()])

    df = pd.DataFrame({"prompt": ["Some source"]})

    result = wf.run(df)

    metrics = result.metrics

    metric_names: List[str] = metrics.columns.tolist()  # pyright: ignore[reportUnknownMemberType]

    assert metric_names == ["id"]


def test_similarity_multiple():
    wf = Workflow(metrics=[lib.prompt.similarity.context()])

    context: InputContext = {
        "entries": [
            {"content": "Some source 1", "metadata": {"source": "https://internal.com/foo"}},
            {"content": "Some source 2", "metadata": {"source": "https://internal.com/bar"}},
        ]
    }

    df = pd.DataFrame({"prompt": ["Some source", "Nothing in common"], "context": [context, context]})

    result = wf.run(df)

    metrics = result.metrics

    metric_names: List[str] = metrics.columns.tolist()  # pyright: ignore[reportUnknownMemberType]

    assert metric_names == ["prompt.similarity.context", "id"]
    assert metrics["prompt.similarity.context"][0] == pytest.approx(0.7447172999382019)  # pyright: ignore[reportUnknownMemberType]
    assert metrics["prompt.similarity.context"][1] < 0.2


def test_similarity_row():
    wf = Workflow(metrics=[lib.prompt.similarity.context()])

    context: InputContext = {
        "entries": [
            {"content": "Some source 1", "metadata": {"source": "https://internal.com/foo"}},
            {"content": "Some source 2", "metadata": {"source": "https://internal.com/bar"}},
        ]
    }

    assert is_input_context(context)

    result = wf.run({"prompt": "Some source", "context": context})

    metrics = result.metrics

    metric_names: List[str] = metrics.columns.tolist()  # pyright: ignore[reportUnknownMemberType]

    assert metric_names == ["prompt.similarity.context", "id"]
    assert metrics["prompt.similarity.context"][0] == pytest.approx(0.7447172999382019)  # pyright: ignore[reportUnknownMemberType]
