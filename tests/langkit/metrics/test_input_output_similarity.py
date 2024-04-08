# pyright: reportUnknownMemberType=none

import pandas as pd

from langkit.core.workflow import Workflow
from langkit.metrics.input_output_similarity import input_output_similarity_metric, prompt_response_input_output_similarity_metric


def test_input_output():
    df = pd.DataFrame(
        {
            "prompt": [
                "I'm going to ask you a question",
            ],
            "response": [
                "I'm going to answer that question!",
            ],
        }
    )

    wf = Workflow(metrics=[input_output_similarity_metric])

    actual = wf.run(df)

    expected_columns = ["response.similarity.prompt", "id"]

    assert actual.metrics.columns.tolist() == expected_columns
    assert actual.metrics["response.similarity.prompt"].tolist()[0] > 0.5


def test_input_output_row():
    row = {
        "prompt": "I'm going to ask you a question",
        "response": "I'm going to answer that question!",
    }

    wf = Workflow(metrics=[prompt_response_input_output_similarity_metric])

    actual = wf.run(row)

    expected_columns = ["response.similarity.prompt", "id"]

    assert actual.metrics.columns.tolist() == expected_columns
    assert actual.metrics["response.similarity.prompt"].tolist()[0] > 0.5


def test_input_output_multiple():
    df = pd.DataFrame(
        {
            "prompt": [
                "If you understand me, say 'yes'",
                "Ask me a question",
            ],
            "response": [
                "yes",
                "the sky is eating",
            ],
        }
    )

    wf = Workflow(metrics=[prompt_response_input_output_similarity_metric])

    actual = wf.run(df)

    expected_columns = ["response.similarity.prompt", "id"]

    assert actual.metrics.columns.tolist() == expected_columns
    assert actual.metrics["response.similarity.prompt"].tolist()[0] > 0
    assert actual.metrics["response.similarity.prompt"].tolist()[1] < 0.5
