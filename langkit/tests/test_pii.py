import pandas as pd
import pytest
from whylogs.experimental.core.udf_schema import udf_schema


@pytest.fixture
def prompts():
    prompts = [
        "Hello, my name is David Johnson and I live in Maine. \
        My credit card number is 4095-2609-9393-4932 and my crypto wallet id is 16Yeky6GMjeNkAiNcBY7ZhrLoMSgg1BoyZ.",
        "On September 18 I visited microsoft.com and sent an email to test@presidio.site,  from the IP 192.168.0.1.",
        "My passport: 191280342 and my phone number: (212) 555-1234.",
        "This is a valid International Bank Account Number: IL150120690000003111111 . \
        Can you please check the status on bank account 954567876544?",
        "Kate's social security number is 078-05-1126.  Her driver license? it is 1234567A.",
        "Hi, My name is John.",
    ]
    return prompts


@pytest.mark.load
def test_row_presidio_pii(prompts):
    from langkit import extract, pii  # noqa

    schema = udf_schema()

    data = {"prompt": prompts[0], "response": prompts[-1]}
    result = extract(
        data,
        schema=schema,
    )
    assert "CRYPTO" in result["prompt.pii_presidio.result"]
    assert "CREDIT_CARD" in result["prompt.pii_presidio.result"]
    assert result["prompt.pii_presidio.entities_count"] == 2
    assert result["response.pii_presidio.result"] == "[]"
    assert result["response.pii_presidio.entities_count"] == 0


@pytest.mark.load
def test_pandas_presidio_pii(prompts):
    from langkit import extract, pii  # noqa

    schema = udf_schema()
    data = pd.DataFrame({"prompt": prompts, "response": prompts})
    result = extract(
        data,
        schema=schema,
    )
    assert result.shape == (6, 6)
    assert result["prompt.pii_presidio.entities_count"].to_list() == [2, 3, 5, 3, 3, 0]
    print(result)
