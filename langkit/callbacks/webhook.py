import json
from dataclasses import asdict
from typing import List, Mapping

import pandas as pd
import requests

from langkit.core.metric import MetricResult
from langkit.core.validation import ValidationResult
from langkit.core.workflow import Callback


class Webhook(Callback):
    def __init__(self, url: str, include_input: bool = False):
        self.url = url
        self.include_input = include_input

    def post_validation(
        self,
        df: pd.DataFrame,
        metric_results: Mapping[str, MetricResult],
        results: pd.DataFrame,
        validation_results: List[ValidationResult],
    ) -> None:
        if not validation_results:
            return

        if not validation_results:
            return

        body = {
            "metrics": results.to_dict(orient="records"),  # pyright: ignore[reportUnknownMemberType]
            "report": [asdict(it) for it in validation_results],
        }

        if self.include_input:
            body["input"] = df.to_dict(orient="records")  # pyright: ignore[reportUnknownMemberType]

        requests.post(self.url, json=body)


class BearerAuthValidationWebhook(Callback):
    def __init__(
        self, url: str, auth_header: str = "Authorization", auth_token: str = "", bearer_prefix: str = "Bearer", include_input: bool = False
    ):
        self.url = url
        self.auth_header = auth_header
        self.auth_token = auth_token
        self.bearer_prefix = bearer_prefix
        self.include_input = include_input

    def post_validation(
        self,
        df: pd.DataFrame,
        metric_results: Mapping[str, MetricResult],
        results: pd.DataFrame,
        validation_results: List[ValidationResult],
    ) -> None:
        if not validation_results:
            return

        body = {
            "metrics": results.to_dict(orient="records"),  # pyright: ignore[reportUnknownMemberType]
            "report": [asdict(it) for it in validation_results],
        }

        if self.include_input:
            body["input"] = df.to_dict(orient="records")  # pyright: ignore[reportUnknownMemberType]

        prefix = f"{self.bearer_prefix} " if self.bearer_prefix else ""
        headers = {self.auth_header: f"{prefix}{self.auth_token}"}
        requests.post(self.url, json=body, headers=headers)


class SlackValidationWebhook(Callback):
    def __init__(self, url: str):
        self.url = url

    def post_validation(
        self,
        df: pd.DataFrame,
        metric_results: Mapping[str, MetricResult],
        results: pd.DataFrame,
        validation_results: List[ValidationResult],
    ) -> None:
        if not validation_results:
            return
        # Convert validation results to JSON with pretty formatting
        formatted_validation_results = json.dumps([asdict(result) for result in validation_results], indent=4)

        # Construct the Slack message payload
        message = f"Failed Validation Results:\n\n```{formatted_validation_results}```"
        body = json.dumps({"text": message})

        requests.post(self.url, data={"payload": body})
