from langkit.core.workflow import Callback


class lib:
    class webhook:
        @staticmethod
        def basic_validation_failure(url: str, include_input: bool = False) -> Callback:
            from langkit.callbacks.webhook import Webhook

            return Webhook(url, include_input)

        @staticmethod
        def static_bearer_auth_validation_failure(
            url: str, auth_token: str, auth_header: str = "Authorization", bearer_prefix: str = "Bearer", include_input: bool = False
        ) -> Callback:
            from langkit.callbacks.webhook import BearerAuthValidationWebhook

            return BearerAuthValidationWebhook(url, auth_header, auth_token, bearer_prefix, include_input)

        @staticmethod
        def slack_validation_failure(url: str) -> Callback:
            from langkit.callbacks.webhook import SlackValidationWebhook

            return SlackValidationWebhook(url)
