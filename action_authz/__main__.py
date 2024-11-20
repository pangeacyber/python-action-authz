from __future__ import annotations

import json
from typing import Any, override

import click
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam
from pangea import PangeaConfig
from pangea.services import AuthZ
from pangea.services.authz import Resource, Subject
from pydantic import SecretStr

load_dotenv(override=True)

TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Use this to search for information.",
            "parameters": {
                "type": "object",
                "properties": {"keywords": {"type": "string", "description": "Query keywords."}},
                "required": ["message"],
            },
        },
    }
]

SYSTEM_PROMPT: str = "Only call a tool once in a single message."


class SecretStrParamType(click.ParamType):
    name = "secret"

    @override
    def convert(self, value: Any, param: click.Parameter | None = None, ctx: click.Context | None = None) -> SecretStr:
        if isinstance(value, SecretStr):
            return value

        return SecretStr(value)


SECRET_STR = SecretStrParamType()


@click.command()
@click.option("--user", required=True, help="Unique username to simulate retrieval as.")
@click.option(
    "--authz-token",
    envvar="PANGEA_AUTHZ_TOKEN",
    type=SECRET_STR,
    required=True,
    help="Pangea AuthZ API token. May also be set via the `PANGEA_AUTHZ_TOKEN` environment variable.",
)
@click.option(
    "--pangea-domain",
    envvar="PANGEA_DOMAIN",
    default="aws.us.pangea.cloud",
    show_default=True,
    required=True,
    help="Pangea API domain. May also be set via the `PANGEA_DOMAIN` environment variable.",
)
@click.option("--model", default="gpt-4o-mini", show_default=True, required=True, help="OpenAI model.")
@click.option(
    "--openai-api-key",
    envvar="OPENAI_API_KEY",
    type=SECRET_STR,
    required=True,
    help="OpenAI API key. May also be set via the `OPENAI_API_KEY` environment variable.",
)
@click.argument("prompt")
def main(
    *,
    prompt: str,
    user: str,
    authz_token: SecretStr,
    pangea_domain: str,
    model: str,
    openai_api_key: SecretStr,
) -> None:
    authz = AuthZ(token=authz_token.get_secret_value(), config=PangeaConfig(domain=pangea_domain))
    subject = Subject(type="user", id=user)

    # Generate chat completions.
    completions = OpenAI(api_key=openai_api_key.get_secret_value()).chat.completions.create(
        messages=(
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ),
        model=model,
        tools=TOOLS,
        tool_choice="required",
    )
    tool_calls = completions.choices[0].message.tool_calls
    assert tool_calls
    for function_call in tool_calls:
        function_name = function_call.function.name
        function_args = json.loads(function_call.function.arguments)

        if function_name == "search":
            # Check if user is authorized to run this tool.
            response = authz.check(subject=subject, action="read", resource=Resource(type="duckduckgo"))
            if response.result is None or not response.result.allowed:
                click.echo(f"User {subject.id} is not authorized to use this tool.")
                return

            results = DDGS().answers(function_args["keywords"])
            click.echo(results[0]["text"])


if __name__ == "__main__":
    main()
