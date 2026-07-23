#!/usr/bin/env python3
import asyncio
from collections.abc import AsyncGenerator
import copy
import dataclasses
import enum
import json
import os
import sys

import dotenv
import httpx
from prompt_toolkit import PromptSession
from transformers import AutoTokenizer

LLAMA_CPP_API_URL: str = "http://llama_cpp:8080/v1/chat/completions"
LLAMA_CPP_API_KEY: str = dotenv.get_key(dotenv.find_dotenv(), "LLAMA_CPP_API_KEY")
GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY: str = dotenv.get_key(dotenv.find_dotenv(), "GROQ_API_KEY")
HF_API_URL: str = "https://router.huggingface.co/v1/chat/completions"
HF_API_KEY: str = dotenv.get_key(dotenv.find_dotenv(), "HF_TOKEN")
OPENROUTER_API_URL: str = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY: str = dotenv.get_key(dotenv.find_dotenv(), "OPENROUTER_API_KEY")
NVIDIA_API_URL: str = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_API_KEY: str = dotenv.get_key(dotenv.find_dotenv(), "NVIDIA_API_KEY")
LITELLM_API_URL: str = "http://litellm:4000/v1/chat/completions"
LITELLM_API_KEY: str = dotenv.get_key(dotenv.find_dotenv(), "LITELLM_API_KEY")

# MODEL_ID = "openai/gpt-oss-20b"
# TOKENIZER_MODEL_ID = "openai/gpt-oss-20b"
MODEL_ID = "z-ai/glm-5.2"
TOKENIZER_MODEL_ID = "zai-org/GLM-5.2"
# MODEL_ID = "poolside/laguna-xs-2.1"
# TOKENIZER_MODEL_ID = "poolside/Laguna-XS-2.1"
TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID)
MODEL_IDS_WITH_TOOL_ARGUMENTS_AS_DICTS = {
    "zai-org/GLM-5.2",
    "poolside/Laguna-XS-2.1",
}

ARTIFACT_DIR = "/artifacts"

MAX_CONTEXT_TOKENS: int = 131_072
# Aim well below the hard cap so the model has room to produce a long response
# AND so the next tool result doesn't immediately trigger another trim.
TRIM_TARGET_TOKENS: int = int(MAX_CONTEXT_TOKENS * 0.75)

headers = {
    "Content-Type": "application/json",
}


class Provider(enum.Enum):
    LLAMA_CPP = "llama_cpp"
    GROQ = "groq"
    HF = "hf"
    OPENROUTER = "openrouter"
    NVIDIA = "nvidia"
    LITELLM = "litellm"


@dataclasses.dataclass(frozen=True)
class ProviderAPIConfig:
    api_url: str
    api_key: str


PROVIDER_API_CONFIG = {
    Provider.LLAMA_CPP: ProviderAPIConfig(LLAMA_CPP_API_URL, LLAMA_CPP_API_KEY),
    Provider.GROQ: ProviderAPIConfig(GROQ_API_URL, GROQ_API_KEY),
    Provider.HF: ProviderAPIConfig(HF_API_URL, HF_API_KEY),
    Provider.OPENROUTER: ProviderAPIConfig(OPENROUTER_API_URL, OPENROUTER_API_KEY),
    Provider.NVIDIA: ProviderAPIConfig(NVIDIA_API_URL, NVIDIA_API_KEY),
    Provider.LITELLM: ProviderAPIConfig(LITELLM_API_URL, LITELLM_API_KEY),
}


def get_provider_api_config() -> ProviderAPIConfig:
    return PROVIDER_API_CONFIG[provider]


def set_provider(new_provider: Provider):
    global headers, provider
    provider = new_provider
    headers["Authorization"] = f"Bearer {get_provider_api_config().api_key}"


provider = Provider(os.getenv("PROVIDER") or Provider.NVIDIA.value)
set_provider(provider)


# Local Python Functions (The real tools on your system)
async def bash_tool(command: str):
    result = await asyncio.create_subprocess_shell(
        command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, stderr = await result.communicate()
    return {
        "data": stdout.decode().strip(),
        "metadata": {
            "success": result.returncode == 0,
            "returncode": result.returncode,
        },
    }


# Define the Tool Schemas (What the AI reads)
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "bash_tool",
            "description": (
                "Execute a shell command in the container to run code, access system utilities, or manage environment resources. "
                "Call this tool when fulfilling the request requires external execution or live system interaction. \n"
                "\n"
                "Rather than returning the raw output directly, the tool replies with a JSON summary that tells you where the full output was saved, whether the command exited cleanly, its exit code, its MIME type, line and byte counts, and a truncated preview of the output (indicated by <TRUNCATED>). "
                "The preview is only meant for quick inspection. "
                "When you need to see more than the preview, use a selective command that extracts only what you need from the saved artifact path - for example with grep, sed, awk, jq, head, or tail. "
                "Note that every command only returns a truncated preview, so non-selective commands like cat will simply produce another low-signal summary. You must use a filtering tool to actually see the content and will never receive the raw file contents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    }
]
TOOL_NAMES: set[str] = {tool["function"]["name"] for tool in TOOL_SCHEMAS}


# Send the initial user message + the tools list to the model
SYSTEM_MESSAGES = [
    {
        "role": "system",
        "content": """\
You are an autonomous agent.

Your task is complete only when the user's request has been successfully accomplished.

Use the tools at your disposal to gather information, perform actions, and achieve the user's goal.
Validate tool output before continuing.

Do not assume what can be observed.

When uncertain, gather information rather than speculate.

Your memory is unreliable and outdated. Every fact you need must be verified through tool use before you act on it or ask the user. The user is only to be consulted when your tools cannot provide the answer.

All facts must be taken directly from a tool's output; you may never answer, act, or make a decision based on any internal knowledge that has not been freshly obtained and confirmed by that tool.

If later actions depend on earlier actions, observe the outcome before proceeding.

Errors are observations, not conclusions.

Continue working toward the goal until:
- the task is complete,
- you determine it cannot be completed, or
- you require information that only the user can provide.

Never hand the task back to the user for them to complete. You are the agent; you must complete the task yourself.

Do not end early because of time, token, effort, or detail concerns; continue until the task is complete or a real stopping condition is met.

Explain failures only after exhausting reasonable attempts to recover.

The user's request grants permission to perform actions that are reasonably necessary to accomplish it.

Prefer removing obstacles to reporting obstacles.

Do not ask the user to perform actions that you can perform yourself.

Only seek user input when information, resources, or decisions are required that only the user can provide.
""",
    },
    {
        "role": "system",
        "content": (
            "System implementation notes:\n"
            "The system is running in a Debian Docker container as root. \n"
        ),
    },
]
messages = copy.deepcopy(SYSTEM_MESSAGES)


def get_tokenizer_compatible_messages(messages):
    if TOKENIZER_MODEL_ID in MODEL_IDS_WITH_TOOL_ARGUMENTS_AS_DICTS:
        messages = copy.deepcopy(messages)
        for assistant_message in messages:
            if assistant_message.get("tool_calls"):
                for tool_call in assistant_message["tool_calls"]:
                    arguments = tool_call["function"].get("arguments")
                    if isinstance(arguments, str):
                        tool_call["function"]["arguments"] = json.loads(arguments)
    return messages


def count_tokens(messages):
    """
    Count the exact number of tokens that would be sent to the model
    for the given conversation history using the model's tokenizer.
    """
    encoded = TOKENIZER.apply_chat_template(
        get_tokenizer_compatible_messages(messages),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
    )
    token_count = len(encoded["input_ids"])
    print(f"🧩 🧮 | Token count: {token_count}")
    print()
    return token_count


async def trim_context() -> None:
    """
    Trim the global ``messages`` list to TRIM_TARGET_TOKENS when it exceeds
    MAX_CONTEXT_TOKENS.

    Strategy
    --------
    Phase 1 - Drop complete old exchanges, oldest first.
        An "exchange" is one user message plus every assistant/tool turn that
        follows, up to (but not including) the next user message. The final
        exchange (the current in-progress request) is never dropped.

    Phase 2 - Drop inner tool-call groups from the surviving exchange.
        If the current exchange alone is still too large, iteratively remove
        its oldest (assistant-with-tool_calls + tool-result) block. The user
        message and any final plain-text assistant reply are left untouched.
    """
    global messages

    token_count = count_tokens(messages)
    if token_count <= MAX_CONTEXT_TOKENS:
        return

    print(
        f"🗜️  | {token_count}/{MAX_CONTEXT_TOKENS} tokens - trimming to <= {TRIM_TARGET_TOKENS}"
    )

    # ── helpers ────────────────────────────────────────────────────────────────

    def exchange_slices(msgs: list[dict]) -> list[tuple[int, int]]:
        """
        Return [(start, end), ...] where each half-open slice covers one exchange.

        System messages sit before the first user message and are naturally
        excluded because we index by user-message positions.
        """
        user_pos = [i for i, m in enumerate(msgs) if m["role"] == "user"]
        return [
            (
                start,
                user_pos[k + 1] if k + 1 < len(user_pos) else len(msgs),
            )
            for k, start in enumerate(user_pos)
        ]

    def tool_call_groups(msgs: list[dict], lo: int, hi: int) -> list[tuple[int, int]]:
        """
        Within msgs[lo:hi] find each atomic (assistant-with-tool_calls +
        following tool messages) block. Returns absolute indices into msgs.

        An assistant message without tool_calls (plain text) is NOT a group;
        neither is a bare user message. We never yield those, so callers
        never accidentally drop them.
        """
        groups: list[tuple[int, int]] = []
        i = lo
        while i < hi:
            if msgs[i]["role"] == "assistant" and msgs[i].get("tool_calls"):
                j = i + 1
                while j < hi and msgs[j]["role"] == "tool":
                    j += 1
                groups.append((i, j))
                i = j
            else:
                i += 1
        return groups

    # ── Phase 1: drop complete old exchanges ───────────────────────────────────

    exchanges = exchange_slices(messages)
    while count_tokens(messages) > TRIM_TARGET_TOKENS and len(exchanges) > 1:
        start, end = exchanges[0]
        print(
            f"🗜️  | dropping exchange [{start}:{end}] "
            f"roles={[m['role'] for m in messages[start:end]]}"
        )
        del messages[start:end]
        exchanges = exchange_slices(messages)  # recalculate; indices shifted

    if count_tokens(messages) <= TRIM_TARGET_TOKENS:
        print(f"🗜️  | done after Phase 1 - {count_tokens(messages)} tokens")
        return

    # ── Phase 2: current exchange is still too large ───────────────────────────

    print("🗜️  | Phase 2 - dropping inner tool groups from current exchange...")
    exchanges = exchange_slices(messages)
    if not exchanges:
        return

    ex_start, ex_end = exchanges[0]  # only one exchange remains

    while count_tokens(messages) > TRIM_TARGET_TOKENS:
        groups = tool_call_groups(messages, ex_start, ex_end)

        if not groups:
            # All that's left is the user message (and maybe a plain assistant
            # reply).  We cannot trim further without destroying the request.
            print("🗜️  | no tool groups left - context cannot be reduced further")
            break

        g_start, g_end = groups[0]
        print(
            f"🗜️  | dropping tool group [{g_start}:{g_end}] "
            f"roles={[m['role'] for m in messages[g_start:g_end]]}"
        )
        del messages[g_start:g_end]
        ex_end -= g_end - g_start  # keep window consistent after in-place delete

    print(f"🗜️  | done after Phase 2 - {count_tokens(messages)} tokens")


async def save_artifacts(tool_call_id: str, tool_result: dict) -> str:
    data_path = f"{ARTIFACT_DIR}/{tool_call_id}.log"
    metadata_path = f"{ARTIFACT_DIR}/{tool_call_id}.metadata.json"

    with open(data_path, "wb") as f:
        f.write(
            tool_result["data"].encode()
            if isinstance(tool_result["data"], str)
            else tool_result["data"]
        )

    async def run(command: str) -> str:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        return stdout.decode().strip()

    data = tool_result["data"]
    output_str = (
        data.decode(errors="replace") if isinstance(data, bytes) else (data or "")
    )
    preview = (
        output_str
        if len(output_str) <= 2000
        else (output_str[:1000] + "<TRUNCATED>" + output_str[-1000:])
    )
    generic_metadata = {
        "preview": preview,
        "mime-type": await run(f"file --brief --mime-type '{data_path}'"),
        "line-count": int(await run(f"wc -l < '{data_path}'")),
        "byte-count": int(await run(f"wc -c < '{data_path}'")),
    }
    tool_message_content = {
        "artifact_path": data_path,
        "metadata": {**generic_metadata, **tool_result["metadata"]},
    }

    with open(metadata_path, "w") as f:
        json.dump(tool_message_content["metadata"], f)
    return json.dumps(tool_message_content)


async def chat_with_agent(user_message_content) -> AsyncGenerator[dict, None]:
    global messages
    global provider
    messages.append({"role": "user", "content": user_message_content})

    # Keep looping as long as the model wants to call tools
    while True:
        await trim_context()

        model = f"{MODEL_ID}:free" if provider == Provider.OPENROUTER else MODEL_ID
        payload = {
            "model": model,
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "tool_choice": "auto",
        }

        # print("Sending messages for inference...")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                get_provider_api_config().api_url, headers=headers, json=payload, timeout=None
            )

        response_data = response.json()

        status_code = response.status_code
        is_success = 200 <= status_code < 300
        if is_success:
            error = response_data.get("error")
            if error:
                status_code = error.get("code")
                is_success = 200 <= status_code < 300

        if not is_success:
            match status_code:
                case 400:
                    print(
                        "⚠️  🔄 | Bad Request, feeding error details to agent for correction..."
                    )
                    print()
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "The server rejected the previous request with a 400 Bad Request error. "
                                "Review the raw server response below to identify what went wrong, "
                                "correct the issue, and try again:\n\n"
                                f"```json\n{response.text}\n```"
                            ),
                        }
                    )
                    continue
                case 429:
                    retry_after = response.headers.get("Retry-After")
                    rate_limit_duration = 10
                    try:
                        rate_limit_duration = float(retry_after)
                    except (TypeError, ValueError):
                        pass
                    print(
                        f"🛑 ⏳ | Rate limited, retrying after {rate_limit_duration:.1f} seconds..."
                    )
                    print()
                    await asyncio.sleep(rate_limit_duration)
                    continue
                case 413:
                    print(
                        "Request entity too large, compacting message history and retrying..."
                    )
                    print()
                    await asyncio.sleep(10)
                    # TODO
                    continue
                case 500 | 502 | 503 | 504:
                    print(
                        f"⚠️  🔄 | {response.reason_phrase}, retrying after 10 seconds..."
                    )
                    print()
                    await asyncio.sleep(10)
                    continue
                case _:
                    print(
                        f"⚠️  | Unexpected HTTP error ({status_code}): {response.text}"
                    )
                    print()
                    response.raise_for_status()

        # Extract the assistant message dictionary
        assistant_message = response_data["choices"][0]["message"]

        # Normalize provider output to prevent Jinja chat template errors
        if not assistant_message.get("content"):
            assistant_message["content"] = ""
        if not assistant_message.get("tool_calls"):
            assistant_message.pop("tool_calls", None)
        if not assistant_message.get("reasoning"):
            assistant_message.pop("reasoning", None)
        if not assistant_message.get("reasoning_content"):
            assistant_message.pop("reasoning_content", None)

        tool_calls = assistant_message.get("tool_calls")

        # We must append the model's exact response containing the tool request to history
        yield assistant_message
        messages.append(assistant_message)

        finish_reason = response_data["choices"][0].get("finish_reason")
        stop_reason = response_data["choices"][0].get("stop_reason")
        # 2000212 == <|call|>
        # https://developers.openai.com/cookbook/articles/openai-harmony#special-tokens
        if finish_reason == "stop" and stop_reason == 200012:
            print(
                "🩹 🔄 | Agent triggered <|call|> but omitted tool payload. Injecting error message..."
            )
            print()
            response_data["choices"][0]["finish_reason"] = "tool_calls"
            messages.append(
                {
                    "role": "system",
                    "content": "You halted to make a tool call but did not provide the command payload. Please emit the tool call parameters for the action you just described.",
                }
            )
            continue
        # Break condition: If there are no tool calls, this is your final response to the user
        if not tool_calls:
            return
        else:
            # Handle the tool calls if the model requested them

            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                if "<|" in function_name:
                    print(
                        "🩹 ✂️  | Stripping Harmony channel tokens from function name for compatibility with tool execution"
                    )
                    print()
                    function_name = function_name.partition("<|")[0].strip()
                if (
                    function_name.endswith("commentary")
                    and (stripped_function_name := function_name[: -len("commentary")])
                    in TOOL_NAMES
                ):
                    print(
                        "🩹 ✂️  | Stripping 'commentary' suffix from function name for compatibility with tool execution"
                    )
                    print()
                    function_name = stripped_function_name
                tool_call["function"]["name"] = function_name
                if function_name not in TOOL_NAMES:
                    print(
                        f"⚠️  🔄 | Unrecognized tool call: '{function_name}', injecting error message..."
                    )
                    print()
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": function_name,
                        "content": json.dumps(
                            {
                                "success": False,
                                "error": "The model requested a tool that is not defined in the system. Please review the tool definitions and ensure the model only calls valid tools.",
                                "received_function_name": function_name,
                                "allowed_tools": list(TOOL_NAMES),
                            }
                        ),
                    }

                    yield tool_message
                    messages.append(tool_message)
                    continue
                try:
                    function_args = json.loads(tool_call["function"]["arguments"])
                except json.decoder.JSONDecodeError as e:
                    function_args = {}
                    print(
                        f"⚠️  🔄 | Failed to decode JSON arguments for tool call: {function_name}, feeding error details to agent for correction..."
                    )
                    print()
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": function_name,
                        "content": json.dumps(
                            {
                                "success": False,
                                "error": (
                                    "Tool arguments were not valid JSON. "
                                    f"{e.msg} at line {e.lineno}, column {e.colno}."
                                ),
                                "received_arguments": tool_call["function"][
                                    "arguments"
                                ],
                            }
                        ),
                    }

                    yield tool_message
                    messages.append(tool_message)
                    continue

                print(
                    f"🕵️  🛠️  | Processing tool call: {function_name} with arguments {function_args}"
                )
                print()

                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": function_name,
                    "content": "",
                }
                match function_name:
                    case "bash_tool":
                        tool_result = await bash_tool(
                            command=function_args.get("command")
                        )
                    case _:
                        tool_result = {
                            "data": None,
                            "metadata": {
                                "success": False,
                                "error": (
                                    f"The function name '{function_name}' is not recognized. "
                                    "Please review your tool definitions and retry using only the "
                                    "exact tool names provided in your configuration."
                                ),
                            },
                        }

                tool_message["content"] = await save_artifacts(
                    tool_call["id"], tool_result
                )

                # Append the execution result to history using the tool role and call ID
                yield tool_message
                messages.append(tool_message)


async def main(argv):
    global messages

    session = PromptSession()

    while True:
        user_message_content = await session.prompt_async("👤 💬 | ", multiline=True)
        print()
        if user_message_content.lower() in ["/exit", "/quit"]:
            break
        elif user_message_content.lower() in ["/clear"]:
            print("🧹 📜 | Clearing conversation history...")
            print()
            messages = copy.deepcopy(SYSTEM_MESSAGES)
            continue
        elif user_message_content.lower().startswith("/provider "):
            provider_name = user_message_content.split(maxsplit=1)[1].strip().lower()
            try:
                provider = Provider(provider_name)
                set_provider(provider)
                print(f"🎚️  🔀 | Switched provider to {provider_name}")
                print()
            except ValueError:
                print(
                    f"🎚️  ⚠️  | Unknown provider: '{provider_name}'. "
                    f"Valid options are: {', '.join(p.value for p in Provider)}"
                )
                print()
            continue

        async for message in chat_with_agent(user_message_content):
            match message["role"]:
                case "assistant":
                    reasoning_content = message.get(
                        "reasoning_content", message.get("reasoning")
                    )
                    message_content = message.get("content")
                    # Agent reasoning
                    if (
                        reasoning_content is not None
                        and reasoning_content.strip() != ""
                    ):
                        print(f"🕵️  🧠 | {reasoning_content}")
                        print()
                    # Agent message
                    if message_content is not None and message_content.strip() != "":
                        print(f"🕵️  💬 | {message_content}")
                        print()
                case "tool":
                    tool_content = message["content"]
                    print(
                        f"🕵️  🛠️  | Tool '{message['name']}' returned: {tool_content}"
                    )
                    print()
                case _:
                    print(f"❓ | Received message: {message}")
                    print()
                    raise ValueError(f"Unknown message role: {message['role']}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv))

# docker compose up --build --watch
# docker compose attach agent

# docker compose run -it --rm --service-ports agent
