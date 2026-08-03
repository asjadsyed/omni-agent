#!/usr/bin/env python3
import asyncio
import copy
import dataclasses
import enum
import json
import os
import sys
from collections.abc import AsyncGenerator

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

STATE_FILE: str = f"{ARTIFACT_DIR}/agent_state.json"
COMPACT_THRESHOLD: int = int(MAX_CONTEXT_TOKENS * 0.85)
COMPACT_KEEP_TOKENS: int = int(MAX_CONTEXT_TOKENS * 0.40)

FLUSH_NUDGE = "flush_nudge"
EXCHANGE_SUMMARY = "exchange_summary"
COMPACTION_SUMMARY = "compaction_summary"
STATE_HINT = "state_hint"
ERROR_INJECTION = "error_injection"
PROTECTED_TYPES = {EXCHANGE_SUMMARY, COMPACTION_SUMMARY, STATE_HINT}

MAX_FLUSH_WAIT_TURNS = 4

_emergency_ceiling: int | None = None

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

INFERENCE_TIMEOUT = httpx.Timeout(
    connect=10.0,
    write=120.0,
    read=600.0,
    pool=15.0,
)


def get_provider_api_config() -> ProviderAPIConfig:
    return PROVIDER_API_CONFIG[provider]


def set_provider(new_provider: Provider):
    global provider
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
    stdout, _stderr = await result.communicate()
    return {
        "data": stdout.decode(errors="replace").strip(),
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


def count_tokens(messages, verbose=True):
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
    if verbose:
        print(f"🧩 🧮 | Token count: {token_count}")
        print()
    return token_count


async def trim_context(*, force: bool = False) -> None:
    """
    All context management in one place. Call at the top of every agent loop iteration.

    Stage 1  >= 85%  STATE_FILE missing -> request state save, return early.
                       After MAX_FLUSH_WAIT_TURNS agent turns without it, compact anyway.
    Stage 2  >= 85%  LLM compaction: summarize old messages, keep recent tail verbatim.
    Stage 3          Structural trim: exchanges -> tool groups -> ephemeral msgs -> offloads -> drops.
                       Fires at TRIM_TARGET if compaction failed, MAX_CONTEXT if it succeeded.

    force=True skips the normal compaction flow and immediately performs
    structural trimming. Used after a 413 response so the next retry always
    sends a smaller request.
    """

    # ── helpers ───────────────────────────────────────────────────────────────

    async def call_llm(payload, max_tokens=2048):
        async with httpx.AsyncClient() as c:
            r = await c.post(
                get_provider_api_config().api_url,
                headers=headers,
                json={
                    "model": MODEL_ID,
                    "messages": payload,
                    "max_tokens": max_tokens,
                },
                timeout=120,
            )
        r.raise_for_status()
        return r.json()["choices"][0]["message"].get("content", "")

    def serialise(msgs):
        parts = []
        for msg in msgs:
            role = msg.get("role", "?")
            if role == "assistant":
                body = []
                if reasoning := (
                    msg.get("reasoning_content") or msg.get("reasoning", "")
                ).strip():
                    body.append(f"<reasoning>{reasoning}</reasoning>")
                if content := (msg.get("content") or "").strip():
                    body.append(content)
                for tool_call in msg.get("tool_calls") or []:
                    body.append(
                        f"<call id={tool_call['id']}>{tool_call['function']['name']}({tool_call['function']['arguments']})</call>"
                    )
                parts.append("[ASSISTANT]\n" + "\n".join(body))
            elif role == "tool":
                parts.append(
                    f"[TOOL id={msg.get('tool_call_id','?')} name={msg.get('name','?')}]\n{(msg.get('content') or '').strip()}"
                )
            else:
                parts.append(f"[{role.upper()}]\n{(msg.get('content') or '').strip()}")
        return "\n\n---\n\n".join(parts)

    async def summarize(head):
        prompt = [
            {
                "role": "system",
                "content": (
                    "Summarize this agent session so the agent can resume without re-doing work. "
                    "Include: GOAL, COMPLETED STEPS (outcomes), DISCOVERED FACTS (exact paths/values/IDs), "
                    "ERRORS & RECOVERY, REMAINING WORK (ordered), KEY ARTIFACTS. "
                    "Past tense. No commentary. Preserve exact values."
                ),
            },
            {"role": "user", "content": "Summarize:\n\n" + serialise(head)},
        ]
        try:
            return (await call_llm(prompt)).strip()
        except httpx.HTTPError as e:
            print(f"🗜️  🪓 | Summarization failed ({e}) - using truncated fallback")
            print()
            serialised = serialise(head)
            return (
                serialised[:1500]
                + f"\n\n[…{len(serialised)-3000}c omitted…]\n\n"
                + serialised[-1500:]
                if len(serialised) > 3000
                else serialised
            )

    def compact_boundary(rest, keep_tokens):
        """
        Split index so rest[boundary:] fits within keep_tokens.
        Boundary always lands on a user message, so every candidate tail
        is a structurally valid conversation for count_tokens.
        The last user message (the current request) is always protected.
        """
        user_positions = [i for i, msg in enumerate(rest) if msg["role"] == "user"]
        if not user_positions:
            return 0

        last_user = user_positions[-1]
        boundary = last_user  # fallback: keep only the last exchange

        # Walk oldest -> newest (excluding the current request).
        # Tails shrink monotonically as positions get more recent, so the
        # first position whose tail fits is the oldest valid split -
        # maximising how much history we keep verbatim.
        for pos in user_positions[:-1]:
            if count_tokens(rest[pos:]) <= keep_tokens:
                boundary = pos
                break

        return boundary

    def get_exchanges():
        user_positions = [i for i, msg in enumerate(messages) if msg["role"] == "user"]
        return [
            (
                start,
                user_positions[k + 1] if k + 1 < len(user_positions) else len(messages),
            )
            for k, start in enumerate(user_positions)
        ]

    def get_tool_groups(start, end):
        groups: list[tuple[int, int]] = []
        i = start
        while i < end:
            if messages[i]["role"] == "assistant" and messages[i].get("tool_calls"):
                j = i + 1
                while j < end and messages[j]["role"] == "tool":
                    j += 1
                groups.append((i, j))
                i = j
            else:
                i += 1
        return groups

    def state_file_valid():
        try:
            with open(STATE_FILE) as f:
                return bool(json.load(f))
        except (OSError, json.JSONDecodeError, ValueError):
            return False

    # ── main logic ────────────────────────────────────────────────────────────

    token_count = count_tokens(messages)
    tokens_before = token_count
    will_trim_context: bool = force or token_count >= COMPACT_THRESHOLD
    try:
        if not will_trim_context:
            return

        # ── stage 1: request state save / stage 2: LLM compaction (both >= 85%) ─

        compaction_succeeded = False
        if not force:
            if not state_file_valid():
                nudge_indices = [
                    i
                    for i, msg in enumerate(messages)
                    if msg.get("_type") == FLUSH_NUDGE and msg["role"] == "system"
                ]
                if nudge_indices:
                    turns_since_nudge = sum(
                        1
                        for msg in messages[nudge_indices[-1] :]
                        if msg["role"] == "assistant"
                    )
                    if turns_since_nudge < MAX_FLUSH_WAIT_TURNS:
                        return
                    # Delete the stale FLUSH_NUDGE message. Otherwise, the next time
                    # the context grows beyond COMPACT_THRESHOLD, we'd find the same FLUSH_NUDGE
                    # message and skip asking the agent to flush its state.
                    messages[:] = [
                        msg
                        for msg in messages
                        if not (
                            msg.get("_type") == FLUSH_NUDGE and msg["role"] == "system"
                        )
                    ]
                    print(
                        "🗜️  💨 | Agent did not save state - compacting without state file"
                    )
                    print()
                else:
                    messages.append(
                        {
                            "role": "system",
                            "_type": FLUSH_NUDGE,
                            "content": (
                                f"⚠️ Context {token_count}/{MAX_CONTEXT_TOKENS} tokens - nearing capacity. "
                                f"Call bash_tool to write working state to `{STATE_FILE}` as JSON with keys: "
                                f"goal, completed, discovered, errors, remaining, artifacts. "
                                f"File persists through compaction. Then continue normally."
                            ),
                        }
                    )
                    print("🗜️  📡 | Requesting state save before compaction")
                    print()
                    return

            original_contents = {msg.get("content", "") for msg in SYSTEM_MESSAGES}
            system_header = [
                msg
                for msg in messages
                if msg["role"] == "system"
                and msg.get("content", "") in original_contents
            ]
            rest = [
                msg
                for msg in messages
                if not (
                    msg["role"] == "system"
                    and msg.get("content", "") in original_contents
                )
            ]
            boundary = compact_boundary(rest, COMPACT_KEEP_TOKENS)
            head, tail = rest[:boundary], rest[boundary:]
            if head and any(msg["role"] != "system" for msg in head):
                head_tokens = count_tokens(head)
                print(
                    f"🗜️  🤏 | Compacting {head_tokens} tokens of history, keeping {count_tokens(tail)} tokens"
                )
                print()
                summarizable = [
                    msg for msg in head if msg.get("_type") != ERROR_INJECTION
                ]
                summary = {
                    "role": "system",
                    "_type": COMPACTION_SUMMARY,
                    "content": (
                        "## Compacted History\n\n" + await summarize(summarizable)
                    ),
                }
                if count_tokens([summary]) < head_tokens:
                    messages.clear()
                    messages.extend(system_header + [summary] + tail)
                    messages.append(
                        {
                            "role": "system",
                            "_type": STATE_HINT,
                            "content": f"Working state is at `{STATE_FILE}` - read it if you need context from before compaction.",
                        }
                    )
                    compaction_succeeded = True
                else:
                    print(
                        "🗜️  🤏 | Compaction skipped: summary was not smaller than the original"
                    )
                    print()

        # ── stage 3: structural trim ───────────────────────────────────────────
        # We determine the target token limit based on the current state.
        # If a 413 error forced this trim, we aggressively cut to 75% of the active ceiling
        # to ensure there is room for a response. Otherwise, if the LLM successfully
        # compacted the history, we are lenient and allow the full maximum limit to avoid
        # unnecessary destructive trimming. If compaction failed, we fall back to a deep
        # structural cut down to our 75% target, creating a buffer so the agent can run
        # for many turns before hitting the limit again.

        system_floor = count_tokens(SYSTEM_MESSAGES)
        current_ceiling = _emergency_ceiling or MAX_CONTEXT_TOKENS
        candidate_ceiling = current_ceiling
        effective_ceiling = max(candidate_ceiling, system_floor)
        effective_target = (
            effective_ceiling
            if force
            else (MAX_CONTEXT_TOKENS if compaction_succeeded else TRIM_TARGET_TOKENS)
        )
        token_count = count_tokens(messages)
        if token_count <= effective_target:
            return
        print(
            f"🗜️  ✂️  | Over limit ({token_count}/{effective_target} target) - running structural trim"
        )
        print()

        # Phase 1: offload old exchanges to disk with concurrent LLM summaries.
        # Only processes enough exchanges to cover the token deficit.
        exchanges = get_exchanges()
        if len(exchanges) > 1 and count_tokens(messages) > effective_target:
            token_deficit = count_tokens(messages) - effective_target
            accumulated, old_exchanges = 0, []
            for start, end in exchanges[:-1]:
                old_exchanges.append((start, end))
                accumulated += count_tokens(messages[start:end])
                if accumulated > token_deficit:
                    break

            offload_start = sum(
                1
                for f in os.listdir(ARTIFACT_DIR)
                if f.startswith("exchange_") and f.endswith(".json")
            )
            tasks = []
            for i, (start, end) in enumerate(old_exchanges):
                content = [
                    msg for msg in messages[start:end] if msg["role"] != "system"
                ]
                sys_msgs = [
                    msg for msg in messages[start:end] if msg["role"] == "system"
                ]
                path = f"{ARTIFACT_DIR}/exchange_{offload_start + i:06d}.json"
                try:
                    with open(path, "w") as f:
                        json.dump(content, f)
                    tasks.append((start, end, content, sys_msgs, path))
                except (OSError, TypeError) as e:
                    print(
                        f"🗜️  ⚠️  | Could not write exchange to disk ({e}) - skipping"
                    )
                    print()

            summaries: list[str | None] = [None] * len(tasks)

            async def _run_summarize(index, content):
                summaries[index] = await summarize(content)

            async with asyncio.TaskGroup() as tg:
                for i, (_, _, content, _, _) in enumerate(tasks):
                    tg.create_task(_run_summarize(i, content))

            for (start, end, content, sys_msgs, path), summary_text in reversed(
                list(zip(tasks, summaries))
            ):
                summary_msg = {
                    "role": "system",
                    "_type": EXCHANGE_SUMMARY,
                    "content": f"[Exchange offloaded to `{path}`]\nSummary: {summary_text}",
                }
                del messages[start:end]
                if count_tokens([summary_msg]) < count_tokens(content):
                    messages.insert(start, summary_msg)
                    for sys_msg in reversed(sys_msgs):
                        messages.insert(start + 1, sys_msg)
                else:
                    for sys_msg in reversed(sys_msgs):
                        messages.insert(start, sys_msg)
                print(f"🗜️  💾 | Offloaded exchange to {os.path.basename(path)}")
                print()

        if count_tokens(messages) <= effective_target:
            return

        # Phase 2: drop tool-call groups in the surviving exchange
        if not (exchanges := get_exchanges()):
            return
        ex_start, ex_end = exchanges[0]
        while count_tokens(messages) > effective_target:
            if not (groups := get_tool_groups(ex_start, ex_end)):
                break
            group_start, group_end = groups[0]
            del messages[group_start:group_end]
            ex_end -= group_end - group_start
        if count_tokens(messages) <= effective_target:
            return

        # Phase 3a: ephemeral system messages (400 error notes, old nudges)
        original_contents = {msg.get("content", "") for msg in SYSTEM_MESSAGES}
        ephemeral = [
            i
            for i, msg in enumerate(messages)
            if msg["role"] == "system"
            and msg.get("content", "") not in original_contents
            and msg.get("_type") not in PROTECTED_TYPES
        ]
        for idx in reversed(ephemeral):
            del messages[idx]
            if count_tokens(messages) <= effective_target:
                return

        # Phase 3b: offload user messages to disk, only if reference is smaller
        offload_index = sum(
            1
            for f in os.listdir(ARTIFACT_DIR)
            if f.startswith("offloaded_user_message_") and f.endswith(".txt")
        )
        for msg in (msg for msg in messages if msg["role"] == "user"):
            raw = msg.get("content", "")
            if not isinstance(raw, str) or not raw:
                continue
            path = f"{ARTIFACT_DIR}/offloaded_user_message_{offload_index}.txt"
            reference = (
                f"[User message offloaded to `{path}` - read it before responding]"
            )
            if count_tokens([{**msg, "content": reference}]) >= count_tokens([msg]):
                continue
            try:
                with open(path, "w") as f:
                    f.write(raw)
            except OSError as e:
                print(f"🗜️  ⚠️  | Could not offload user message ({e}) - skipping")
                print()
                continue
            msg["content"] = reference
            offload_index += 1
            print(f"🗜️  💾 | Offloaded user message ({len(raw)} chars) to disk")
            print()
            if count_tokens(messages) <= effective_target:
                return

        # Phase 3c: plain assistant messages - offload if smaller, then drop
        offload_index = sum(
            1
            for f in os.listdir(ARTIFACT_DIR)
            if f.startswith("offloaded_assistant_message_") and f.endswith(".txt")
        )
        for i, msg in reversed(list(enumerate(messages))):
            if msg["role"] != "assistant" or msg.get("tool_calls"):
                continue
            raw = msg.get("content", "")
            if isinstance(raw, str) and raw:
                path = f"{ARTIFACT_DIR}/offloaded_assistant_message_{offload_index}.txt"
                reference = f"[Assistant response offloaded to `{path}`]"
                if count_tokens([{**msg, "content": reference}]) < count_tokens([msg]):
                    try:
                        with open(path, "w") as f:
                            f.write(raw)
                        msg["content"] = reference
                        offload_index += 1
                        print(
                            f"🗜️  💾 | Offloaded assistant response ({len(raw)} chars) to disk"
                        )
                        print()
                        if count_tokens(messages) <= effective_target:
                            return
                    except OSError as e:
                        print(
                            f"🗜️  🪓 | Could not offload assistant response ({e}) - dropping instead"
                        )
                        print()
            del messages[i]
            print("🗜️  🪓 | Dropped assistant response to free context")
            print()
            if count_tokens(messages) <= effective_target:
                return

        # Phase 3d: drop exchange summaries - files remain on disk
        for idx in reversed(
            [
                i
                for i, msg in enumerate(messages)
                if msg.get("_type") == EXCHANGE_SUMMARY
            ]
        ):
            del messages[idx]
            if count_tokens(messages) <= effective_target:
                return

        # Phase 3e: drop compaction summary and state hint - absolute last resort
        for idx in reversed(
            [
                i
                for i, msg in enumerate(messages)
                if msg.get("_type") in {COMPACTION_SUMMARY, STATE_HINT}
            ]
        ):
            del messages[idx]
            if count_tokens(messages) <= effective_target:
                return

        print(
            f"🗜️  💥 | Unable to reduce context to target ({effective_target}) - "
            f"{count_tokens(messages)}/{MAX_CONTEXT_TOKENS} tokens remain"
        )
        print()

    finally:
        if will_trim_context:
            tokens_after = count_tokens(messages)
            if tokens_after != tokens_before:
                print(
                    f"🗜️  ℹ️  | {tokens_before} -> {tokens_after}/{MAX_CONTEXT_TOKENS} tokens"
                )
                print()


def handle_clear() -> None:
    global messages
    messages = copy.deepcopy(SYSTEM_MESSAGES)
    try:
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
        removed = 0
        for filename in os.listdir(ARTIFACT_DIR):
            if (
                filename.startswith("exchange_")
                and filename.endswith(".json")
                or filename.startswith("offloaded_assistant_message_")
                and filename.endswith(".txt")
                or filename.startswith("offloaded_user_message_")
                and filename.endswith(".txt")
            ):
                os.remove(os.path.join(ARTIFACT_DIR, filename))
                removed += 1
        suffix = (
            f" and removed {removed} offloaded file{'s' if removed != 1 else ''}"
            if removed
            else ""
        )
        print(f"🧹 📜 | Conversation cleared{suffix}")
    except OSError as e:
        print(
            f"🧹 📜 | Conversation cleared (warning: could not remove some files: {e})"
        )
    print()


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
        return stdout.decode(errors="replace").strip()

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
    messages.append({"role": "user", "content": user_message_content})

    # Keep looping as long as the model wants to call tools
    while True:
        await trim_context()

        model = f"{MODEL_ID}:free" if provider == Provider.OPENROUTER else MODEL_ID
        payload = {
            "model": model,
            "messages": [
                {k: v for k, v in msg.items() if not k.startswith("_")}
                for msg in messages
            ],
            "tools": TOOL_SCHEMAS,
            "tool_choice": "auto",
        }

        # print("Sending messages for inference...")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    get_provider_api_config().api_url,
                    headers=headers,
                    json=payload,
                    timeout=INFERENCE_TIMEOUT,
                )
        except httpx.TimeoutException as e:
            print(f"⏱️  🔄 | Inference timeout, retrying due to {type(e).__name__}")
            print()
            await asyncio.sleep(5)
            continue
        except httpx.RequestError as e:
            print(
                f"🌐 🔄 | Inference request failed, retrying due to {type(e).__name__}"
            )
            print()
            await asyncio.sleep(5)
            continue

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
                            "_type": ERROR_INJECTION,
                            "content": (
                                "The server rejected the previous request with a 400 Bad Request error. "
                                "Review the raw server response below to identify what went wrong, "
                                "correct the issue, and try again:\n\n"
                                f"```json\n{response.text}\n```"
                            ),
                        }
                    )
                    continue
                case 429 | 529:
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
                    global _emergency_ceiling
                    # Reduce the assumed ceiling by 25% on each 413,
                    # converging on the provider's real limit over time.
                    system_floor = count_tokens(SYSTEM_MESSAGES)
                    current_ceiling = _emergency_ceiling or MAX_CONTEXT_TOKENS
                    candidate_ceiling = int(0.75 * current_ceiling)
                    effective_ceiling = max(candidate_ceiling, system_floor)
                    # Always leave room for at least one exchange
                    _emergency_ceiling = effective_ceiling
                    print(
                        f"🛑 📦 | Payload too large - reducing budget to "
                        f"{effective_ceiling} tokens and retrying..."
                    )
                    print()
                    await trim_context(force=True)
                    await asyncio.sleep(10)
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
                        f"⚠️  🛑 | Unexpected HTTP error ({status_code}): {response.text}"
                    )
                    print()
                    response.raise_for_status()

        # The request succeeded, so reset the emergency ceiling.
        # A single 413 shouldn't reduce the context budget for the rest
        # of the session.
        _emergency_ceiling = None

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

        # The request succeeded, so drop any ERROR_INJECTION messages.
        messages[:] = [
            msg
            for msg in messages
            if not (msg.get("_type") == ERROR_INJECTION and msg["role"] == "system")
        ]

        # We must append the model's exact response containing the tool request to history
        yield assistant_message
        messages.append(assistant_message)

        finish_reason = response_data["choices"][0].get("finish_reason")
        stop_reason = response_data["choices"][0].get("stop_reason")
        # 200012 == <|call|>
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
                    "_type": ERROR_INJECTION,
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
    session = PromptSession()

    while True:
        user_message_content = await session.prompt_async("👤 💬 | ", multiline=True)
        print()
        if user_message_content.lower() in ["/exit", "/quit"]:
            break
        elif user_message_content.lower() in ["/clear"]:
            handle_clear()
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
                    print(f"❓ 💬 | Received message: {message}")
                    print()
                    raise ValueError(f"Unknown message role: {message['role']}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv))

# docker compose up --build --watch
# docker compose attach agent

# docker compose run -it --rm --service-ports agent
