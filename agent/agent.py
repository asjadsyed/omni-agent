#!/usr/bin/env python3
import asyncio
import json
import re
import sys

import dotenv
import httpx

GROQ_API_KEY = dotenv.get_key(dotenv.find_dotenv(), "GROQ_API_KEY")

# Configuration
API_URL = "http://llama-cpp:8080/v1/chat/completions"
# API_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    # "Authorization": f"Bearer {GROQ_API_KEY}",
}

RATE_LIMIT_RETRY_RE = re.compile(
    r"Please try again in\s+"
    r"(?:(?P<hours>\d+(?:\.\d+)?)h)?"
    r"(?:(?P<minutes>\d+(?:\.\d+)?)m)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)s)?"
    r"(?:(?P<milliseconds>\d+(?:\.\d+)?)ms)?"
)


# Local Python Functions (The real tools on your system)
async def bash_tool(command: str):
    result = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await result.communicate()
    return {
        "success": result.returncode == 0,
        "stdout": stdout.decode().strip(),
        "stderr": stderr.decode().strip(),
        "returncode": result.returncode,
    }


# Define the Tool Schemas (What the AI reads)
tools = [
    {
        "type": "function",
        "function": {
            "name": "bash_tool",
            "description": "Execute a shell command on the host system to run code, access system utilities, or manage environment resources. Call this tool when fulfilling the request requires external execution or live system interaction.",
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


# Send the initial user message + the tools list to the model
messages = [
    {
        "role": "system",
        "content": """\
You are an autonomous agent.

Your task is complete only when the user's request has been successfully accomplished.

Tool results are observations about the world.

Do not assume what can be observed.

When uncertain, gather information rather than speculate.

If later actions depend on earlier actions, observe the outcome before proceeding.

Errors are observations, not conclusions.

Continue working toward the goal until:
- the task is complete,
- you determine it cannot be completed, or
- you require information that only the user can provide.

Explain failures only after exhausting reasonable attempts to recover.

Prefer actions over descriptions when actions are possible.
""",
    }
]


async def chat_with_agent(user_message_content):
    global messages
    messages.append({"role": "user", "content": user_message_content})

    # Keep looping as long as the model wants to call tools
    while True:
        payload = {
            "model": "openai/gpt-oss-20b",
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }

        # print("Sending messages for inference...")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                API_URL, headers=HEADERS, json=payload, timeout=None
            )
        if not response.is_success:
            print(f"{response.status_code=}")
            print(f"{response.text=}")
            print()

            match response.status_code:
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

                    data = response.json()
                    error_message = data["error"]["message"]
                    match = RATE_LIMIT_RETRY_RE.search(error_message)

                    if match:
                        hours = float(match.group("hours") or 0)
                        minutes = float(match.group("minutes") or 0)
                        seconds = float(match.group("seconds") or 0)
                        milliseconds = float(match.group("milliseconds") or 0)

                        rate_limit_duration = (
                            hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
                        )

                        print(
                            f"Extracted rate limit duration: {rate_limit_duration} seconds"
                        )
                    else:
                        rate_limit_duration = 10
                        print(
                            "Could not parse rate limit duration; defaulting to 10 seconds"
                        )

                    print(
                        f"⏳ | Sleeping for {rate_limit_duration} seconds before retrying..."
                    )
                    print()
                    await asyncio.sleep(rate_limit_duration)
                    continue
                case 413:
                    print(
                        "Request entity too large, compacting message history and retrying..."
                    )
                    # TODO
                    continue

            response.raise_for_status()
        response_data = response.json()

        # Extract the assistant message dictionary
        assistant_message = response_data["choices"][0]["message"]
        # print(f"{assistant_message=}")
        tool_calls = assistant_message.get("tool_calls")
        # print(f"{tool_calls=}")

        # We must append the model's exact response containing the tool request to history
        yield assistant_message
        messages.append(assistant_message)

        # Break condition: If there are no tool calls, this is your final response to the user
        if tool_calls is None:
            return
        else:
            # Handle the tool calls if the model requested them
            # print("The assistant decided that tool calls were necessary.")

            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])

                print(
                    f"🕵️  🛠️  | Processing tool call: {function_name} with arguments {function_args}"
                )
                print()

                match function_name:
                    case "bash_tool":
                        tool_result = await bash_tool(
                            command=function_args.get("command")
                        )

                        # Append the execution result to history using the tool role and call ID
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": function_name,
                            "content": json.dumps(tool_result),
                        }
                        yield tool_message
                        messages.append(tool_message)


async def main(argv):
    from prompt_toolkit import PromptSession

    session = PromptSession()

    while True:
        user_message_content = await session.prompt_async("👤 💬 | ", multiline=True)
        print()
        if user_message_content.lower() in ["exit", "quit"]:
            break

        async for message in chat_with_agent(user_message_content):
            match message["role"]:
                case "assistant":
                    reasoning_content = message.get("reasoning_content")
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
                    print(f"Received message: {message}")
                    print()
                    raise ValueError(f"Unknown message role: {message['role']}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv))

# docker compose up --build --watch
# docker compose attach agent

# docker compose run -it --rm agent
