"""Prompt construction, context window, and history compaction."""

from __future__ import annotations

from dataclasses import dataclass

from harness.registry import ToolSpec, render_tool_block
from harness.tokens import count_tokens

SYSTEM_PROMPT = (
    "You are an assistant that calls one tool at a time. "
    "Choose a tool whose phrase appears in the current clause. "
    "Copy arguments from key=value pairs in that clause. "
    "If a tool is side-effecting, call it once to request confirmation, "
    "then call it again with confirmation after the user agrees."
)

# Fixed before any test-set evaluation. Sized like a small local context window.
CONTEXT_LIMIT = 2048
RETRIEVAL_K = 5


@dataclass
class RenderedTool:
    tool: ToolSpec
    block: str
    tokens: int


def render_tools(tools: list[ToolSpec]) -> list[RenderedTool]:
    rendered: list[RenderedTool] = []
    for tool in tools:
        block = render_tool_block(tool)
        rendered.append(RenderedTool(tool=tool, block=block, tokens=count_tokens(block)))
    return rendered


def compact_history(history: list[str]) -> list[str]:
    """Keep the latest observation verbatim and drop argument payloads from older ones."""
    if len(history) <= 1:
        return list(history)
    older = []
    for line in history[:-1]:
        head = line.split(" args=", 1)[0]
        older.append(f"memory: {head[:80]}")
    return older + [history[-1]]


def build_header(task_text: str) -> str:
    return f"{SYSTEM_PROMPT}\nUSER TASK:\n{task_text}\nTOOLS:\n"


@dataclass
class PromptPacket:
    full_text: str
    full_tokens: int
    visible: list[ToolSpec]
    window_tokens: int


def assemble_prompt(
    task_text: str,
    rendered: list[RenderedTool],
    history: list[str],
    *,
    compact: bool,
    context_limit: int = CONTEXT_LIMIT,
) -> PromptPacket:
    """Build the strategy prompt and the prefix that fits in the context window.

    Tool blocks are listed before history, so a tail truncation drops history
    first and then the lowest-ranked tools. A tool counts as visible only when
    its entire block fits.
    """
    used_history = compact_history(history) if compact else list(history)
    header = build_header(task_text)
    header_tokens = count_tokens(header)
    history_text = "\n".join(used_history)
    history_block = f"HISTORY:\n{history_text}\n" if used_history else "HISTORY:\n"
    history_tokens = count_tokens(history_block)
    tool_tokens = sum(item.tokens for item in rendered)
    full_tokens = header_tokens + tool_tokens + history_tokens

    visible: list[ToolSpec] = []
    used = header_tokens
    visible_tool_tokens = 0
    for item in rendered:
        if used + item.tokens <= context_limit:
            used += item.tokens
            visible_tool_tokens += item.tokens
            visible.append(item.tool)
        else:
            break
    window_tokens = min(context_limit, header_tokens + visible_tool_tokens + history_tokens)
    # The full text is materialized only when a caller needs it. Token counts
    # are exact because the tokenizer ignores whitespace, so pieces add.
    full_text = header + "".join(item.block for item in rendered) + history_block
    return PromptPacket(
        full_text=full_text,
        full_tokens=full_tokens,
        visible=visible,
        window_tokens=window_tokens,
    )
