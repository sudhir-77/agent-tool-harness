"""Synthetic app catalog: 12 apps and 300+ tools with unique action phrases."""

from __future__ import annotations

from harness.registry import ToolRegistry, ToolSpec, schema_for_fields

APPS: list[tuple[str, str]] = [
    ("calendar", "event"),
    ("notes", "note"),
    ("messages", "message"),
    ("files", "file"),
    ("settings", "setting"),
    ("shopping", "item"),
    ("mail", "email"),
    ("contacts", "contact"),
    ("tasks", "task"),
    ("weather", "forecast"),
    ("navigation", "route"),
    ("banking", "payment"),
]

# (action, verb, side_effect). Every generic tool takes title and note.
GENERIC: list[tuple[str, str, bool]] = [
    ("create", "create", False),
    ("get", "get", False),
    ("list", "list", False),
    ("update", "update", False),
    ("delete", "delete", True),
    ("search", "search", False),
    ("archive", "archive", True),
    ("restore", "restore", False),
    ("duplicate", "duplicate", False),
    ("export", "export", False),
    ("share", "share", True),
    ("rename", "rename", False),
    ("count", "count", False),
    ("describe", "describe", False),
    ("tag", "tag", False),
    ("untag", "untag", False),
    ("pin", "pin", False),
    ("unpin", "unpin", False),
    ("star", "star", False),
    ("unstar", "unstar", False),
    ("comment", "comment", False),
    ("assign", "assign", False),
    ("set_status", "set status", False),
    ("attach", "attach", False),
    ("copy_link", "copy link", False),
    ("summarize", "summarize", False),
]

# Explicit phrases keep domain tools distinguishable from the generic set.
DOMAIN: dict[str, list[tuple[str, str, list[str], bool]]] = {
    "calendar": [
        ("schedule_meeting", "schedule the meeting in calendar", ["title", "date", "time", "duration_minutes"], False),
        ("cancel_meeting", "cancel the meeting in calendar", ["title", "date"], True),
    ],
    "notes": [
        ("add_notebook", "add the notebook in notes", ["title", "note"], False),
        ("lock_note", "lock the note in notes", ["title"], True),
    ],
    "messages": [
        ("send_message", "send the message in messages", ["title", "note"], True),
        ("mark_read", "mark read the message in messages", ["title"], False),
    ],
    "files": [
        ("upload_file", "upload the file in files", ["title", "note"], False),
        ("move_file", "move the file in files", ["title", "note"], False),
    ],
    "settings": [
        ("set_preference", "set the preference in settings", ["title", "note"], False),
        ("reset_preference", "reset the preference in settings", ["title"], True),
    ],
    "shopping": [
        ("add_cart_item", "add the cart item in shopping", ["title", "quantity"], False),
        ("checkout", "checkout the cart in shopping", ["title"], True),
    ],
    "mail": [
        ("send_email", "send the email in mail", ["title", "note"], True),
        ("draft_email", "draft the email in mail", ["title", "note"], False),
    ],
    "contacts": [
        ("add_contact", "add the contact in contacts", ["title", "note"], False),
        ("merge_contact", "merge the contact in contacts", ["title", "note"], False),
    ],
    "tasks": [
        ("add_task", "add the task in tasks", ["title", "note", "status"], False),
        ("complete_task", "complete the task in tasks", ["title"], False),
    ],
    "weather": [
        ("get_forecast", "fetch the forecast in weather", ["title", "date"], False),
        ("set_alert", "set the alert in weather", ["title", "date"], True),
    ],
    "navigation": [
        ("plan_route", "plan the route in navigation", ["title", "note"], False),
        ("save_place", "save the place in navigation", ["title", "note"], False),
    ],
    "banking": [
        ("transfer_funds", "transfer the funds in banking", ["title", "amount"], True),
        ("check_balance", "check the balance in banking", ["title"], False),
    ],
}


def _make_tool(app: str, action: str, phrase: str, fields: list[str], side_effect: bool) -> ToolSpec:
    name = f"{app}.{action}"
    description = f"{phrase}. Use {name} when the user asks to {phrase}."
    return ToolSpec(
        name=name,
        app=app,
        action=action,
        phrase=phrase,
        description=description,
        side_effect=side_effect,
        parameters=schema_for_fields(fields),
        retrieval_text=f"{name.replace('.', ' ')} {phrase}",
    )


def build_tools() -> list[ToolSpec]:
    tools: list[ToolSpec] = []
    for app, noun in APPS:
        for action, verb, side_effect in GENERIC:
            phrase = f"{verb} the {noun} in {app}"
            tools.append(_make_tool(app, action, phrase, ["title", "note"], side_effect))
        for action, phrase, fields, side_effect in DOMAIN[app]:
            tools.append(_make_tool(app, action, phrase, fields, side_effect))
    phrases = [tool.phrase for tool in tools]
    if len(phrases) != len(set(phrases)):
        counts: dict[str, int] = {}
        for phrase in phrases:
            counts[phrase] = counts.get(phrase, 0) + 1
        dupes = [phrase for phrase, count in counts.items() if count > 1]
        raise RuntimeError(f"tool phrases must be unique: {dupes}")
    if len(tools) < 300:
        raise RuntimeError("catalog must contain at least 300 tools")
    if len({tool.app for tool in tools}) != 12:
        raise RuntimeError("catalog must contain 12 apps")
    return tools


def build_registry() -> ToolRegistry:
    return ToolRegistry(build_tools())
