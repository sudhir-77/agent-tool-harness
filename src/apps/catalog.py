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

# Natural-language caller wording. Test instructions use these sentences and do not
# contain the canonical action phrase or the tool id. The same sentence is part of
# the tool card so retrieval and the model can match documented caller language.
CALLER_TEMPLATES: dict[str, str] = {
    "create": "Start a fresh {noun} inside the {app} workspace",
    "get": "Pull up one saved {noun} from the {app} workspace",
    "list": "Show the full inventory of {noun} records from {app}",
    "update": "Revise fields on an existing {noun} in the {app} workspace",
    "delete": "Erase a stored {noun} from the {app} workspace",
    "search": "Run a lookup for a {noun} in the {app} workspace",
    "archive": "Shelve an old {noun} from the {app} workspace",
    "restore": "Reinstate a shelved {noun} in the {app} workspace",
    "duplicate": "Make a clone of this {noun} in the {app} workspace",
    "export": "Download a copy of this {noun} from {app}",
    "share": "Circulate a copy of this {noun} from {app}",
    "rename": "Retitle this {noun} in the {app} workspace",
    "count": "Tally the {noun} records stored in {app}",
    "describe": "Narrate the details of this {noun} in {app}",
    "tag": "Apply a label to this {noun} in the {app} workspace",
    "untag": "Clear an unlabel from this {noun} in {app}",
    "pin": "Keep a sticky {noun} at the top of {app}",
    "unpin": "Release an unsticky {noun} in {app}",
    "star": "Mark a favorite {noun} in the {app} workspace",
    "unstar": "Clear an unfavorite {noun} in {app}",
    "comment": "Leave a remark on this {noun} in {app}",
    "assign": "Delegate ownership of this {noun} in {app}",
    "set_status": "Change the progress state of this {noun} in {app}",
    "attach": "Enclose an extra file on this {noun} in {app}",
    "copy_link": "Produce a permalink for this {noun} in {app}",
    "summarize": "Write a short recap of this {noun} in {app}",
    "schedule_meeting": "Book a sitdown on the {app} board",
    "cancel_meeting": "Call off that sitdown on the {app} board",
    "add_notebook": "Begin a jotter inside {app}",
    "lock_note": "Apply a padlock to this {noun} in {app}",
    "send_message": "Dispatch an outbound {noun} through {app}",
    "mark_read": "Acknowledge this {noun} in {app} as seen",
    "upload_file": "Ingest a {noun} into {app}",
    "move_file": "Relocate this {noun} within {app}",
    "set_preference": "Configure one {noun} value in {app}",
    "reset_preference": "Apply factory defaults for a {noun} in {app}",
    "add_cart_item": "Place one {noun} into the {app} basket",
    "checkout": "Purchase the current basket in {app}",
    "send_email": "Fire an outbound {noun} from {app}",
    "draft_email": "Compose an unsent {noun} in {app}",
    "add_contact": "Record a new acquaintance in {app}",
    "merge_contact": "Dedupe two people inside {app}",
    "add_task": "Open a todo inside {app}",
    "complete_task": "Finish the todo inside {app}",
    "get_forecast": "Read the outlook for a day in {app}",
    "set_alert": "Create a warning entry inside {app}",
    "plan_route": "Build an itinerary inside {app}",
    "save_place": "Remember a landmark inside {app}",
    "transfer_funds": "Send a remittance through {app}",
    "check_balance": "Read the holdings of an account in {app}",
}

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


def caller_text(app: str, noun: str, action: str) -> str:
    """Documented way a person asks for this action, without the tool id."""
    try:
        template = CALLER_TEMPLATES[action]
    except KeyError as exc:
        raise KeyError(f"no caller wording for {action}") from exc
    return template.format(app=app, noun=noun)


def _make_tool(app: str, noun: str, action: str, phrase: str, fields: list[str], side_effect: bool) -> ToolSpec:
    name = f"{app}.{action}"
    wording = caller_text(app, noun, action)
    description = f"{phrase}. Callers may say: {wording}."
    return ToolSpec(
        name=name,
        app=app,
        action=action,
        phrase=phrase,
        description=description,
        side_effect=side_effect,
        parameters=schema_for_fields(fields),
        retrieval_text=f"{name.replace('.', ' ')} {phrase} {wording}",
    )


def build_tools() -> list[ToolSpec]:
    nouns = dict(APPS)
    tools: list[ToolSpec] = []
    for app, noun in APPS:
        for action, verb, side_effect in GENERIC:
            phrase = f"{verb} the {noun} in {app}"
            tools.append(_make_tool(app, noun, action, phrase, ["title", "note"], side_effect))
        for action, phrase, fields, side_effect in DOMAIN[app]:
            tools.append(_make_tool(app, noun, action, phrase, fields, side_effect))
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
    for tool in tools:
        wording = caller_text(tool.app, nouns[tool.app], tool.action)
        if tool.phrase in wording or tool.name in wording:
            raise RuntimeError(f"caller wording mentions {tool.name}")
        for other in tools:
            if other.phrase in wording or other.name in wording:
                raise RuntimeError(f"caller wording for {tool.name} contains {other.name}")
    return tools


def build_registry() -> ToolRegistry:
    return ToolRegistry(build_tools())
