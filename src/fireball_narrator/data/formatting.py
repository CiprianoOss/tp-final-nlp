from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = (
    "You are a narrative role-playing agent in a medieval fantasy world. "
    "Continue the scene using only the supplied dialogue, game state, and "
    "resolved action. Preserve continuity and character facts. Do not add "
    "modern technology or science-fiction elements. Return only the next "
    "in-character narrative turn."
)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item)
    return str(value).strip()


def format_actor_short(actor: dict[str, Any]) -> str:
    name = _clean(actor.get("name")) or "Unknown"
    identity = "; ".join(
        value
        for value in (_clean(actor.get("race")), _clean(actor.get("class")))
        if value
    )
    parts = [name]
    if identity:
        parts.append(f"({identity})")
    hp = _clean(actor.get("hp"))
    if hp:
        parts.append(hp)
    effects = _clean(actor.get("effects"))
    if effects:
        parts.append(f"[Effects: {effects}]")
    return " ".join(parts)


def format_actor_long(actor: dict[str, Any] | None) -> str:
    if not actor:
        return "None"
    fields = (
        ("Name", "name"),
        ("Class", "class"),
        ("Race", "race"),
        ("HP", "hp"),
        ("Attacks", "attacks"),
        ("Spells", "spells"),
        ("Actions", "actions"),
        ("Effects", "effects"),
    )
    lines = []
    for label, key in fields:
        value = _clean(actor.get(key))
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines) or "None"


def build_state_to_narration_prompt(record: dict[str, Any]) -> str:
    sections: list[str] = []

    history = record.get("utterance_history") or []
    if history:
        sections.append("Recent dialogue:\n" + "\n".join(map(str, history)))

    actors = record.get("combat_state_after") or []
    if actors:
        actor_lines = "\n".join(f"- {format_actor_short(actor)}" for actor in actors)
        sections.append("Actors after the action:\n" + actor_lines)

    caster = record.get("caster_after")
    if caster:
        sections.append("Acting character:\n" + format_actor_long(caster))

    targets = record.get("targets_after") or []
    if targets:
        target_lines = "\n".join(f"- {format_actor_short(actor)}" for actor in targets)
        sections.append("Targets:\n" + target_lines)

    commands = record.get("commands_norm") or []
    if commands:
        sections.append("Game command:\n" + "\n".join(map(str, commands)))

    results = record.get("automation_results") or []
    if results:
        sections.append("Resolved action:\n" + "\n".join(map(str, results)))

    sections.append("Write the next narrative turn.")
    return "\n\n".join(sections)


def record_to_chat(record: dict[str, Any]) -> dict[str, Any] | None:
    target_parts = record.get("after_utterances") or []
    results = record.get("automation_results") or []
    if not target_parts or not results:
        return None

    target = "\n".join(str(part).strip() for part in target_parts if str(part).strip())
    if not target:
        return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_state_to_narration_prompt(record)},
            {"role": "assistant", "content": target},
        ],
        "metadata": {
            "speaker_id": str(record.get("speaker_id", "")),
            "task": "state_to_narration",
        },
    }

