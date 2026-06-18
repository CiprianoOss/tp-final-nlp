from __future__ import annotations

import argparse
import json
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from fireball_narrator.compression.caveman import compress_text
from fireball_narrator.evaluation.metrics import (
    information_retention_accuracy,
    semantic_preservation_score,
    token_compression_ratio,
)

QA_SYSTEM_PROMPT = (
    "Answer the closed question using only the supplied campaign memory. "
    "Return a short factual answer. If the memory does not contain the answer, "
    "reply UNKNOWN."
)


def normalize_answer(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def answer_is_correct(response: str, expected: str | list[str]) -> bool:
    accepted = [expected] if isinstance(expected, str) else expected
    normalized_response = f" {normalize_answer(response)} "
    return any(
        f" {normalize_answer(answer)} " in normalized_response
        for answer in accepted
        if normalize_answer(answer)
    )


def fact_is_preserved(fact: str | dict[str, Any], context: str) -> bool:
    if isinstance(fact, dict):
        keywords = [str(value) for value in fact.get("keywords", [])]
        if not keywords and fact.get("text"):
            keywords = compress_text(str(fact["text"])).split()
    else:
        keywords = compress_text(fact).split()

    normalized_context = f" {normalize_answer(context)} "
    return bool(keywords) and all(
        f" {normalize_answer(keyword)} " in normalized_context
        for keyword in keywords
        if normalize_answer(keyword)
    )


def read_campaigns(path: Path) -> list[dict[str, Any]]:
    campaigns = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("history"), list):
                raise ValueError(f"{path}:{line_number} requires a history list")
            if not isinstance(row.get("facts"), list):
                raise ValueError(f"{path}:{line_number} requires a facts list")
            if not isinstance(row.get("qa"), list):
                raise ValueError(f"{path}:{line_number} requires a qa list")
            campaigns.append(row)
    if not campaigns:
        raise ValueError(f"No campaigns found in {path}")
    return campaigns


def render_history(turns: list[str]) -> str:
    return "\n".join(
        f"[Turn {index}] {turn}" for index, turn in enumerate(turns, start=1)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TCR, SPS and QA retention accuracy on campaign logs."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--campaigns", type=Path, required=True)
    parser.add_argument(
        "--history-mode",
        choices=("original", "caveman"),
        required=True,
    )
    parser.add_argument("--steering-vector", type=Path)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--max-context-tokens", type=int, default=30000)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    import torch

    from fireball_narrator.modeling import (
        load_model_and_tokenizer,
        model_input_device,
    )
    from fireball_narrator.steering.runtime import (
        apply_steering,
        load_steering_vectors,
    )

    args = parse_args()
    campaigns = read_campaigns(args.campaigns)
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        adapter_path=args.adapter,
        load_in_4bit=args.load_in_4bit,
    )
    vectors = (
        load_steering_vectors(args.steering_vector)
        if args.steering_vector
        else None
    )

    total_original_tokens = 0
    total_compressed_tokens = 0
    all_fact_results: list[bool] = []
    all_qa_results: list[bool] = []
    campaign_reports = []

    for campaign_index, campaign in enumerate(campaigns, start=1):
        original_turns = [str(turn) for turn in campaign["history"]]
        # Preserve turn indices so questions about an early numbered turn remain valid.
        compressed_turns = [compress_text(turn) for turn in original_turns]
        original_text = render_history(original_turns)
        compressed_text = render_history(compressed_turns)
        selected_text = (
            original_text if args.history_mode == "original" else compressed_text
        )

        original_tokens = len(tokenizer.encode(original_text, add_special_tokens=False))
        compressed_tokens = len(
            tokenizer.encode(compressed_text, add_special_tokens=False)
        )
        total_original_tokens += original_tokens
        total_compressed_tokens += compressed_tokens

        fact_results = [
            fact_is_preserved(fact, selected_text) for fact in campaign["facts"]
        ]
        all_fact_results.extend(fact_results)
        qa_reports = []

        for qa in campaign["qa"]:
            question = str(qa["question"])
            expected = qa.get("answers", qa.get("answer", ""))
            user_prompt = f"Campaign memory:\n{selected_text}\n\nQuestion: {question}"
            inputs = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": QA_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )

            max_input_tokens = args.max_context_tokens - args.max_new_tokens
            truncated = inputs["input_ids"].shape[1] > max_input_tokens
            if truncated:
                inputs = {
                    key: value[:, -max_input_tokens:]
                    for key, value in inputs.items()
                }
            inputs = {
                key: value.to(model_input_device(model))
                for key, value in inputs.items()
            }
            steering_context = (
                apply_steering(model, vectors, args.alpha)
                if vectors
                else nullcontext()
            )
            with torch.inference_mode(), steering_context:
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(
                generated[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ).strip()
            correct = answer_is_correct(response, expected)
            all_qa_results.append(correct)
            qa_reports.append(
                {
                    "question": question,
                    "expected": expected,
                    "response": response,
                    "correct": correct,
                    "context_truncated": truncated,
                }
            )

        campaign_reports.append(
            {
                "id": campaign.get("id", f"campaign-{campaign_index:03d}"),
                "turns": len(original_turns),
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "TCR": token_compression_ratio(original_tokens, compressed_tokens),
                "SPS": semantic_preservation_score(
                    [str(index) for index in range(len(fact_results))],
                    [
                        str(index)
                        for index, preserved in enumerate(fact_results)
                        if preserved
                    ],
                ),
                "IRA": information_retention_accuracy(
                    report["correct"] for report in qa_reports
                ),
                "qa": qa_reports,
            }
        )

    caveman_tcr = token_compression_ratio(
        total_original_tokens,
        total_compressed_tokens,
    )
    report = {
        "model": args.model,
        "adapter": str(args.adapter),
        "history_mode": args.history_mode,
        "steering_vector": (
            str(args.steering_vector) if args.steering_vector else ""
        ),
        "alpha": args.alpha if args.steering_vector else 0.0,
        "campaign_count": len(campaigns),
        "question_count": len(all_qa_results),
        "fact_count": len(all_fact_results),
        "TCR": caveman_tcr if args.history_mode == "caveman" else 0.0,
        "caveman_TCR": caveman_tcr,
        "SPS": semantic_preservation_score(
            [str(index) for index in range(len(all_fact_results))],
            [
                str(index)
                for index, preserved in enumerate(all_fact_results)
                if preserved
            ],
        ),
        "IRA": information_retention_accuracy(all_qa_results),
        "campaigns": campaign_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {key: value for key, value in report.items() if key != "campaigns"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
