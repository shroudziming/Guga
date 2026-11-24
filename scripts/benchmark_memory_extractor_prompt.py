from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from guga.memory.summarizer import MemoryBankSummarizer, SummaryGenerationError


DEFAULT_PACKET = {
    "new_turns": [
        {
            "user_message_id": "msg_user_benchmark",
            "assistant_message_id": "msg_assistant_benchmark",
            "created_at": "2026-11-24T10:00:00+08:00",
            "user_text": "我明天下午三点要参加一家公司的产品经理面试。",
            "assistant_text": "好，我记住了。明天下午三点是这次产品经理面试，祝你准备顺利。",
        }
    ],
    "recent_active_events": [],
    "relevant_active_events": [],
    "retrieved_context": [],
}


class _PromptCaptureModel:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def generate_reply(self, messages, gen) -> str:
        _ = gen
        self.messages = messages
        return json.dumps({"semantic_event_operations": [], "event_summaries": []})


class _TokenTimer(LogitsProcessor):
    def __init__(self) -> None:
        self.timestamps: list[float] = []

    def __call__(self, input_ids, scores):
        _ = input_ids
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timestamps.append(time.perf_counter())
        return scores


def _build_messages(persona_path: Path) -> list[dict[str, str]]:
    capture = _PromptCaptureModel()
    summarizer = MemoryBankSummarizer(model=capture, use_llm=True, retry_delays=())
    summarizer.consolidate_low_level_memory(
        DEFAULT_PACKET,
        include_guga_reflection=True,
        reflection_context=persona_path.read_text(encoding="utf-8"),
    )
    return capture.messages


def _run_generation(model, tokenizer, messages, max_new_tokens: int) -> dict[str, object]:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {name: value.to(model.device) for name, value in inputs.items()}
    input_tokens = int(inputs["input_ids"].shape[-1])
    timer = _TokenTimer()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=[timer],
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ended = time.perf_counter()

    generated = output_ids[:, input_tokens:]
    output_tokens = int(generated.shape[-1])
    raw_output = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
    ttft_ms = (timer.timestamps[0] - started) * 1000 if timer.timestamps else None
    decode_seconds = timer.timestamps[-1] - timer.timestamps[0] if len(timer.timestamps) > 1 else 0.0
    decode_tokens_per_second = (output_tokens - 1) / decode_seconds if decode_seconds > 0 else None
    elapsed_seconds = ended - started

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "hit_eos": output_tokens < max_new_tokens,
        "ttft_ms": ttft_ms,
        "decode_tokens_per_second": decode_tokens_per_second,
        "end_to_end_tokens_per_second": output_tokens / elapsed_seconds,
        "elapsed_ms": elapsed_seconds * 1000,
        "raw_output": raw_output,
    }


def _validation_result(raw_output: str) -> dict[str, object]:
    summarizer = MemoryBankSummarizer(model=_PromptCaptureModel(), use_llm=True, retry_delays=())
    parsed = summarizer._parse_json_object(raw_output)
    if not parsed:
        return {"json_parseable": False, "validator_passed": False, "error": "invalid JSON object"}
    try:
        validated = summarizer._validate_low_level_result(parsed, DEFAULT_PACKET, include_guga_reflection=True)
    except SummaryGenerationError as exc:
        return {"json_parseable": True, "validator_passed": False, "error": str(exc), "parsed": parsed}
    return {"json_parseable": True, "validator_passed": True, "validated": validated}


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the local model against Guga's current Stage 1 prompt.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--persona", type=Path, default=Path("config/personas/default/SKILL.md"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--speed-tokens", type=int, default=256)
    parser.add_argument("--full-tokens", type=int, default=768)
    args = parser.parse_args()

    messages = _build_messages(args.persona)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    load_seconds = time.perf_counter() - started
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    speed_runs = [_run_generation(model, tokenizer, messages, args.speed_tokens) for _ in range(args.repeats)]
    full_run = _run_generation(model, tokenizer, messages, args.full_tokens)
    validation = _validation_result(str(full_run["raw_output"]))

    metrics = {
        "model_path": str(args.model.resolve()),
        "local_files_only": True,
        "decode_mode": "greedy",
        "packet": DEFAULT_PACKET,
        "load_seconds": load_seconds,
        "speed_runs": speed_runs,
        "speed_medians": {
            "ttft_ms": statistics.median(float(run["ttft_ms"]) for run in speed_runs),
            "decode_tokens_per_second": statistics.median(
                float(run["decode_tokens_per_second"]) for run in speed_runs
            ),
            "end_to_end_tokens_per_second": statistics.median(
                float(run["end_to_end_tokens_per_second"]) for run in speed_runs
            ),
        },
        "full_run": full_run,
        "validation": validation,
        "peak_cuda_memory_mib": (
            torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in metrics.items() if key not in {"speed_runs", "full_run"}}, ensure_ascii=False, indent=2))
    print(f"full_run={json.dumps(full_run, ensure_ascii=False, indent=2)}")
    print(f"saved={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
