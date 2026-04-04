import os
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def _mib(num_bytes: int) -> float:
    return num_bytes / 1024 / 1024


def main() -> None:
    model_id = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    prompt = os.environ.get("QWEN_VL_PROMPT", "用一句话介绍你自己，并说你能做什么。")

    Guga_dir = Path(__file__).resolve().parents[1]
    default_cache_dir = str(Guga_dir / "models_cache_vl_3b")
    cache_dir = os.environ.get("QWEN_VL_CACHE_DIR", default_cache_dir)

    max_new_tokens = int(os.environ.get("QWEN_VL_MAX_NEW_TOKENS", "128"))
    temperature = float(os.environ.get("QWEN_VL_TEMPERATURE", "0.7"))
    top_p = float(os.environ.get("QWEN_VL_TOP_P", "0.9"))

    print(f"model: {model_id}")
    print(f"cache_dir: {cache_dir}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device: {torch.cuda.get_device_name(0)}")

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=False,
        cache_dir=cache_dir,
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]

    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"System: You are a helpful assistant.\nUser: {prompt}\nAssistant:"

    try:
        inputs = processor(text=[text], images=None, padding=True, return_tensors="pt")
    except TypeError:
        inputs = processor(text=[text], padding=True, return_tensors="pt")

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            cache_dir=cache_dir,
            trust_remote_code=False,
        )
    except ValueError as e:
        # 未安装 accelerate 时，device_map 会直接报错；这里做一个最小降级。
        msg = str(e)
        if torch.cuda.is_available() and "requires `accelerate`" in msg:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=None,
                cache_dir=cache_dir,
                trust_remote_code=False,
            )
            model.to("cuda")
        else:
            raise

    model.eval()

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        print(f"vram_allocated_after_load_mib: {_mib(torch.cuda.memory_allocated()):.1f}")
        print(f"vram_reserved_after_load_mib:  {_mib(torch.cuda.memory_reserved()):.1f}")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    in_len = int(inputs["input_ids"].shape[-1])

    t0 = time.time()
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    gen_len = int(out_ids.shape[-1]) - in_len
    gen_len = max(gen_len, 0)

    gen_ids = out_ids[:, in_len:]
    text_out = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    print("\n--- output ---")
    print(text_out)
    print("---")
    print(f"elapsed_sec: {t1 - t0:.2f}")
    print(f"generated_tokens: {gen_len}")
    if t1 > t0:
        print(f"tokens_per_sec: {gen_len / (t1 - t0):.2f}")

    if torch.cuda.is_available():
        print(f"vram_peak_allocated_mib: {_mib(torch.cuda.max_memory_allocated()):.1f}")
        print(f"vram_peak_reserved_mib:  {_mib(torch.cuda.max_memory_reserved()):.1f}")
        print(f"vram_allocated_end_mib:  {_mib(torch.cuda.memory_allocated()):.1f}")
        print(f"vram_reserved_end_mib:   {_mib(torch.cuda.memory_reserved()):.1f}")


if __name__ == "__main__":
    main()
