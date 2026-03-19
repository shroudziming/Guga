import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    model_id = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    prompt = os.environ.get("QWEN_PROMPT", "用一句话介绍你自己，并说你能做什么。")

    print(f"model: {model_id}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)

    # 优先使用 chat template（Qwen Instruct 通常支持）
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # 兜底：不依赖模板，保证本课可跑通
        text = f"System: You are a helpful assistant.\nUser: {prompt}\nAssistant:"  # noqa: E501

    encoded = tokenizer(text, return_tensors="pt")

    if torch.cuda.is_available():
        dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 让输入跟随模型所在设备（device_map=auto 时可用 model.device for 单卡）
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    gen_kwargs = dict(
        max_new_tokens=int(os.environ.get("QWEN_MAX_NEW_TOKENS", "128")),
        do_sample=True,
        temperature=float(os.environ.get("QWEN_TEMPERATURE", "0.7")),
        top_p=float(os.environ.get("QWEN_TOP_P", "0.9")),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    start = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
    end = time.time()

    generated = output_ids[0][input_ids.shape[-1] :]
    text_out = tokenizer.decode(generated, skip_special_tokens=True)

    print("\n--- output ---")
    print(text_out.strip())
    print("---")
    print(f"elapsed_sec: {end - start:.2f}")


if __name__ == "__main__":
    main()
