from __future__ import annotations

from collections.abc import Iterator
from threading import Event

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import TextIteratorStreamer

from guga.types import GenerationConfig


class _CancelStoppingCriteria(StoppingCriteria):
    def __init__(self, cancel_event: Event) -> None:
        super().__init__()
        self._cancel_event = cancel_event

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:  # type: ignore[override]
        return self._cancel_event.is_set()


class QwenVLChatModel:
    def __init__(self, model_id: str, cache_dir: str | None = None) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=False,
            cache_dir=cache_dir,
        )

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                cache_dir=cache_dir,
                trust_remote_code=False,
            )
        except TypeError:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                cache_dir=cache_dir,
                trust_remote_code=False,
            )
        except ValueError as e:
            msg = str(e)
            if torch.cuda.is_available() and "requires `accelerate`" in msg:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id,
                    dtype=dtype,
                    device_map=None,
                    cache_dir=cache_dir,
                    trust_remote_code=False,
                )
                self.model.to("cuda")
            else:
                raise

        self.model.eval()

    def generate_reply(self, messages: list[dict[str, str]], gen: GenerationConfig) -> str:
        vl_messages = []
        for m in messages:
            vl_messages.append(
                {
                    "role": m["role"],
                    "content": [{"type": "text", "text": m["content"]}],
                }
            )

        if hasattr(self.processor, "apply_chat_template"):
            text = self.processor.apply_chat_template(
                vl_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"

        try:
            inputs = self.processor(text=[text], images=None, padding=True, return_tensors="pt")
        except TypeError:
            inputs = self.processor(text=[text], padding=True, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        in_len = int(inputs["input_ids"].shape[-1])

        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=gen.max_new_tokens,
                do_sample=True,
                temperature=gen.temperature,
                top_p=gen.top_p,
            )

        gen_ids = out_ids[:, in_len:]
        return self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    def generate_reply_stream(
        self,
        messages: list[dict[str, str]],
        gen: GenerationConfig,
        cancel_event: Event | None = None,
    ) -> Iterator[str]:
        import threading

        vl_messages = []
        for m in messages:
            vl_messages.append(
                {
                    "role": m["role"],
                    "content": [{"type": "text", "text": m["content"]}],
                }
            )

        if hasattr(self.processor, "apply_chat_template"):
            text = self.processor.apply_chat_template(
                vl_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"

        try:
            inputs = self.processor(text=[text], images=None, padding=True, return_tensors="pt")
        except TypeError:
            inputs = self.processor(text=[text], padding=True, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("processor.tokenizer is required for streaming generation")

        if cancel_event is None:
            cancel_event = Event()

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        stopping = StoppingCriteriaList([_CancelStoppingCriteria(cancel_event)])

        def _run_generate() -> None:
            with torch.inference_mode():
                self.model.generate(
                    **inputs,
                    max_new_tokens=gen.max_new_tokens,
                    do_sample=True,
                    temperature=gen.temperature,
                    top_p=gen.top_p,
                    streamer=streamer,
                    stopping_criteria=stopping,
                )

        t = threading.Thread(target=_run_generate, daemon=True)
        t.start()

        for piece in streamer:
            if piece:
                yield piece

        t.join(timeout=5)
