"""One local Qwen model for actor and monitor inference, without model APIs.

The GPU path preserves the user's working Qwen3-4B loading settings, with a
fresh dynamic KV cache for every role call. Actor thinking is enabled and
monitor thinking is disabled. This model substitution is not the Claude
configuration in arXiv:2603.16928v1. Importing this module neither imports
torch nor downloads model weights.

``max_time`` is a soft generation deadline: Transformers checks it after a
generation step, so a long prefill cannot be interrupted immediately. Run
the entry point in a bounded subprocess for an enforceable outer deadline.
An incomplete or overdue completion must never authorize tool execution.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from importlib import metadata
import math
import re
import sys
import time
from typing import Any

from local_monitoring.parsing import Completion, ContextLimitError, GenerationTimeout


PREFILL_CHUNK_SIZE = 256
REQUIRED_TRANSFORMERS_VERSION = "4.57.1"
MIN_FREE_GPU_GIB = 6.0
_PACKAGES = (
    "torch", "transformers", "accelerate", "bitsandbytes", "huggingface-hub",
    "tokenizers", "safetensors", "pydantic", "pandas",
)
_NORMALIZED_THINKING = re.compile(r"<thinking>.*?</thinking>", flags=re.DOTALL)


def _versions() -> dict[str, str | None]:
    """An allowlist, never a dump of environment variables or credentials."""
    versions = {}
    for package in _PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _preserve_reasoning(messages: Sequence[Mapping[str, Any]], rendered: str) -> None:
    """Fail rather than silently dropping archived actor thinking.

    Native Qwen templates can strip old ``<think>`` blocks. The engine uses
    normalized ``<thinking>`` blocks in assistant content instead; verify
    those exact blocks survive rendering even when a new episode starts.
    """
    expected = Counter()
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        expected.update(_NORMALIZED_THINKING.findall(content))
    for block, count in expected.items():
        if rendered.count(block) < count:
            raise RuntimeError("Chat template dropped archived actor thinking; no inference performed")


class LocalQwenBackend:
    """Load one quantized CUDA model, then generate actor/monitor responses.

    There is one acting agent. The monitor uses the same local model weights
    in a separately prompted inference call, without sharing conversational
    KV caches. This is an explicit local implementation choice: the paper
    does not identify its online monitor model.
    """

    execution_mode = "local_hf_real_model"

    def __init__(self, config: Any) -> None:
        self.config = config
        versions = _versions()
        if versions["transformers"] != REQUIRED_TRANSFORMERS_VERSION:
            raise RuntimeError(
                f"Use Transformers {REQUIRED_TRANSFORMERS_VERSION}; run scripts/setup_qwen_gpu.sh first"
            )

        # Heavy libraries are intentionally imported only on explicit backend
        # initialization. Offline parsing and engine tests do not load them.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU is connected. Select a Colab GPU runtime before loading Qwen.")
        free, total = torch.cuda.mem_get_info(0)
        if free / 1024**3 < MIN_FREE_GPU_GIB:
            raise RuntimeError(
                f"Only {free / 1024**3:.2f} GiB GPU memory is free; at least {MIN_FREE_GPU_GIB:g} GiB is required"
            )

        # This checks cache correctness using tiny random CPU weights. It is
        # not a bank-task result, downloaded model test, or GPU validation.
        from OZLab_Qwen_Bank_Transfer import check_chunked_generation

        preflight = check_chunked_generation()
        if not preflight["successful"]:
            raise RuntimeError("Tiny CPU chunked-generation cache check failed; no task model was loaded")
        set_seed(config.seed)
        print(f"Loading {config.model_id} locally; no paid model API is used.", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, revision=config.revision, trust_remote_code=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=False,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
            dtype=torch.float16,
            device_map={"": 0},
            attn_implementation="sdpa",
        ).eval()
        resolved_revision = getattr(self.model.config, "_commit_hash", None)
        if resolved_revision != config.revision:
            raise RuntimeError("Loaded model revision differs from the configured immutable revision")
        if getattr(self.model.config, "model_type", None) != "qwen3":
            raise RuntimeError("This backend requires a Qwen3 causal language model")
        position_limit = getattr(self.model.config, "max_position_embeddings", None)
        self.context_limit = int(config.context_limit)
        if isinstance(position_limit, int) and position_limit > 0:
            self.context_limit = min(self.context_limit, position_limit)
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, (list, tuple, set)) else [eos]
        self.eos_token_ids = {int(token) for token in eos if token is not None}
        if not self.eos_token_ids:
            raise RuntimeError("Model generation configuration has no end-of-sequence token")
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id
        if self.pad_token_id is None:
            self.pad_token_id = min(self.eos_token_ids)
        torch.cuda.reset_peak_memory_stats(0)
        self._manifest = {
            "execution_mode": self.execution_mode,
            "model_responses_mocked": False,
            "model_api_calls": 0,
            "model_id": config.model_id,
            "requested_model_revision": config.revision,
            "resolved_model_revision": resolved_revision,
            "python": sys.version,
            "packages": versions,
            "cuda_available": True,
            "torch_cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "free_gpu_gib_before_loading": free / 1024**3,
            "total_gpu_gib": total / 1024**3,
            "quantization": "bitsandbytes NF4 4bit double-quant",
            "compute_dtype": "float16",
            "attention_implementation": "sdpa",
            "prefill_chunk_size": PREFILL_CHUNK_SIZE,
            "cache_implementation": "dynamic; new cache for each role call",
            "logits_to_keep": 1,
            "context_limit": self.context_limit,
            "prompt_truncation": False,
            "actor_thinking": True,
            "monitor_thinking": False,
            "actor_and_monitor_share_weights": True,
            "training_performed": False,
            "seed": config.seed,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": 0,
            "generation_timeout": "soft Transformers max_time; outer subprocess required for hard timeout",
            "generation_preflight": preflight,
        }

    def manifest(self) -> dict[str, Any]:
        """Return selected model/hardware metadata, including current peaks."""
        result = dict(self._manifest)
        result["peak_gpu_allocated_gib"] = self.torch.cuda.max_memory_allocated(0) / 1024**3
        return result

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        thinking: bool = False,
        max_new_tokens: int,
        role: str = "actor",
        timeout_seconds: float,
    ) -> Completion:
        """Generate one full-context response without tool execution.

        The engine owns tool execution and only accepts an EOS-complete,
        on-time response. No response repair, history truncation, remote
        inference, or role-to-role cache reuse occurs here.
        """
        if role not in {"actor", "monitor"}:
            raise ValueError("Role must be actor or monitor")
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise GenerationTimeout("No generation time budget remains")
        started = time.monotonic()
        rendered = self.tokenizer.apply_chat_template(
            list(messages),
            tools=list(tools) if tools is not None else None,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        _preserve_reasoning(messages, rendered)
        # The template already supplies its special-token text. Passing
        # add_special_tokens=False matches apply_chat_template(tokenize=True).
        inputs = self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        input_count = int(inputs["input_ids"].shape[-1])
        if input_count + max_new_tokens > self.context_limit:
            raise ContextLimitError(
                f"{role}: {input_count} input + {max_new_tokens} requested output tokens exceed "
                f"context limit {self.context_limit}; history was retained and inference was not started"
            )
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise GenerationTimeout("Generation deadline elapsed during full-context tokenization")
        # No past_key_values/cache object is supplied: each actor/monitor call
        # starts a fresh dynamic cache. The weight allocation alone is shared.
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs.to("cuda"),
                max_new_tokens=max_new_tokens,
                max_time=remaining,
                prefill_chunk_size=PREFILL_CHUNK_SIZE,
                use_cache=True,
                cache_implementation="dynamic",
                logits_to_keep=1,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=0,
                pad_token_id=self.pad_token_id,
            )
        output_ids = output[0, input_count:].tolist()
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        elapsed = time.monotonic() - started
        finished = bool(output_ids) and output_ids[-1] in self.eos_token_ids and elapsed < timeout_seconds
        opened_in_prompt = thinking and bool(re.search(r"<think>\s*$", rendered))
        return Completion(
            text=text,
            input_tokens=input_count,
            output_tokens=len(output_ids),
            finished=finished,
            elapsed_seconds=elapsed,
            thinking_opened_in_prompt=opened_in_prompt,
        )
