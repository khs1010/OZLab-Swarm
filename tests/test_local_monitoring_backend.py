"""Authored backend doubles, not Qwen inference or GPU performance results.

No torch/Transformers import is required for the main tests. One optional
test uses an already cached, pinned Qwen tokenizer with local_files_only;
it skips if the public tokenizer is unavailable, never downloads weights,
and does not run a model or a paid API.
"""

from __future__ import annotations

from contextlib import nullcontext
import copy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

import local_monitoring.backend as backend_module
from local_monitoring.backend import LocalQwenBackend, _preserve_reasoning
from local_monitoring.config import Config
from local_monitoring.parsing import ContextLimitError, GenerationTimeout
import OZLab_Qwen_Bank_Transfer as original_driver


@pytest.fixture
def authored_backend(monkeypatch):
    """Only Python doubles are injected; no real CUDA, weights, or inference."""
    trace = SimpleNamespace(
        tokenizer_loads=[], model_loads=[], quantization=[], seeds=[],
        templates=[], tokenizations=[], generations=[], devices=[],
        input_tokens=80, output_ids=[33, 2], raw="<think>authored</think>done",
        cuda_available=True, free_gib=14, preflight_ok=True,
        resolved_revision=Config().revision, model_type="qwen3",
    )

    class Inputs(dict):
        def to(self, device):
            trace.devices.append(device)
            return self

    class Output:
        def __getitem__(self, key):
            assert key[0] == 0 and key[1].start == trace.input_tokens
            return SimpleNamespace(tolist=lambda: list(trace.output_ids))

    class AuthoredTokenizer:
        eos_token_id = 2
        pad_token_id = None

        def apply_chat_template(self, messages, **kwargs):
            trace.templates.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
            # This is an authored template double, not the native Qwen Jinja.
            contents = "\n".join(str(message.get("content", "")) for message in messages)
            prefix = "<think>\n" if kwargs["enable_thinking"] else "<think>\n\n</think>\n\n"
            return contents + "\n<|im_start|>assistant\n" + prefix

        def __call__(self, text, **kwargs):
            trace.tokenizations.append((text, kwargs))
            return Inputs(input_ids=SimpleNamespace(shape=(1, trace.input_tokens)))

        def decode(self, ids, **kwargs):
            assert ids == trace.output_ids and kwargs == {"skip_special_tokens": True}
            return trace.raw

    class AuthoredModel:
        @property
        def config(self):
            return SimpleNamespace(
                _commit_hash=trace.resolved_revision, model_type=trace.model_type,
                max_position_embeddings=40960,
            )

        generation_config = SimpleNamespace(eos_token_id=[2, 3])

        def eval(self):
            return self

        def generate(self, **kwargs):
            trace.generations.append(kwargs)
            return Output()

    def tokenizer_load(model_id, **kwargs):
        trace.tokenizer_loads.append((model_id, kwargs))
        return AuthoredTokenizer()

    def model_load(model_id, **kwargs):
        trace.model_loads.append((model_id, kwargs))
        return AuthoredModel()

    def quantization_config(**kwargs):
        trace.quantization.append(kwargs)
        return SimpleNamespace(**kwargs)

    fake_torch = ModuleType("torch")
    fake_torch.float16 = "OFFLINE_DOUBLE_FLOAT16_NOT_REAL_TORCH"
    fake_torch.version = SimpleNamespace(cuda="OFFLINE_DOUBLE_NO_CUDA")
    fake_torch.inference_mode = nullcontext
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: trace.cuda_available,
        mem_get_info=lambda device: (trace.free_gib * 1024**3, 16 * 1024**3),
        get_device_name=lambda device: "OFFLINE DOUBLE A100 — NO GPU USED",
        reset_peak_memory_stats=lambda device: None,
        max_memory_allocated=lambda device: 0,
    )
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = SimpleNamespace(from_pretrained=tokenizer_load)
    fake_transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=model_load)
    fake_transformers.BitsAndBytesConfig = quantization_config
    fake_transformers.set_seed = trace.seeds.append
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(backend_module, "_versions", lambda: {
        "transformers": "4.57.1", "torch": "OFFLINE_DOUBLE_NOT_INSTALLED",
    })
    monkeypatch.setattr(original_driver, "check_chunked_generation", lambda: {
        "successful": trace.preflight_ok,
        "scope": "OFFLINE_STUB_NOT_A_REAL_PREFILL_CHECK",
    })
    return trace


def test_authored_initialization_preserves_quantization_and_single_model(authored_backend):
    trace = authored_backend
    backend = LocalQwenBackend(Config())
    assert len(trace.model_loads) == len(trace.tokenizer_loads) == 1
    assert trace.seeds == [42]
    model_id, options = trace.model_loads[0]
    assert model_id == "Qwen/Qwen3-4B"
    assert options["revision"] == Config().revision
    assert options["trust_remote_code"] is False
    assert options["device_map"] == {"": 0}
    assert options["attn_implementation"] == "sdpa"
    assert options["dtype"] == "OFFLINE_DOUBLE_FLOAT16_NOT_REAL_TORCH"
    assert trace.quantization == [{
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "OFFLINE_DOUBLE_FLOAT16_NOT_REAL_TORCH",
    }]
    assert backend.execution_mode == "local_hf_real_model"


@pytest.mark.parametrize("reason", ["no_cuda", "memory", "preflight", "revision", "model_type"])
def test_authored_initialization_rejects_incompatible_setup(authored_backend, reason):
    trace = authored_backend
    if reason == "no_cuda":
        trace.cuda_available = False
    elif reason == "memory":
        trace.free_gib = 2
    elif reason == "preflight":
        trace.preflight_ok = False
    elif reason == "revision":
        trace.resolved_revision = "0" * 40
    else:
        trace.model_type = "unrelated_model"
    with pytest.raises(RuntimeError):
        LocalQwenBackend(Config())
    if reason in {"no_cuda", "memory", "preflight"}:
        assert trace.model_loads == trace.tokenizer_loads == []


def test_version_check_prevents_loading_even_with_authored_backend(authored_backend, monkeypatch):
    monkeypatch.setattr(backend_module, "_versions", lambda: {"transformers": "0.0"})
    with pytest.raises(RuntimeError, match="Transformers 4.57.1"):
        LocalQwenBackend(Config())
    assert authored_backend.model_loads == []


def test_authored_generate_passes_settings_and_native_thinking_prefix(authored_backend):
    trace = authored_backend
    backend = LocalQwenBackend(Config())
    thinking = "<thinking>past reasoning retained</thinking>"
    messages = [{"role": "assistant", "content": thinking}]
    result = backend.generate(messages, thinking=True, max_new_tokens=128, timeout_seconds=10)
    assert result.finished and result.thinking_opened_in_prompt
    assert result.text == trace.raw
    assert result.input_tokens == 80 and result.output_tokens == 2
    assert thinking in trace.tokenizations[-1][0]
    assert trace.tokenizations[-1][1] == {"add_special_tokens": False, "return_tensors": "pt"}
    options = trace.generations[-1]
    assert options["prefill_chunk_size"] == 256
    assert options["logits_to_keep"] == 1 and options["cache_implementation"] == "dynamic"
    assert options["use_cache"] and options["do_sample"]
    assert options["temperature"] == options["top_p"] == 1.0 and options["top_k"] == 0
    assert "past_key_values" not in options
    assert trace.devices == ["cuda"]


def test_authored_monitor_reuses_weights_with_separate_context_and_cache(authored_backend):
    trace = authored_backend
    backend = LocalQwenBackend(Config())
    backend.generate([{"role": "user", "content": "actor"}], thinking=True,
                     max_new_tokens=64, timeout_seconds=10)
    result = backend.generate([{"role": "user", "content": "filtered monitor view"}],
                              role="monitor", thinking=False, max_new_tokens=32, timeout_seconds=10)
    assert result.finished and not result.thinking_opened_in_prompt
    assert len(trace.model_loads) == 1 and len(trace.generations) == 2
    assert trace.templates[1][0] == [{"role": "user", "content": "filtered monitor view"}]
    assert trace.templates[1][1]["enable_thinking"] is False
    assert all("past_key_values" not in options for options in trace.generations)


def test_authored_context_limit_prevents_generation_without_truncation(authored_backend):
    trace = authored_backend
    trace.input_tokens = 1000
    backend = LocalQwenBackend(Config(context_limit=3000))
    with pytest.raises(ContextLimitError, match="history was retained"):
        backend.generate([{"role": "user", "content": "full retained history"}],
                         max_new_tokens=2500, timeout_seconds=10)
    assert trace.generations == []
    assert "full retained history" in trace.tokenizations[0][0]


def test_authored_no_eos_and_soft_deadline_are_not_complete(authored_backend, monkeypatch):
    trace = authored_backend
    backend = LocalQwenBackend(Config())
    trace.output_ids = [17, 999]
    incomplete = backend.generate([], max_new_tokens=2, timeout_seconds=10)
    assert incomplete.finished is False
    trace.output_ids = [17, 2]
    clock = iter([100, 100.1, 101])
    monkeypatch.setattr(backend_module.time, "monotonic", lambda: next(clock))
    overdue = backend.generate([], max_new_tokens=2, timeout_seconds=1)
    assert overdue.finished is False and overdue.elapsed_seconds == 1


def test_authored_tokenization_deadline_prevents_generation(authored_backend, monkeypatch):
    trace = authored_backend
    backend = LocalQwenBackend(Config())
    clock = iter([100, 102])
    monkeypatch.setattr(backend_module.time, "monotonic", lambda: next(clock))
    with pytest.raises(GenerationTimeout, match="tokenization"):
        backend.generate([], max_new_tokens=2, timeout_seconds=1)
    assert trace.generations == []


def test_history_preservation_including_duplicate_thinking_blocks():
    history = [{"role": "assistant", "content": "<thinking>past</thinking>"}] * 2
    _preserve_reasoning(history, "<thinking>past</thinking>" * 2)
    with pytest.raises(RuntimeError, match="dropped archived"):
        _preserve_reasoning(history, "<thinking>past</thinking>")


def test_authored_manifest_does_not_copy_credentials(authored_backend, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "OFFLINE_TEST_SECRET_NEVER_LOG")
    backend = LocalQwenBackend(Config())
    manifest = backend.manifest()
    assert "OFFLINE_TEST_SECRET_NEVER_LOG" not in json.dumps(manifest)
    assert manifest["model_api_calls"] == 0
    assert manifest["actor_and_monitor_share_weights"] is True
    assert manifest["prompt_truncation"] is False
    assert manifest["generation_preflight"]["scope"] == "OFFLINE_STUB_NOT_A_REAL_PREFILL_CHECK"


def test_optional_cached_pinned_qwen_tokenizer_full_history_without_model(monkeypatch):
    """Actual native template only if cached; no network, GPU, or model load."""
    root = Path(__file__).resolve().parents[1]
    snapshot = root / ".cache/hf-tokenizer-validation/hub/models--Qwen--Qwen3-4B/snapshots" / Config().revision
    if not (snapshot / "tokenizer_config.json").exists():
        pytest.skip("Pinned Qwen tokenizer is not cached; network download was blocked by proxy 403")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot, local_files_only=True,
                                                          trust_remote_code=False)
    from scripts.smoke_bank_transfer import load_upstream, new_world

    _, _, runtime = new_world(load_upstream())
    tools = [{"type": "function", "function": {
        "name": function.name,
        "description": function.full_docstring,
        "parameters": function.parameters.model_json_schema(),
    }} for function in runtime.functions.values()]
    assert len(tools) == 71
    history = [
        {"role": "system", "content": "Tokenizer-only template validation."},
        {"role": "user", "content": "Episode 1"},
        {"role": "assistant", "content": "<thinking>retained old reasoning</thinking>done"},
        {"role": "user", "content": "Episode 2"},
    ]
    rendered = tokenizer.apply_chat_template(history, tools=tools, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=True)
    _preserve_reasoning(history, rendered)
    assert "<thinking>retained old reasoning</thinking>" in rendered
    tokenized = tokenizer.apply_chat_template(history, tools=tools, tokenize=True,
                                              add_generation_prompt=True, enable_thinking=True)
    assert tokenized == tokenizer(rendered, add_special_tokens=False)["input_ids"]
