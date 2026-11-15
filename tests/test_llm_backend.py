"""Stage parameters -> Unsloth / TRL keyword arguments. Unsloth itself is never imported: its
parameter names below were read from the unsloth 2026.11.4 source."""

import dataclasses
import json

import pytest

from stagetrain.tracks.llm.backend import (
    CptLora,
    CptTrainArgs,
    DpoTrainArgs,
    LlmLora,
    LlmTrainArgs,
    LoadSpec,
    UnslothBackend,
    adapter_base,
    adapter_trains_embeddings,
    dpo_config_kwargs,
    fit_config_kwargs,
    get_backend,
    has_checkpoint,
    is_adapter_dir,
    load_kwargs,
    peft_kwargs,
    set_backend,
    sft_config_kwargs,
)

# FastLanguageModel.from_pretrained / get_peft_model parameters (unsloth 2026.11.4).
FROM_PRETRAINED = {
    "model_name", "max_seq_length", "dtype", "load_in_4bit", "load_in_8bit", "load_in_16bit", "full_finetuning",
    "token", "device_map", "rope_scaling", "fix_tokenizer", "trust_remote_code", "use_gradient_checkpointing",
    "resize_model_vocab", "revision", "random_state", "max_lora_rank", "qat_scheme", "load_in_fp8", "text_only",
}
GET_PEFT_MODEL = {
    "r", "target_modules", "lora_alpha", "lora_dropout", "bias", "layers_to_transform", "layers_pattern",
    "finetune_last_n_layers", "use_gradient_checkpointing", "random_state", "max_seq_length", "use_rslora",
    "modules_to_save", "init_lora_weights", "loftq_config", "qat_scheme", "target_parameters",
}


def test_load_kwargs():
    kw = load_kwargs(LoadSpec("org/model", max_seq_length=4096))
    assert set(kw) <= FROM_PRETRAINED
    assert kw["model_name"] == "org/model" and kw["max_seq_length"] == 4096
    assert (kw["load_in_4bit"], kw["load_in_8bit"], kw["load_in_16bit"], kw["full_finetuning"]) == (True, False, False, False)
    kw = load_kwargs(LoadSpec("m", load_in="8bit"))
    assert (kw["load_in_4bit"], kw["load_in_8bit"], kw["load_in_16bit"]) == (False, True, False)
    kw = load_kwargs(LoadSpec("m", load_in="16bit"))
    assert (kw["load_in_4bit"], kw["load_in_8bit"], kw["load_in_16bit"]) == (False, False, True)
    kw = load_kwargs(LoadSpec("m", load_in="16bit", full_finetuning=True))
    assert kw["full_finetuning"] and not (kw["load_in_4bit"] or kw["load_in_8bit"] or kw["load_in_16bit"])
    with pytest.raises(ValueError, match="needs load_in: 16bit"):
        load_kwargs(LoadSpec("m", full_finetuning=True))
    with pytest.raises(ValueError, match="load_in must be"):
        load_kwargs(LoadSpec("m", load_in="2bit"))


def test_peft_kwargs():
    kw = peft_kwargs(LlmLora(), seed=7)
    assert set(kw) <= GET_PEFT_MODEL
    assert kw["r"] == 16 and kw["lora_alpha"] == 16 and kw["random_state"] == 7 and kw["modules_to_save"] is None
    assert kw["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    assert kw["use_gradient_checkpointing"] == "unsloth"
    # a config that only sets the rank keeps Unsloth's modules
    assert LlmLora.model_validate({"r": 64}).target_modules == kw["target_modules"]

    cpt = peft_kwargs(CptLora(), seed=1, train_embeddings=True)
    assert cpt["r"] == 128 and cpt["lora_alpha"] == 32 and cpt["use_rslora"] is True
    assert cpt["modules_to_save"] == ["embed_tokens", "lm_head"]
    already = peft_kwargs(LlmLora(modules_to_save=["lm_head"]), seed=1, train_embeddings=True)
    assert already["modules_to_save"] == ["lm_head", "embed_tokens"]
    assert peft_kwargs(LlmLora(target_modules="all-linear"), seed=1)["target_modules"] == "all-linear"


def test_track_defaults_follow_unsloths_recipes():
    assert LlmTrainArgs().learning_rate == 2e-4 and LlmTrainArgs().optim == "adamw_8bit"
    assert LlmTrainArgs.model_validate({"num_train_epochs": 3}).learning_rate == 2e-4  # a partial dict keeps them
    assert CptTrainArgs().learning_rate == 5e-5 and DpoTrainArgs().learning_rate == 5e-6


def test_sft_config_kwargs(tmp_path):
    kw = sft_config_kwargs(
        LlmTrainArgs(eval_steps=50), output_dir=tmp_path, seed=3, has_eval=True, max_seq_length=1024,
        run_name="r", packing=True, embedding_learning_rate=1e-5, bf16=False,
    )
    assert kw["dataset_text_field"] == "text" and kw["max_length"] == 1024 and kw["packing"] is True
    assert kw["embedding_learning_rate"] == 1e-5 and kw["output_dir"] == str(tmp_path) and kw["seed"] == 3
    assert (kw["bf16"], kw["fp16"]) == (False, True)  # Unsloth's rule on a device without bf16
    assert kw["eval_strategy"] == "steps" and kw["eval_steps"] == 50 and kw["optim"] == "adamw_8bit"
    plain = sft_config_kwargs(LlmTrainArgs(precision="fp32"), output_dir=tmp_path, seed=3, has_eval=False,
                              max_seq_length=512, bf16=True)
    assert (plain["bf16"], plain["fp16"]) == (False, False) and "embedding_learning_rate" not in plain
    assert plain["eval_strategy"] == "no"
    extra = sft_config_kwargs(LlmTrainArgs(extra={"max_length": 99}), output_dir=tmp_path, seed=3, has_eval=False,
                              max_seq_length=512)
    assert extra["max_length"] == 99  # `extra` wins over what the stage derives


def test_config_kwargs_build_the_installed_trl_configs(tmp_path):
    pytest.importorskip("torch")
    trl = pytest.importorskip("trl")
    kw = sft_config_kwargs(LlmTrainArgs(optim="adamw_torch", group_by_length=True), output_dir=tmp_path, seed=3,
                           has_eval=True, max_seq_length=1024, bf16=False)
    kw["fp16"] = False  # no CUDA on the test machine
    cfg = trl.SFTConfig(**fit_config_kwargs(kw, trl.SFTConfig))
    assert cfg.max_length == 1024 and cfg.dataset_text_field == "text" and cfg.learning_rate == 2e-4
    kw = dpo_config_kwargs(DpoTrainArgs(optim="adamw_torch"), output_dir=tmp_path, seed=3, has_eval=False,
                           max_seq_length=768, beta=0.25, bf16=False)
    kw["fp16"] = False
    assert "loss_type" not in kw
    cfg = trl.DPOConfig(**fit_config_kwargs(kw, trl.DPOConfig))
    assert cfg.beta == 0.25 and cfg.max_length == 768 and cfg.learning_rate == 5e-6
    assert dpo_config_kwargs(DpoTrainArgs(), output_dir=tmp_path, seed=3, has_eval=False, max_seq_length=8,
                             loss_type="ipo")["loss_type"] == "ipo"


def test_fit_config_kwargs_maps_names_back_for_older_releases():
    @dataclasses.dataclass
    class Old:  # the names transformers 4.x / trl 0.1x used
        output_dir: str = ""
        warmup_ratio: float = 0.0
        warmup_steps: int = 0
        group_by_length: bool = False
        evaluation_strategy: str = "no"
        max_seq_length: int = 1024
        gradient_checkpointing: bool = False

    new = {"output_dir": "o", "warmup_steps": 0.05, "train_sampling_strategy": "group_by_length",
           "eval_strategy": "steps", "max_length": 512, "activation_checkpointing": True, "seed": 1}
    assert fit_config_kwargs(new, Old) == {
        "output_dir": "o", "warmup_ratio": 0.05, "group_by_length": True, "evaluation_strategy": "steps",
        "max_seq_length": 512, "gradient_checkpointing": True, "seed": 1,
    }
    assert "group_by_length" not in fit_config_kwargs({"train_sampling_strategy": "random"}, Old)
    assert fit_config_kwargs({"warmup_steps": 100}, Old) == {"warmup_steps": 100}

    @dataclasses.dataclass
    class New:
        warmup_steps: float = 0
        train_sampling_strategy: str = "random"
        eval_strategy: str = "no"
        max_length: int = 1024
        activation_checkpointing: bool = False

    assert fit_config_kwargs(new, New) == new  # nothing to rename on a current release


def test_adapter_directory_helpers(tmp_path):
    assert not is_adapter_dir(tmp_path) and not is_adapter_dir("org/model")
    assert adapter_base(tmp_path) is None and not adapter_trains_embeddings(tmp_path)
    (tmp_path / "adapter_config.json").write_text(json.dumps(
        {"base_model_name_or_path": "org/base", "modules_to_save": ["embed_tokens", "lm_head"]}))
    assert is_adapter_dir(tmp_path) and adapter_base(tmp_path) == "org/base" and adapter_trains_embeddings(tmp_path)
    (tmp_path / "adapter_config.json").write_text(json.dumps({"modules_to_save": None}))
    assert not adapter_trains_embeddings(tmp_path)
    assert not has_checkpoint(tmp_path / "missing") and not has_checkpoint(tmp_path)
    (tmp_path / "checkpoint-10").mkdir()
    assert has_checkpoint(tmp_path)


def test_backend_can_be_replaced():
    fake = object()
    set_backend(fake)
    try:
        assert get_backend() is fake
    finally:
        set_backend(None)
    assert isinstance(get_backend(), UnslothBackend)
    set_backend(None)
