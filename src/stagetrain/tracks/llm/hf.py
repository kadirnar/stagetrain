"""The `hf` backend: plain transformers + PEFT + TRL + accelerate, with no Unsloth.

It trains on several GPUs under `accelerate launch` or `torchrun` (DDP, or FSDP through
accelerate), on any hardware transformers runs on, and it is the backend the tests can actually
run: the same stages, on a tiny random model, on a CPU.

What it does not do: Unsloth's fused kernels and memory savings, vLLM generation for GRPO, GGUF
export and 4-bit merged export. For those, use `backend: unsloth`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from stagetrain.core.launch import fsdp_requested, in_worker, is_main_process, world_size
from stagetrain.core.training import final_metrics

from .backend import (
    MERGE_METHODS,
    LoadSpec,
    ResponsesOnly,
    TrainOutput,
    adapter_base,
    fit_config_kwargs,
    has_checkpoint,
    is_adapter_dir,
)

# What `sft_config_kwargs` adds for TRL's SFTConfig; this backend trains text with `Trainer`.
SFT_ONLY = ("dataset_text_field", "max_length", "packing", "embedding_learning_rate", "loraplus_lr_ratio")
EMBEDDING_PARAMETERS = ("embed_tokens", "lm_head")


# --------------------------------------------------------------------------- #
# loss on the assistant turns only
# --------------------------------------------------------------------------- #


def message_spans(text: str, messages: list[dict[str, str]]) -> list[tuple[int, int, str]]:
    """`(start, end, role)` of every message's content inside the rendered conversation, found in
    order: each message is looked for after the end of the one before it."""
    spans, cursor = [], 0
    for message in messages:
        content = message["content"]
        if not content:
            continue
        start = text.find(content, cursor)
        if start < 0:  # the template rewrote this message (trimmed it, escaped it): no span for it
            continue
        spans.append((start, start + len(content), message["role"]))
        cursor = start + len(content)
    return spans


def response_labels(
    tokenizer, text: str, messages: list[dict[str, str]], max_length: int, last_response_only: bool = False
) -> tuple[list[int], list[int]]:
    """`(input_ids, labels)` with the loss on assistant content only: every other token is -100.

    The end-of-turn token that follows an assistant message is kept in the loss, so the model
    learns to stop. Works from character offsets, so it needs no knowledge of the chat template.
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=True, max_length=max_length)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    spans = [(a, b) for a, b, role in message_spans(text, messages) if role == "assistant"]
    if last_response_only:
        spans = spans[-1:]
    labels = [-100] * len(ids)
    special = set(tokenizer.all_special_ids)
    for start, end in spans:
        last = None
        for i, (a, b) in enumerate(offsets):
            if b > a and a >= start and b <= end:
                labels[i] = ids[i]
                last = i
        if last is not None:
            j = last + 1
            while j < len(ids) and ids[j] in special:  # <|im_end|>, <|eot_id|>, <end_of_turn>, ...
                labels[j] = ids[j]
                j += 1
    return ids, labels


def encode_text(tokenizer, text: str, max_length: int, chat: bool) -> tuple[list[int], list[int]]:
    """`(input_ids, labels)` with the loss on every token. A chat template writes its own special
    tokens; raw text gets the tokenizer's."""
    ids = tokenizer(text, add_special_tokens=not chat, truncation=True, max_length=max_length)["input_ids"]
    return ids, list(ids)


class CausalCollator:
    """Right-pads `input_ids` and `labels` (with -100) to the longest row of the batch."""

    def __init__(self, pad_token_id: int):
        self.pad = pad_token_id

    def __call__(self, rows: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        width = max(len(r["input_ids"]) for r in rows)
        ids = torch.full((len(rows), width), self.pad, dtype=torch.long)
        labels = torch.full((len(rows), width), -100, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            n = len(r["input_ids"])
            ids[i, :n] = torch.tensor(r["input_ids"])
            labels[i, :n] = torch.tensor(r["labels"])
            mask[i, :n] = 1
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


class Rows:
    """A list of tokenised rows as a map-style dataset."""

    def __init__(self, rows: list[dict[str, list[int]]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, list[int]]:
        return self.rows[i]


# --------------------------------------------------------------------------- #
# the backend
# --------------------------------------------------------------------------- #


class HfBackend:
    name = "hf"

    def _prepare(self) -> None:
        """Hook for a subclass that must import something before transformers and trl."""

    # ------------------------------------------------------------------ model

    @staticmethod
    def _device():
        """This process's device: its own GPU in a distributed run, the first GPU otherwise."""
        import torch

        if not torch.cuda.is_available():
            return None
        rank = os.environ.get("LOCAL_RANK", "0") if in_worker() else "0"
        return f"cuda:{int(rank) if rank.isdigit() else 0}"

    def bf16_supported(self) -> bool | None:
        import torch

        if not torch.cuda.is_available():
            return None  # no GPU: `TrainArgs` resolves `auto` to fp32
        return bool(torch.cuda.is_bf16_supported())

    def load(self, spec: LoadSpec):
        """`(model, tokenizer)`. A LoRA adapter directory loads its base model with the adapter on
        top, trainable. The model goes to this process's own device: never `device_map="auto"`,
        which would spread one copy over the GPUs a distributed run gives one process each."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._prepare()
        self.release()
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")  # TRL reports usage unless this is set
        if spec.fast_inference:
            raise ValueError("fast_inference (vLLM generation) belongs to the unsloth backend")
        path = spec.model_name
        adapter = is_adapter_dir(path)
        base = adapter_base(path) if adapter else path
        if adapter and not base:
            raise ValueError(f"{path}: adapter_config.json names no base model")
        has_tokenizer = adapter and (Path(path) / "tokenizer_config.json").is_file()
        tokenizer = AutoTokenizer.from_pretrained(path if has_tokenizer else base, trust_remote_code=spec.trust_remote_code)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        device = self._device()
        kwargs: dict[str, Any] = {"trust_remote_code": spec.trust_remote_code}
        if device is None:
            kwargs["dtype"] = torch.float32
        else:
            kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            kwargs["device_map"] = {"": device}
        quantized = spec.load_in in ("4bit", "8bit") and not spec.full_finetuning
        if quantized:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = (
                BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=kwargs["dtype"],
                )
                if spec.load_in == "4bit" else BitsAndBytesConfig(load_in_8bit=True)
            )
        model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
        if quantized:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, path, is_trainable=True)
        return model, tokenizer

    def has_adapter(self, model) -> bool:
        return hasattr(model, "peft_config")

    def attach_lora(self, model, kwargs: dict[str, Any]):
        """A new LoRA adapter. `kwargs` are `peft_kwargs`'s, which follow Unsloth's names."""
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            task_type="CAUSAL_LM", r=kwargs["r"], lora_alpha=kwargs["lora_alpha"], lora_dropout=kwargs["lora_dropout"],
            bias=kwargs["bias"], target_modules=kwargs["target_modules"], use_rslora=kwargs["use_rslora"],
            modules_to_save=kwargs.get("modules_to_save"),
        )
        return get_peft_model(model, config)

    def set_chat_template(self, tokenizer, name: str):
        raise ValueError(
            f"chat_template: {name!r} is one of Unsloth's named templates; with backend: hf use a tokenizer that "
            "carries its own chat template"
        )

    def release(self) -> None:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def count_tokens(self, tokenizer, texts: list[str]) -> list[int]:
        if not texts:
            return []
        return [len(ids) for ids in tokenizer(list(texts), add_special_tokens=False)["input_ids"]]

    # --------------------------------------------------------------- training

    @staticmethod
    def _callbacks(history_dir: str | Path | None) -> list[Any]:
        if history_dir is None:
            return []
        from stagetrain.core.history import history_callback

        return [history_callback(history_dir)]

    @staticmethod
    def _arguments(config_kwargs: dict[str, Any], cls: type):
        """`TrainingArguments` (or a TRL config) from the stage's keyword arguments."""
        import importlib.util

        import torch

        kwargs = fit_config_kwargs(dict(config_kwargs), cls)
        if kwargs.get("optim") == "adamw_8bit" and (
            importlib.util.find_spec("bitsandbytes") is None or not torch.cuda.is_available()
        ):
            kwargs["optim"] = "adamw_torch"  # the 8-bit optimizer needs bitsandbytes and a GPU
        # A PEFT model is not a PreTrainedModel, so the Trainer would look for unused parameters
        # at every step of a distributed run; with LoRA there are none to find.
        kwargs.setdefault("ddp_find_unused_parameters", False)
        if fsdp_requested():
            kwargs.setdefault("fsdp", True)
            kwargs.setdefault("fsdp_config", {"version": 2})
        if not torch.cuda.is_available():
            # Without CUDA the Trainer would pick another accelerator if it finds one (Apple's MPS),
            # which this backend does not load models for. Train where the model is: on the CPU.
            kwargs.setdefault("use_cpu", True)
        args = cls(**kwargs)
        if world_size() == 1 and getattr(args, "n_gpu", 1) > 1:
            # One process that sees several GPUs: the Trainer would wrap the model in DataParallel,
            # which TRL's trainers refuse. Train on this process's GPU; `launch: {num_processes: N}`
            # is how the others are used.
            args._n_gpu = 1
        return args

    def _dataset(self, tokenizer, texts, conversations, max_length: int, responses_only: ResponsesOnly | None) -> Rows:
        chat = conversations is not None
        rows = []
        for i, text in enumerate(texts):
            if responses_only is not None:
                if not chat:
                    raise ValueError("loss on the responses only needs the conversations the texts were rendered from")
                ids, labels = response_labels(
                    tokenizer, text, conversations[i], max_length, responses_only.last_response_only
                )
                if all(x == -100 for x in labels):
                    continue  # the response was cut off entirely: nothing to learn from this row
            else:
                ids, labels = encode_text(tokenizer, text, max_length, chat)
            rows.append({"input_ids": ids, "labels": labels})
        return Rows(rows)

    @staticmethod
    def _embedding_optimizer(model, args, embedding_learning_rate: float):
        """AdamW with a lower learning rate for the embeddings and the output head, as the
        continued-pretraining recipe asks."""
        import torch

        slow, rest = [], []
        for name, param in model.named_parameters():
            if param.requires_grad:
                (slow if any(key in name for key in EMBEDDING_PARAMETERS) else rest).append(param)
        groups = [{"params": rest, "lr": args.learning_rate}]
        if slow:
            groups.append({"params": slow, "lr": embedding_learning_rate})
        return torch.optim.AdamW(
            groups, lr=args.learning_rate, weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_epsilon,
        )

    @staticmethod
    def _loraplus_optimizer(model, args, ratio: float):
        """LoRA+: AdamW with the B matrices at `ratio` times the learning rate of the A matrices."""
        import torch
        from peft import PeftModel
        from peft.optimizers import create_loraplus_optimizer

        if not isinstance(model, PeftModel):
            raise ValueError("loraplus_lr_ratio needs a LoRA adapter; this stage trains every weight")
        return create_loraplus_optimizer(
            model, torch.optim.AdamW, lr=args.learning_rate, loraplus_lr_ratio=ratio,
            loraplus_weight_decay=args.weight_decay, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_epsilon,
        )

    def _text_trainer(
        self, model, tokenizer, train_texts, eval_texts, config_kwargs, responses_only, history_dir=None,
        conversations=None, eval_conversations=None,
    ):
        from transformers import Trainer, TrainingArguments

        if responses_only is not None and not getattr(tokenizer, "is_fast", False):
            raise ValueError("loss on the responses only needs a fast tokenizer (character offsets)")
        if config_kwargs.get("packing"):
            raise ValueError("packing is not available with backend: hf")
        max_length = int(config_kwargs.get("max_length") or tokenizer.model_max_length)
        embedding_lr = config_kwargs.get("embedding_learning_rate")
        args = self._arguments({k: v for k, v in config_kwargs.items() if k not in SFT_ONLY}, TrainingArguments)
        train = self._dataset(tokenizer, train_texts, conversations, max_length, responses_only)
        evaluation = self._dataset(tokenizer, eval_texts, eval_conversations, max_length, responses_only)
        if not len(train):
            raise ValueError("no training row is left after tokenisation")
        optimizers: tuple[Any, Any] = (None, None)
        if config_kwargs.get("loraplus_lr_ratio") is not None:
            optimizers = (self._loraplus_optimizer(model, args, config_kwargs["loraplus_lr_ratio"]), None)
        elif embedding_lr is not None:
            optimizers = (self._embedding_optimizer(model, args, embedding_lr), None)
        return Trainer(
            model=model, args=args, train_dataset=train, eval_dataset=evaluation if len(evaluation) else None,
            data_collator=CausalCollator(tokenizer.pad_token_id), callbacks=self._callbacks(history_dir),
            optimizers=optimizers,
            processing_class=tokenizer,  # so that every checkpoint carries its tokenizer and loads on its own
        )

    def train_text(
        self, model, tokenizer, train_texts: list[str], eval_texts: list[str], config_kwargs: dict[str, Any], *,
        responses_only: ResponsesOnly | None = None, resume: bool = True, history_dir: str | Path | None = None,
        conversations: list[list[dict[str, str]]] | None = None,
        eval_conversations: list[list[dict[str, str]]] | None = None,
    ) -> TrainOutput:
        """Causal-LM training on rendered text with `transformers.Trainer`. Continued pretraining
        puts the loss on every token; chat fine-tuning on the assistant turns."""
        self._prepare()
        trainer = self._text_trainer(
            model, tokenizer, train_texts, eval_texts, config_kwargs, responses_only, history_dir, conversations,
            eval_conversations,
        )
        return self._fit(trainer, config_kwargs["output_dir"], trainer.eval_dataset is not None, resume)

    def train_dpo(
        self, model, tokenizer, train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]],
        config_kwargs: dict[str, Any], *, resume: bool = True, history_dir: str | Path | None = None,
    ) -> TrainOutput:
        self._prepare()
        from datasets import Dataset
        from trl import DPOConfig, DPOTrainer

        trainer = DPOTrainer(
            model=model, ref_model=None, args=self._arguments(config_kwargs, DPOConfig),
            train_dataset=Dataset.from_list(list(train_rows)),
            eval_dataset=Dataset.from_list(list(eval_rows)) if eval_rows else None,
            processing_class=tokenizer, callbacks=self._callbacks(history_dir),
        )
        return self._fit(trainer, config_kwargs["output_dir"], bool(eval_rows), resume)

    def train_grpo(
        self, model, tokenizer, rows: list[dict[str, Any]], reward_funcs: list[Any], config_kwargs: dict[str, Any], *,
        resume: bool = True, history_dir: str | Path | None = None,
    ) -> TrainOutput:
        self._prepare()
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer

        trainer = GRPOTrainer(
            model=model, processing_class=tokenizer, reward_funcs=list(reward_funcs),
            args=self._arguments(config_kwargs, GRPOConfig), train_dataset=Dataset.from_list(list(rows)),
            callbacks=self._callbacks(history_dir),
        )
        return self._fit(trainer, config_kwargs["output_dir"], False, resume)

    def train_distill(
        self, model, tokenizer, rows: list[dict[str, Any]], config_kwargs: dict[str, Any], *,
        resume: bool = True, history_dir: str | Path | None = None,
    ) -> TrainOutput:
        """On-policy distillation from a teacher's logits (TRL's `DistillationTrainer`)."""
        self._prepare()
        from datasets import Dataset
        from trl import DistillationConfig, DistillationTrainer

        trainer = DistillationTrainer(
            model=model, processing_class=tokenizer, args=self._arguments(config_kwargs, DistillationConfig),
            # The trainer loads the teacher from this argument; the config field alone is not read.
            teacher_model=config_kwargs["teacher_model_name_or_path"],
            train_dataset=Dataset.from_list(list(rows)), callbacks=self._callbacks(history_dir),
        )
        return self._fit(trainer, config_kwargs["output_dir"], False, resume)

    @staticmethod
    def _fit(trainer, output_dir: str, has_eval: bool, resume: bool) -> TrainOutput:
        result = trainer.train(resume_from_checkpoint=True if resume and has_checkpoint(output_dir) else None)
        metrics = dict(result.metrics)
        if has_eval and is_main_process():
            metrics.update(trainer.evaluate())
        return TrainOutput(final_metrics(metrics), list(trainer.state.log_history))

    # ------------------------------------------------------------- evaluation

    def evaluate_loss(
        self, model, tokenizer, texts: list[str], config_kwargs: dict[str, Any], *,
        responses_only: ResponsesOnly | None = None, conversations: list[list[dict[str, str]]] | None = None,
    ) -> dict[str, float]:
        """Held-out loss of `texts`, with the same masking as training."""
        self._prepare()
        trainer = self._text_trainer(
            model, tokenizer, texts, texts, config_kwargs, responses_only, None, conversations, conversations
        )
        return final_metrics(trainer.evaluate())

    def sample(
        self, model, tokenizer, conversations: list[list[dict[str, str]]], *, n: int = 1, max_new_tokens: int = 256,
        temperature: float = 0.0, top_p: float = 1.0, template_kwargs: dict[str, Any] | None = None,
    ) -> list[list[str]]:
        """`n` replies to each conversation, one conversation at a time; greedy when `temperature` is 0."""
        import torch

        if n > 1 and temperature <= 0:
            raise ValueError("several samples per prompt need temperature > 0")
        model.eval()
        sampling = {"do_sample": True, "temperature": temperature, "top_p": top_p} if temperature > 0 else {"do_sample": False}
        device = next(model.parameters()).device
        out = []
        for messages in conversations:
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                **(template_kwargs or {}),
            ).to(device)
            with torch.no_grad():
                ids = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, num_return_sequences=n,
                    pad_token_id=tokenizer.pad_token_id, **sampling,
                )
            start = inputs["input_ids"].shape[1]
            out.append([tokenizer.decode(row[start:], skip_special_tokens=True).strip() for row in ids])
        return out

    def generate(
        self, model, tokenizer, conversations: list[list[dict[str, str]]], *, max_new_tokens: int = 256,
        temperature: float = 0.0, top_p: float = 1.0, template_kwargs: dict[str, Any] | None = None,
    ) -> list[str]:
        replies = self.sample(
            model, tokenizer, conversations, n=1, max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, template_kwargs=template_kwargs,
        )
        return [r[0] for r in replies]

    # ----------------------------------------------------------------- saving

    def save(self, model, tokenizer, out_dir: str | Path) -> None:
        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

    def save_merged(self, model, tokenizer, out_dir: str | Path, method: str = "merged_16bit") -> None:
        """The adapter folded into the base weights. This consumes the loaded adapter: load the
        model again to go on using it."""
        if method not in MERGE_METHODS:
            raise ValueError(f"merge method must be one of {MERGE_METHODS}, got {method!r}")
        if method != "merged_16bit":
            raise ValueError("backend: hf writes merged_16bit only; merged_4bit is the unsloth backend's")
        merged = model.merge_and_unload() if self.has_adapter(model) else model
        merged.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

    def save_gguf(self, model, tokenizer, out_dir: str | Path, quantization: list[str]) -> tuple[str, list[str]]:
        raise ValueError(
            "GGUF export is the unsloth backend's; with backend: hf export merged_16bit and convert it with llama.cpp"
        )

    def push(
        self, model, tokenizer, repo_id: str, method: str, *, private: bool = True, quantization: list[str] | None = None,
    ) -> None:
        if method not in ("lora", "merged_16bit"):
            raise ValueError(f"backend: hf can push lora and merged_16bit, not {method}")
        model.push_to_hub(repo_id, private=private)
        tokenizer.push_to_hub(repo_id, private=private)
