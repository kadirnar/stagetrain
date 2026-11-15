"""A stand-in for the Unsloth backend: it records what the stages ask for and writes the files a
real backend would leave behind. No Unsloth, no weights, no network."""

import json
from pathlib import Path

from stagetrain.tracks.llm.backend import TrainOutput, is_adapter_dir


class FakeTokenizer:
    eos_token = "</s>"
    chat_template = "fake"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        assert tokenize is False
        text = "".join(f"<|{m['role']}|>{m['content']}</s>" for m in messages)
        if kwargs.get("enable_thinking") is False:
            text += "<nothink>"
        return text + ("<|assistant|>" if add_generation_prompt else "")


class FakeModel:
    def __init__(self, name, adapter, full):
        self.name, self.adapter, self.full, self.lora = name, adapter, full, None


class FakeBackend:
    """Records what the stages ask for and writes the files a real backend would leave behind."""

    def __init__(self, eval_loss=None):
        self.calls = []
        self.eval_loss = eval_loss or {}  # substring of the model's name -> loss
        self.checkpoint_steps = []  # steps a training call leaves a checkpoint at
        self.skill = {}  # substring of the model's name -> how many of 10 questions it gets right
        self.loss_by_lr = {}  # learning rate -> the eval loss a training run with it ends at

    def _loss(self, model):
        for key, value in self.eval_loss.items():
            if key in model.name:
                return value
        return 1.0

    def load(self, spec):
        self.calls.append(("load", spec.model_name, spec.load_in, spec.full_finetuning))
        return FakeModel(spec.model_name, is_adapter_dir(spec.model_name), spec.full_finetuning), FakeTokenizer()

    def has_adapter(self, model):
        return model.adapter

    def attach_lora(self, model, kwargs):
        self.calls.append(("attach_lora", kwargs))
        model.adapter, model.lora = True, kwargs
        return model

    def set_chat_template(self, tokenizer, name):
        self.calls.append(("chat_template", name))
        tokenizer.chat_template = name
        return tokenizer

    def bf16_supported(self):
        return True

    def release(self):
        self.calls.append(("release",))

    def count_tokens(self, tokenizer, texts):
        return [len(t.split()) for t in texts]  # one token per word is enough for the length filter

    def _checkpoints(self, model, config):
        for step in self.checkpoint_steps:
            path = Path(config["output_dir"], f"checkpoint-{step}")
            path.mkdir(parents=True, exist_ok=True)
            (path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": model.name}))

    def train_text(self, model, tokenizer, train_texts, eval_texts, config, *, responses_only=None, resume=True,
                   history_dir=None):
        self.calls.append(("train_text", {
            "model": model.name, "train": list(train_texts), "eval": list(eval_texts), "config": config,
            "responses_only": responses_only, "resume": resume, "history_dir": history_dir,
        }))
        self._checkpoints(model, config)
        loss = self.loss_by_lr.get(config["learning_rate"], self._loss(model))
        metrics = {"train/loss": 0.9}
        if eval_texts:
            metrics["eval/loss"] = loss
        return TrainOutput(metrics, [{"step": 1, "loss": 0.9}, {"step": 1, "eval_loss": loss}])

    def train_dpo(self, model, tokenizer, train_rows, eval_rows, config, *, resume=True, history_dir=None):
        self.calls.append(("train_dpo", {"model": model.name, "train": list(train_rows), "config": config}))
        return TrainOutput({"train/loss": 0.6, "eval/loss": 0.5} if eval_rows else {"train/loss": 0.6}, [])

    def train_grpo(self, model, tokenizer, rows, reward_funcs, config, *, resume=True, history_dir=None):
        # Call the reward functions the way TRL does: keyword arguments, one completion list per prompt.
        completions = [[{"role": "assistant", "content": r["answer"] if i % 2 == 0 else "wrong"}] for i, r in enumerate(rows)]
        rewards = {
            f.__name__: f(prompts=[r["prompt"] for r in rows], completions=completions,
                          completion_ids=None, answer=[r["answer"] for r in rows], trainer_state=None)
            for f in reward_funcs
        }
        self.calls.append(("train_grpo", {"model": model.name, "rows": list(rows), "config": config, "rewards": rewards}))
        log = [{"step": i + 1, "reward": 0.2 + 0.06 * i, "rewards/correct/mean": 0.2 + 0.06 * i, "loss": 0.1} for i in range(10)]
        return TrainOutput({"train/loss": 0.1}, log)

    def evaluate_loss(self, model, tokenizer, texts, config, *, responses_only=None):
        self.calls.append(("evaluate_loss", {"model": model.name, "texts": list(texts), "responses_only": responses_only}))
        return {"eval/loss": self._loss(model), "eval/runtime": 1.0}

    @staticmethod
    def truth(messages):
        assert messages[-1]["role"] == "user"  # the reference answer is never shown to the model
        asked = messages[-1]["content"]
        return asked.replace("Question", "Answer").rstrip("?") if asked.startswith("Question") else asked.replace("Do", "Done")

    def generate(self, model, tokenizer, conversations, *, max_new_tokens=256, temperature=0.0, top_p=1.0,
                 template_kwargs=None):
        self.calls.append(("generate", len(conversations)))
        self.prompts = [list(c) for c in conversations]
        for key, right in self.skill.items():
            if key in model.name:  # a model of known skill: right on the first `right` of every 10 questions
                return [self.truth(m) if i % 10 < right else "no idea" for i, m in enumerate(conversations)]
        # the first answer is exactly right, the second only contains it, the rest are wrong
        return [
            [f"<think>hmm</think> {self.truth(m)}", f"Sure: {self.truth(m).lower()}!", "no idea"][min(i, 2)]
            for i, m in enumerate(conversations)
        ]

    def sample(self, model, tokenizer, conversations, *, n=1, max_new_tokens=256, temperature=0.0, top_p=1.0,
               template_kwargs=None):
        self.calls.append(("sample", len(conversations), n, temperature))
        out = []
        for i, messages in enumerate(conversations):
            good = f"<think>step by step</think> {self.truth(messages)}"
            # of every four prompts: always right, sometimes right, sometimes right without reasoning, never right
            group = [[good] * n, [good, "wrong a", "wrong b"] * n, [self.truth(messages), "wrong"] * n, ["wrong"] * n][i % 4]
            out.append(group[:n])
        return out

    def save(self, model, tokenizer, out_dir):
        self.calls.append(("save", str(out_dir)))
        name = "adapter_config.json" if model.adapter else "config.json"
        saved = (model.lora or {}).get("modules_to_save")
        Path(out_dir, name).write_text(json.dumps({"base_model_name_or_path": model.name, "modules_to_save": saved}))

    def save_merged(self, model, tokenizer, out_dir, method="merged_16bit"):
        self.calls.append(("save_merged", str(out_dir), method))
        Path(out_dir, "config.json").write_text(json.dumps({"merged_from": model.name, "method": method}))

    def save_gguf(self, model, tokenizer, out_dir, quantization):
        self.calls.append(("save_gguf", str(out_dir), list(quantization)))
        directory = Path(f"{out_dir}_gguf")
        directory.mkdir(parents=True, exist_ok=True)
        files = [str(directory / f"model.{q}.gguf") for q in quantization]
        for f in files:
            Path(f).write_bytes(b"gguf")
        return str(directory), files

    def push(self, model, tokenizer, repo_id, method, *, private=True, quantization=None):
        self.calls.append(("push", repo_id, method, private))

    def of(self, name):
        return [c for c in self.calls if c[0] == name]


def partial_credit(output, reference, row):
    """A custom scorer for the tests: 5 (which a scorer must clamp to 1) for the reference, 0.5 for
    an answer that only mentions it, -3 (clamped to 0) otherwise."""
    if output.strip() == reference:
        return 5
    return 0.5 if reference in output else -3


def half_reward(completion, reference, prompt):
    return 0.5
