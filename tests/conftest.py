import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # tests build tiny random models; nothing is downloaded
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")
