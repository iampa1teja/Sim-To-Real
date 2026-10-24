"""Load the pinned Cosmos processor from its cached offline snapshot.

Transformers 4.57.3's tokenizer checks remote Mistral metadata for large tokenizers even with
local_files_only=True. A local Cosmos snapshot avoids that unrelated request while preserving the
official processor and weights. Copied from the training-side gr00t_offline_cosmos.py in train/ so the generic
policy server image does not need the training scripts.
"""
from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import re

COSMOS_REPO = "nvidia/Cosmos-Reason2-2B"


def transformers_hub_cache(environ=None):
    """Match the default cache used by the pinned Transformers model loaders."""
    env = os.environ if environ is None else environ

    def expand_hub_path(value):
        value = os.path.expanduser(str(value))

        def replace(match):
            name = match.group(1)
            if name.startswith("{"):
                name = name[1:-1]
            return env.get(name, match.group(0))

        return re.sub(r"\$(\w+|\{[^}]*\})", replace, value)

    default_home = str(Path.home() / ".cache")
    hf_home = expand_hub_path(env.get("HF_HOME", os.path.join(env.get("XDG_CACHE_HOME", default_home), "huggingface")))
    hub_cache = expand_hub_path(env.get("HF_HUB_CACHE", env.get("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))))
    legacy = env.get("PYTORCH_PRETRAINED_BERT_CACHE", hub_cache)
    legacy = env.get("PYTORCH_TRANSFORMERS_CACHE", legacy)
    legacy = env.get("TRANSFORMERS_CACHE", legacy)
    return Path(legacy).resolve()


@contextmanager
def offline_cosmos_processor():
    """Temporarily change only the Cosmos processor builder's offline path."""
    module = importlib.import_module("gr00t.model.gr00t_n1d7.processing_gr00t_n1d7")
    original = module.build_processor

    def build_processor(model_name, transformers_loading_kwargs):
        from transformers.utils import cached_file, is_offline_mode

        if model_name == COSMOS_REPO and is_offline_mode():
            lookup = dict(transformers_loading_kwargs, local_files_only=True)
            config = cached_file(model_name, "config.json", **lookup)
            return original(str(Path(config).parent), transformers_loading_kwargs)
        return original(model_name, transformers_loading_kwargs)

    module.build_processor = build_processor
    try:
        yield
    finally:
        module.build_processor = original
