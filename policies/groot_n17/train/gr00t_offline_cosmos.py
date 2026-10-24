"""Load the pinned Cosmos processor from its cached offline snapshot.

Transformers 4.57.3's tokenizer checks remote Mistral metadata for large
tokenizers even with local_files_only=True. A local Cosmos snapshot avoids
that unrelated request while preserving the official processor and weights.
"""

from contextlib import contextmanager
import importlib
from pathlib import Path

from gr00t_model_profiles import COSMOS_REPO


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
