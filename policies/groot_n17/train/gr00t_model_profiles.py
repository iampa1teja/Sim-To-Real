"""Pinned GR00T profiles and read-only validation of the official SO-arm model."""
import json
import os
from pathlib import Path
import re
import struct

N16_PIN = 'ead52833afbbf4243f8cd5e7664f48a94de03b19'
N17_PIN = '51d4c89f72fda44cbf77285c6a8114b52676b8a1'
SO_ARM_REPO = 'nvidia/SO_ARM_Starter_Gr00tN17'
SO_ARM_REVISION = '93a5a88f78a395939f784a5fe3685184913dcc60'
COSMOS_REPO = 'nvidia/Cosmos-Reason2-2B'
PROFILES = {
    'so100_n16': {'revision': N16_PIN, 'cameras': ['front', 'wrist'],
                  'video_backend': 'ffmpeg', 'relative_actions': True},
    'so_arm_n17': {'revision': N17_PIN, 'cameras': ['room', 'wrist'],
                   'video_backend': 'torchcodec', 'relative_actions': False},
}


def download_command(destination):
    """Download root inference weights, excluding saved optimizer/training state."""
    import shlex
    patterns = ('config.json', 'processor_config.json', 'embodiment_id.json',
                'statistics.json', 'model-*.safetensors', 'model.safetensors.index.json')
    includes = ' '.join(f'--include {shlex.quote(pattern)}' for pattern in patterns)
    return (f'hf download {SO_ARM_REPO} --revision {SO_ARM_REVISION} '
            f'{includes} --local-dir {shlex.quote(str(destination))}')


def _inspect_safetensors(path, indexed_keys=()):
    """Inspect a shard's header and payload length without materializing tensors."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    with path.open('rb') as stream:
        length_bytes = stream.read(8)
        require(len(length_bytes) == 8, f'Incomplete weights: {path.name}')
        header_length, = struct.unpack('<Q', length_bytes)
        require(0 < header_length <= 16 * 1024 * 1024, f'Invalid weights header: {path.name}')
        encoded = stream.read(header_length)
        require(len(encoded) == header_length, f'Incomplete weights header: {path.name}')
        header = json.loads(encoded)
    require(isinstance(header, dict), f'Invalid weights header: {path.name}')
    tensors = {key: value for key, value in header.items() if key != '__metadata__'}
    require(bool(tensors), f'Empty weights: {path.name}')
    require(all(key in tensors for key in indexed_keys), f'Indexed tensors missing from {path.name}')
    for tensor in tensors.values():
        offsets = tensor.get('data_offsets') if isinstance(tensor, dict) else None
        require(isinstance(offsets, list) and len(offsets) == 2
                and all(type(value) is int for value in offsets)
                and 0 <= offsets[0] <= offsets[1], f'Invalid weights offsets: {path.name}')
    payload_length = max(tensor['data_offsets'][1] for tensor in tensors.values())
    require(payload_length > 0 and path.stat().st_size == 8 + header_length + payload_length,
            f'Incomplete weights: {path.name}')


def transformers_hub_cache(environ=None):
    """Match the default cache used by the pinned Transformers model loaders."""
    env = os.environ if environ is None else environ
    def expand_hub_path(value):
        # Hub modern constants expand ~ and variables, in this order.
        value = os.path.expanduser(str(value))
        def replace(match):
            name = match.group(1)
            if name.startswith('{'):
                name = name[1:-1]
            return env.get(name, match.group(0))
        return re.sub(r'\$(\w+|\{[^}]*\})', replace, value)
    default_home = str(Path.home() / '.cache')
    hf_home = expand_hub_path(env.get('HF_HOME', os.path.join(
        env.get('XDG_CACHE_HOME', default_home), 'huggingface')))
    hub_cache = expand_hub_path(env.get('HF_HUB_CACHE', env.get(
        'HUGGINGFACE_HUB_CACHE', os.path.join(hf_home, 'hub'))))
    # Transformers 4.x uses os.getenv, preserving explicitly empty legacy
    # values and leaving legacy ~/$VAR strings literal. Resolve at startup
    # so the caller can export one absolute path before changing directory.
    legacy = env.get('PYTORCH_PRETRAINED_BERT_CACHE', hub_cache)
    legacy = env.get('PYTORCH_TRANSFORMERS_CACHE', legacy)
    legacy = env.get('TRANSFORMERS_CACHE', legacy)
    return Path(legacy).resolve()


def inspect_cosmos_cache(cache_root=None, *, environ=None):
    """Validate the offline default-main Qwen3-VL backbone and processor cache.

    Current official Cosmos files include a single model.safetensors, image and
    video preprocessor JSONs, and a chat template. Indexed weights are accepted
    too. This performs no Hub calls and never modifies the cache.
    """
    cache = (Path(cache_root).expanduser().resolve() if cache_root is not None
             else transformers_hub_cache(environ))
    repository = cache / 'models--nvidia--Cosmos-Reason2-2B'
    revision = (repository / 'refs/main').read_text().strip()
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Invalid Cosmos cache main reference')
    snapshot = repository / 'snapshots' / revision
    if not snapshot.is_dir() or not snapshot.resolve().is_relative_to(repository.resolve()):
        raise ValueError(f'Missing or invalid Cosmos cache snapshot: {snapshot}')

    def file(name):
        path = snapshot / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f'Missing or empty Cosmos cache file: {name}')
        # Normal HF snapshot symlinks point into this repository's blobs.
        if not path.resolve().is_relative_to(repository.resolve()):
            raise ValueError(f'Cosmos cache file leaves repository: {name}')
        return path

    required = ('config.json', 'tokenizer.json', 'tokenizer_config.json',
                'preprocessor_config.json', 'video_preprocessor_config.json')
    configs = {}
    for name in required:
        configs[name] = json.loads(file(name).read_text())
        if not isinstance(configs[name], dict) or not configs[name]:
            raise ValueError(f'Invalid Cosmos cache JSON: {name}')
    # ProcessorMixin loads this independently of tokenizer_config.json.
    # The legacy JSON file takes precedence over a raw Jinja file.
    if (snapshot / 'chat_template.json').is_file():
        document = json.loads(file('chat_template.json').read_text())
        template = document.get('chat_template') if isinstance(document, dict) else None
        if not isinstance(template, str) or not template.strip():
            raise ValueError('Cosmos chat_template.json needs a nonempty chat_template')
    elif (snapshot / 'chat_template.jinja').is_file():
        if not file('chat_template.jinja').read_text().strip():
            raise ValueError('Empty Cosmos processor chat template')
    else:
        raise ValueError('Missing Cosmos processor chat template')

    if (snapshot / 'model.safetensors').is_file():
        shards = ['model.safetensors']
        _inspect_safetensors(file(shards[0]))
    else:
        index = json.loads(file('model.safetensors.index.json').read_text())
        mapping = index.get('weight_map') if isinstance(index, dict) else None
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError('Empty Cosmos safetensors weight index')
        if any(not isinstance(name, str) for name in mapping.values()):
            raise ValueError('Invalid Cosmos shard filename')
        shards = sorted(set(mapping.values()))
        for name in shards:
            if not isinstance(name, str) or Path(name).name != name or not name.endswith('.safetensors'):
                raise ValueError('Invalid Cosmos shard filename')
            _inspect_safetensors(file(name), [key for key, shard in mapping.items() if shard == name])
    return {'repo_id': COSMOS_REPO, 'cache_root': str(cache), 'revision': revision,
            'snapshot': str(snapshot), 'weight_shards': shards, 'offline_cache_checked': True}


def inspect_so_arm_checkpoint(root, *, check_weights=True):
    """Fail before launching if the saved embodiment or weight layout differs."""
    root = Path(root)
    def read(name):
        return json.loads((root / name).read_text())
    config = read('config.json')
    processor = read('processor_config.json')
    ids = read('embodiment_id.json')
    statistics = read('statistics.json')
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(config.get('model_type') == 'Gr00tN1d7', 'Expected GR00T N1.7 model_type Gr00tN1d7')
    require(processor.get('processor_class') == 'Gr00tN1d7Processor', 'Expected N1.7 processor')
    kwargs = processor['processor_kwargs']
    modalities = kwargs['modality_configs']['new_embodiment']
    require(ids.get('new_embodiment') == 10, 'NEW_EMBODIMENT must have ID 10')
    for kind, keys in [('video', ['room', 'wrist']), ('state', ['single_arm', 'gripper']),
                       ('action', ['single_arm', 'gripper']),
                       ('language', ['annotation.human.task_description'])]:
        require(modalities[kind]['modality_keys'] == keys, f'Unexpected {kind} keys')
        expected = list(range(16)) if kind == 'action' else [0]
        require(modalities[kind]['delta_indices'] == expected, f'Unexpected {kind} horizon')
    actions = modalities['action']['action_configs']
    require(len(actions) == 2 and all(
        a.get('rep') == 'ABSOLUTE' and a.get('type') == 'NON_EEF'
        and a.get('format') == 'DEFAULT' and a.get('state_key') is None for a in actions),
        'SO-arm actions must be absolute joint targets for both arm and gripper')
    for kind in ['state', 'action']:
        for key, size in [('single_arm', 5), ('gripper', 1)]:
            stats = statistics['new_embodiment'][kind][key]
            require(all(len(stats[name]) == size for name in ('min', 'max', 'mean', 'std', 'q01', 'q99')),
                    f'Unexpected {kind}.{key} dimension')
    require(config.get('action_horizon') == 40 and kwargs.get('max_action_horizon') == 40,
            'Expected model capacity 40 (SO-arm embodiment predicts 16)')
    index = read('model.safetensors.index.json')
    mapping = index['weight_map']
    require(bool(mapping), 'Empty safetensors weight index')
    shards = sorted(set(mapping.values()))
    for name in shards:
        require(Path(name).name == name and name.endswith('.safetensors'), 'Invalid shard filename')
        if check_weights:
            _inspect_safetensors(root / name, [key for key, shard in mapping.items() if shard == name])
    return {'model_version': 'N1.7', 'embodiment_tag': 'NEW_EMBODIMENT', 'embodiment_id': 10,
            'video_keys': ['room', 'wrist'], 'state_dimensions': {'single_arm': 5, 'gripper': 1},
            'action_representation': {'single_arm': 'ABSOLUTE', 'gripper': 'ABSOLUTE'},
            'embodiment_action_horizon': 16, 'model_action_capacity': 40,
            'layout': 'checkpoint files at repository root', 'weight_shards': shards,
            'weights_checked': check_weights}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--checkpoint', type=Path)
    mode.add_argument('--cosmos-cache', action='store_true')
    mode.add_argument('--cache-path', action='store_true')
    parser.add_argument('--hf-hub-cache', type=Path)
    args = parser.parse_args()
    if args.cache_path:
        print(transformers_hub_cache())
        parser.exit()
    try:
        result = (inspect_cosmos_cache(args.hf_hub_cache) if args.cosmos_cache
                  else inspect_so_arm_checkpoint(args.checkpoint))
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        if args.cosmos_cache:
            parser.exit(1, f'Cosmos cache validation failed: {error}\n'
                        f'Accept access at https://huggingface.co/{COSMOS_REPO}, then cache it explicitly:\n'
                        f'hf auth login\nhf download {COSMOS_REPO}\n')
        parser.exit(1, f'Checkpoint validation failed: {error}\n')
