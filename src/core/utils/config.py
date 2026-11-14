"""Flat YAML defaults for argparse entry points; explicit command flags take precedence."""
import argparse
import json
from pathlib import Path


def parse_yaml_args(parser, argv=None, *, json_fields=()):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--config_path', type=Path)
    path = preliminary.parse_known_args(argv)[0].config_path
    if path is not None:
        import yaml

        defaults = yaml.safe_load(path.read_text()) or {}
        if not isinstance(defaults, dict):
            parser.error(f'{path}: configuration must be a mapping')
        actions = {action.dest: action for action in parser._actions if action.dest != 'help'}
        unknown = set(defaults) - set(actions)
        if unknown:
            parser.error(f'{path}: unknown options {sorted(map(str, unknown))}')
        for key, value in defaults.items():
            action = actions[key]
            multiple = isinstance(action, argparse._AppendAction) or action.nargs in ('*', '+')
            if multiple and value is not None and not isinstance(value, list):
                parser.error(f'{path}: {key} must be a list')
            if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction,
                                   argparse.BooleanOptionalAction)):
                if not isinstance(value, bool):
                    parser.error(f'{path}: {key} must be a boolean')
            elif key in json_fields and isinstance(value, dict):
                defaults[key] = json.dumps(value)
            elif value is not None and action.type is not None:
                try:
                    defaults[key] = ([action.type(str(item)) for item in value] if multiple
                                     else action.type(str(value)))
                except (ValueError, TypeError) as exc:
                    parser.error(f'{path}: invalid {key}: {exc}')
            choices = defaults[key] if multiple and value is not None else [defaults[key]]
            if action.choices is not None and any(item not in action.choices for item in choices):
                parser.error(f'{path}: {key} must be one of {list(action.choices)}')
            if value is not None:
                action.required = False
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)
