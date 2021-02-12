from collections import OrderedDict
import inspect
import typing as th


def args_dict(prefix: str, args_dict: dict, remove: bool = False, short_hand: bool = False):
    result = {f'{key.replace(f"{prefix}_", "")}': value for key, value in args_dict.items() if
              key.startswith(f"{prefix}_")}
    if short_hand is True:
        result = {**args_dict.get(prefix, dict()), **result}
    elif short_hand is not False:
        assert isinstance(short_hand, str), 'unsupported shorthand specified'
        if prefix in args_dict:
            result[short_hand] = args_dict[prefix]
    if not remove:
        return result
    to_remove = [key for key in args_dict.keys() if key.startswith(f"{prefix}_" if short_hand is False else prefix)]
    for item in to_remove:
        del args_dict[item]
    return result


def squeeze_dict(input_dict: dict, base_dict: th.Optional[str] = None, squeeze: str = '_'):
    result = input_dict.get(base_dict, dict())
    for name, value in input_dict.items():
        if  name == base_dict:
            continue
        for sub_name, sub_value in value.items():
            result[f'{name}{squeeze}{sub_name}'] = sub_value
    return result


def unsqueeze_dict(input_dict: dict, **kwargs):
    result = OrderedDict()
    for name, value in input_dict.items():
        for suffix, context in kwargs.items():
            if name.endswith(f'_{suffix}'):
                context[name.replace(f'_{suffix}', '')] = value
                break
        else:
            result[name] = value
    return result


def function_parameters(f: th.Callable):
    sig = inspect.signature(f)
    params = OrderedDict()
    for name, par in sig.parameters.items():
        param_dict = dict()
        if par.kind == inspect.Parameter.VAR_KEYWORD:
            param_dict['kind'] = 'VAR_KEYWORD'
        elif par.kind == inspect.Parameter.VAR_POSITIONAL:
            param_dict['kind'] = 'VAR_POSITIONAL'
        else:
            param_dict['kind'] = 'NORMAL'
        if par.default != inspect.Parameter.empty:
            param_dict['default'] = par.default
        params[name] = param_dict
    return params
