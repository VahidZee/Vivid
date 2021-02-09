import inspect
import pprint
import typing as th

from vivid.utilities.variables.exceptions import VariableLookupException


def get_value(name: str, context: th.Any, strict: bool = True):
    var = context
    for split in name.split('.'):
        if isinstance(context, dict):
            if split not in var:
                if strict:
                    raise KeyError('Invalid key "%s"' % name)
                else:
                    return None
            var = var[split]
        else:
            if not hasattr(var, split):
                if strict:
                    raise AttributeError('Invalid attribute %s' % name)
                else:
                    return None
            var = getattr(var, split)
    return var


def local_context(context_level: int = 1, verbose: bool = False):
    # finding the frame of interest
    interest_frame = inspect.currentframe()
    for i in range(context_level):
        interest_frame = interest_frame.f_back
        if verbose:
            print(f'context-level {i}:')
            pprint.pprint(interest_frame.f_locals)
            print()
    return interest_frame.f_locals


def get_context(context: th.Union[None, str, th.Any], local_context_dict: dict):
    if context is None:
        return local_context_dict
    if isinstance(context, str):
        return get_value(context, local_context_dict, strict=True)
    return context


def evaluate_in_context(name: str, context: th.Any, defaults: dict = None, strict=True):
    defaults = defaults or dict()
    # looking to see if any default value is provided
    default_found = False
    try:
        default = get_value(name, defaults, strict=True)
        default_found = True
    except (KeyError, AttributeError):
        pass

    # getting value
    try:
        return get_value(name, context, strict=strict)
    except Exception:
        if default_found:
            return default
        raise VariableLookupException(f'No value was found for "{name}"')


def exists_in_context(name: str, context: th.Any, defaults: dict = None, **kwargs):
    defaults = defaults or dict()
    # looking to see if any default value is provided
    try:
        _ = get_value(name, defaults, strict=True)
        return True
    except (KeyError, AttributeError):
        pass
    try:
        _ = get_value(name, context, strict=True)
        return True
    except Exception:
        raise VariableLookupException(f'No value was found for "{name}"')
