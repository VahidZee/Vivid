from .variable import Var, VariableLookupException
import typing as th


def var_args_description(
        var: Var, block_name: th.Optional[str] = None, arg_name: th.Optional[str] = None,
        defaults: th.Optional[dict] = None, kwargs: th.Optional[dict] = None, context_level: int = 1):
    # for evaluation context
    defaults = defaults or dict()
    kwargs = kwargs or dict()
    names, contexts = (var._names, var._contexts) if var.priority_lookup else (
        [var.name], [var.context])
    description = dict(kind='VAR', variable=var, active=var.active, lookup=[])

    for i, (var_name, context) in enumerate(zip(names, contexts)):
        if context == 'kwargs' and (var_name is None or '.' not in var_name):
            description['lookup'].append(var_name or arg_name)
    try:
        if var.is_active(prefix=block_name, name=block_name, context_level=context_level):
            description['default'] = var._value_decorators[0](
                name=arg_name, prefix=block_name, context_level=context_level)
    except VariableLookupException:
        pass
    return description
