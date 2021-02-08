import pprint
import typing as th
import functools
import inspect


class VariableLookupException(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super(VariableLookupException, self).__init__(message)


class Var:
    log_lookup_frames = False
    log_lookup = False

    def __init__(
            self,
            name: th.Optional[str] = None,
            context: th.Optional[th.Union[str, th.Any]] = None,
            active: th.Union[bool, th.Any] = True,
            lookup: th.Optional[th.Callable[[object, dict], th.Any]] = None,
            decorator: th.Optional[th.Callable[[th.Any], th.Any]] = None,
            **kwargs: th.Optional[th.OrderedDict[str, th.Union[th.Any, th.Dict[str, th.Any]]]]  # [default], decorators
    ):
        self.name = name
        self.context = context
        self.default_set = False

        self.decorator = decorator
        self.active = active
        self.default = None

        if 'default' in kwargs:
            self.default = kwargs.pop('default')
            self.default_set = True

        decorators = kwargs
        num_decorators = len(decorators) + (1 if decorator else 0)
        self.value_decorators = [self.value]
        for i, (name, value) in enumerate(decorators.items()):
            dec = self.initialize_var_decorator(name, value) if hasattr(Var, name) else value
            self.value_decorators.append(self._decorate_value(self.value_decorators[-1], dec))

        if decorator is not None:
            self.value_decorators.append(self._decorate_value(self.value_decorators[-1], decorator))

        if num_decorators:
            self.value = functools.wraps(self.value_decorators[0])(
                lambda *args, context_level=1, **kwargs: self.value_decorators[-1](
                    *args, context_level=context_level + num_decorators + 1, **kwargs)
            )

    @staticmethod
    def _decorate_value(function, decorator):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return decorator(function(*args, **kwargs))

        return wrapper

    def is_active(self, prefix=None, context_level=1):
        if isinstance(self.active, Var):
            return self.active.value(prefix=prefix, name='active', context_level=context_level + 1)
        return self.active

    @staticmethod
    def initialize_var_decorator(name, arg):
        if isinstance(arg, dict):
            return getattr(Var, name)(**arg)
        return getattr(Var, name)(arg)

    @staticmethod
    def tee(msg='', *args, **kwargs):
        """
        Decorator to log the resulting value
        """

        def log(value):
            print(f'{msg or "tee"} - {value}')
            return value

        return log

    @staticmethod
    def dig_value(name, context, strict=True):
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

    @staticmethod
    def dig(context, strict=True):
        """decorator util"""
        return functools.partial(Var.dig_value, context=context, strict=strict)

    def value(self, name=None, prefix=None, defaults=None, context_level=1, strict=False):
        assert (name is not None and self.name is None) or self.name is not None, 'no name is provided for the variable'
        if not strict:
            name = self.name or name

        variable_name = f'{prefix}_{name}' if prefix is not None else name
        if prefix is not None:
            try:
                return self.value_decorators[0](name=variable_name, context_level=context_level + 1, strict=True)
            except VariableLookupException:
                variable_name = name
        if self.log_lookup or self.log_lookup_frames:
            print('looking-up variable | name:', name, 'variable_name:', variable_name)
        # finding the frame of interest
        interest_frame = inspect.currentframe()
        for i in range(context_level):
            interest_frame = interest_frame.f_back
            if self.log_lookup_frames:
                print(f'context-level {i}:')
                pprint.pprint(interest_frame.f_locals)
                print()
        values = interest_frame.f_locals
        defaults = values.get('defaults', defaults if defaults is not None else dict())

        # looking to see if any default value is provided
        default = None
        default_found = True
        try:
            default = self.dig_value(variable_name, defaults, strict=True)
        except (KeyError, AttributeError):
            if self.default_set and not strict:
                default = self.default
            else:
                default_found = False

        # getting value
        try:
            if self.context is None:
                return self.dig_value(variable_name, values)
            context = self.dig_value(self.context, values) if \
                isinstance(self.context, str) else values if self.context is None else self.context
            return self.dig_value(variable_name, context)
        except Exception:
            if default_found:
                return default
            raise VariableLookupException(
                f'No value was found for "{variable_name}"' + (
                    f'in context {self.context}' if self.context is not None else ''))

    def __repr__(self):
        args = []
        if self.name is not None:
            args.append(f'name="{self.name}"')
        if self.context is not None:
            args.append(f'context="{self.context}"')
        if self.default:
            args.append(f'default={self.default}')
        if self.active is not True:
            args.append(f'active={self.active}')
        return f'Var({", ".join(args)})'


KWVar = functools.partial(Var, context='kwargs')
