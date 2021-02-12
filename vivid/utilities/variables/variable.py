import typing as th
import functools
from .exceptions import VariableLookupException
import vivid.utilities.variables.lookup as lookup

CONTEXT_TYPE = th.Union[str, th.Any]
LOOP_UP_TYPE = th.Callable[[object, dict], th.Any]
DECORATOR_TYPE = th.Callable[[th.Any], th.Any]

try:
    # Literal is available in python > 3.8
    LOOKUP_OPTIONS = th.Literal["evaluate", "exists"]
except AttributeError:
    LOOKUP_OPTIONS = str


class Var:
    log_lookup_frames = False
    log_lookup = False

    def __init__(
            self,
            name: th.Optional[th.Union[str, th.List[str]]] = None,
            context: th.Optional[th.Union[th.List[CONTEXT_TYPE], CONTEXT_TYPE]] = None,
            active: th.Union[bool, th.Any] = True,
            lookup_function: th.Union[LOOKUP_OPTIONS, th.Callable] = 'evaluate',
            decorator: th.Optional[DECORATOR_TYPE] = None,
            **kwargs: th.Optional[th.OrderedDict[str, th.Union[th.Any, th.Dict[str, th.Any]]]]  # [default], decorators
    ):
        """

        :param name: name(s) of the variable to be looked up (in order of priority)
        :param context: context in which the variable should be looked up
        :param is_active: whether the variable should be looked up
        :param lookup_function: the function to use for variable look up in the context
        :param default: if mentioned will be returned if no value is found for the variable
        :param decorator: if mentioned will be the final decorator function for the evaluated value
        :param kwargs: decorators to apply to the evaluated value in call order
        """
        self.name = name
        self.context = context
        self.default_set = False

        self.decorator = decorator
        self.active = active
        self.default = None

        # default value setup
        if 'default' in kwargs:
            self.default = kwargs.pop('default')
            self.default_set = True

        # lookup function setup
        self.__lookup_value = lookup.evaluate_in_context
        if isinstance(lookup_function, str):
            self.__lookup_value = lookup.get_value(f'{lookup_function}_in_context', lookup)
            if lookup_function == 'exists':
                self.default = self.default if self.default_set else False
                self.default_set = True
        elif callable(lookup_function):
            self.__lookup_value = lookup.get_value(f'{lookup_function}_in_context', lookup)

        # priority lookup setup
        self.priority_lookup = False
        if isinstance(self.name, (list, tuple)) or isinstance(self.context, (list, tuple)):
            count_names = len(self.name) if isinstance(self.name, (list, tuple)) else 1
            count_contexts = len(self.context) if isinstance(self.context, (list, tuple)) else 1
            choices_count = max(count_names, count_contexts)
            if not choices_count == 1:
                self._names = self.name if count_names == choices_count else [self.name] * choices_count
                self._contexts = self.context if count_contexts == choices_count else [self.context] * choices_count

                assert count_names == count_contexts or count_names == 1 or count_contexts == 1, \
                    'inconsistent number of values are provided'
                self.priority_lookup = True
            else:
                self.name = self.name[0] if isinstance(self.name, (tuple, list)) else self.name
                self.context = self.context[0] if isinstance(self.context, (tuple, list)) else self.context

        # decorators setup
        decorators = kwargs
        num_decorators = len(decorators) + (1 if decorator else 0)
        self._value_decorators = [self.value]
        for i, (name, value) in enumerate(decorators.items()):
            dec = self.__initialize_var_decorator(name, value) if hasattr(Var, name) else value
            self._value_decorators.append(self.__decorate_value(self._value_decorators[-1], dec))

        if decorator is not None:
            self._value_decorators.append(self.__decorate_value(self._value_decorators[-1], decorator))

        if num_decorators:
            self.value = functools.wraps(self._value_decorators[0])(
                lambda *args, context_level=1, **kwargs: self._value_decorators[-1](
                    *args, context_level=context_level + num_decorators + 1, **kwargs)
            )

    def is_active(self, prefix: th.Optional[str] = None, name: th.Optional[str] = None, context_level=1):
        """indicates whether this variable should be looked up!"""
        if Var.log_lookup:
            print(f'check is active: {self}')
        if isinstance(self.active, Var):
            result = self.active.value(prefix=prefix, name='active', context_level=context_level + 1)
        elif isinstance(self.active, str) or callable(self.active):
            names = self._names if self.priority_lookup else [self.name]
            names = [s or name for s in names]
            result = Var(name=names, context=self.context,
                         lookup_function=self.active).value(prefix=prefix, context_level=context_level + 1)
        else:
            result = self.active
        if Var.log_lookup:
            print(f'\t active: {result}')
        return result

    # decoration
    @staticmethod
    def __decorate_value(function, decorator):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return decorator(function(*args, **kwargs))

        return wrapper

    @staticmethod
    def __initialize_var_decorator(name, arg):
        if isinstance(arg, dict):
            return getattr(Var, name)(**arg)
        return getattr(Var, name)(arg)

    # decorators
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
    def dig(context, strict=True):
        """decorator util"""
        return functools.partial(lookup.get_value, context=context, strict=strict)

    def value(self, name=None, prefix=None, defaults=None, context_level=1, strict=False):
        names = self._names if self.priority_lookup else [self.name if not strict else name]
        names = [s or name for s in names]
        local_context = lookup.local_context(context_level=context_level + 1, verbose=Var.log_lookup_frames)
        contexts = self._contexts if self.priority_lookup else [self.context]

        if self.log_lookup or self.log_lookup_frames:
            print(f'lookup: {names} in context: {contexts}')

        if prefix is not None:
            names = [f'{prefix}_{i}' if i is not None else prefix for i in names] + names
            contexts = contexts + contexts
        base_defaults = defaults or dict()
        for name, context in zip(names, contexts):
            try:
                context = lookup.get_context(context=context, local_context_dict=local_context)
                defaults = lookup.evaluate_in_context(
                    name='defaults', context=local_context, strict=False) or base_defaults
                result = self.__lookup_value(name=name, context=context, defaults=defaults)
                if Var.log_lookup:
                    print(f'\t value: {result}')
                return result
            except (VariableLookupException, KeyError, AttributeError):
                pass

        if self.default_set and not strict:
            if Var.log_lookup:
                print(f'\t value-default: {self.default}')
            return self.default
        raise VariableLookupException(f'no value was found for {names}')

    def __repr__(self):
        args = []
        if self.name is not None:
            args.append(f'name="{self.name}"')
        if self.context is not None:
            args.append(f'context="{self.context}"')
        if self.default_set:
            args.append(f'default={self.default}')
        if self.active is not True:
            args.append(f'active={self.active}')
        return f'Var({", ".join(args)})'


KWVar = functools.partial(Var, context='kwargs')
