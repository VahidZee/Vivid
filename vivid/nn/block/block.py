import torch
import typing as th
import inspect
from vivid.utilities import parse
from .repr import _BlockRepr
from vivid.utilities.variables import Var, var_args_description

from collections import OrderedDict, defaultdict


# from .instance import Block


class _Block(metaclass=_BlockRepr):
    initialized = False

    def __init__(
            self,
            repeat=None,  # number (count), {connection}, weight_tied (number/bool)
            parallel=None,
            connection=None,  # str (residual/dense), connection_link ([str]*, bool), connection_operation (callable)
            defaults=None,  # dict
            init=None,
            **kwargs
    ):
        """
        instantiate the Block template
        """
        # initializing base classes
        torch.nn.Module.__init__(self)

        # translating variable names
        temp_kwargs = dict()
        for key, value in kwargs.items():
            if key in self._translation_table:
                temp_kwargs[f'{self._translation_table[key]["block"][0]}_{key}'] = value
            else:
                temp_kwargs[key] = value
        kwargs = temp_kwargs

        self.initialized = True
        self.block_names = []
        self.connection = connection

        # initializing variables
        defaults = defaults or dict()
        defaults = {**defaults, **self._defaults}

        # connection details
        connection = self.__get_connection_description(kwargs, connection)

        # initialization details todo

        init = self._init
        # repeat
        self.repeat = self.__get_repeat_description(kwargs=kwargs, repeat=repeat)

        # instantiating blocks
        if self.repeat['count']:
            self.__instantiate_repeat(kwargs=kwargs, defaults=defaults, init=init)
            return

        blocks = self._block
        if not isinstance(blocks, dict):
            blocks = dict(block=blocks)
            return

        for name, item in blocks.items():
            # skipping non-active blocks
            if (inspect.isclass(item) and issubclass(item, _Block)) or isinstance(item, Var):
                if not item.is_active(prefix=name):
                    continue

            self.block_names.append(name)

            # processing args
            block_args = self._block_args_table.get(name, dict())
            related_kwargs = parse.args_dict(name, kwargs)
            args = OrderedDict()
            for arg_name, arg_description in block_args.get('args', dict()).items():
                if arg_description['kind'] == 'VAR' and arg_description['variable'].is_active(
                        prefix=name, name=arg_name):
                    args[arg_name] = arg_description['variable'].value(name=arg_name, prefix=name)
                else:
                    args[arg_name] = related_kwargs[arg_name] if arg_name in related_kwargs else arg_description[
                        'default']
            if 'active' in related_kwargs:  # block activity (special argument) todo
                del related_kwargs['active']

            if isinstance(item, Var):
                item_cls = item.value(prefix=name, name=name)
                cls_names = [item.name] if not item.priority_lookup else item._names
                for n in [n or name for n in cls_names]:
                    if n in args and n in related_kwargs:
                        del related_kwargs[n]
            else:
                item_cls = item

            for arg_name, arg_value in related_kwargs.items():
                if arg_name not in args:
                    assert isinstance(
                        item, Var) or 'var_keyword' in block_args, f'unknown argument is provided for block: {name}'
                    args[arg_name] = arg_value
            previous_block = item_cls(**args)
            setattr(self, name, previous_block)

        # initializing weights
        init = None

    # properties
    @property
    def last_block(self):
        return getattr(self, self.block_names[-1])

    @classmethod
    def args_table(cls):
        args_table = defaultdict(list)
        defaults = cls._defaults
        kwargs = dict()
        for block_name, module_cls in cls._block.items():
            if inspect.isclass(module_cls) and issubclass(module_cls, _Block):
                block_args = module_cls.args_table()
            elif inspect.isclass(module_cls) and issubclass(module_cls, torch.nn.Module):
                block_args = parse.function_parameters(module_cls)
                for arg, value in cls._args.get(block_name, dict()).items():
                    if isinstance(value, Var):
                        block_args[arg] = var_args_description(
                            value, block_name=block_name, arg_name=arg, defaults=defaults, kwargs=kwargs)
                    else:
                        block_args[arg] = dict(kind='NORMAL', default=value)
                # setting defaults
                for arg, value in cls._defaults.get(block_name, dict()).items():
                    if block_args[arg]['kind'] != 'VAR':
                        block_args[arg]['default'] = value
            elif isinstance(module_cls, Var):
                block_args = dict()
                description = var_args_description(
                    module_cls, block_name=block_name, arg_name=block_name, defaults=defaults, kwargs=kwargs)
                description['VAR_CLS'] = True
                for name in description['lookup']:
                    block_args[name] = description
            else:
                block_args = dict()

            for arg, description in block_args.items():
                if isinstance(description, (list, tuple)):
                    for des in description:
                        des['block'] = [block_name] + des['block']
                        if des['kind'] == 'VAR':
                            kw_context = (des['variable'].context == 'kwargs') if not des[
                                'variable'].priority_lookup else ('kwargs' in des['variable']._contexts)
                            if kw_context and not des['lookup']:
                                continue
                        args_table[arg].append(des)
                    continue
                description['block'] = [block_name] if 'block' not in description else [block_name] + description[
                    'block']
                args_table[arg].append(description)

        for arg_name, value in cls._args.get('args', dict()).items():
            assert len(args_table[arg_name]) == 1, 'name overriding is provided for multiple choices'
            if isinstance(value, Var):
                description = var_args_description(
                    value, block_name=args_table[arg_name][0]['block'], arg_name=arg_name, defaults=defaults,
                    kwargs=kwargs)
                args_table[arg_name][0] = {**args_table[arg_name][0], **description}
        args_table['previous_block'].append(var_args_description(Var('previous_block', active='exists')))
        args_table['previous_block'][-1]['block'] = []
        return args_table

    @classmethod
    def block_args_table(cls):
        block_args_table = defaultdict(dict)
        args_table = cls._args_table
        for arg_name, args in args_table.items():
            for description in args:
                if not description.get('block', False):
                    continue
                block_name = description['block'][0]
                if description['kind'] == 'VAR_KEYWORD':
                    block_args_table[block_name]['var_keyword'] = dict(name=arg_name, args=[])
                elif description['kind'] == 'VAR_POSITIONAL':
                    block_args_table[block_name]['var_positional'] = dict(name=arg_name, args=[])
                elif 'VAR_CLS' in description:
                    if 'var_cls' not in block_args_table[block_name]:
                        block_args_table[block_name]['var_cls'] = OrderedDict()
                    block_args_table[block_name]['var_cls'][arg_name] = description
                else:
                    if 'args' not in block_args_table[block_name]:
                        block_args_table[block_name]['args'] = OrderedDict()
                    block_args_table[block_name]['args'][arg_name] = description
        return block_args_table

    @classmethod
    def translation_table(cls):
        table = defaultdict(list)
        block_args_table = cls._block_args_table
        for block_name, block_args in block_args_table.items():
            for arg_name, arg in block_args.get('args', dict()).items():
                if arg['kind'] == 'VAR':
                    for name in arg['lookup']:
                        table[name].append(arg)
                else:
                    table[arg_name].append(arg)
        table = {name: value[0] for name, value in table.items() if len(value) == 1}
        # setting up default values
        for name, value in cls._defaults.items():
            assert not isinstance(value, Var), 'default values cannot be variables'
            if name in table and table[name]['kind'] != 'VAR':
                table[name]['default'] = value
                block_name = table[name]['block'][0]
                if name in cls._block_args_table[block_name].get('args', dict()):
                    cls._block_args_table[block_name]['args'][name]['default'] = value
                if name in cls._block_args_table[block_name].get('var_cls', dict()):
                    cls._block_args_table[block_name]['var_cls'][name]['default'] = value
                cls._args_table[name][0]['default'] = value
        return table

    # descriptions
    def __get_connection_description(self, kwargs, connection=None, context_level=1):
        if isinstance(connection, Var):
            connection = connection.value('connection', context_level=context_level + 1)
        description = {**self._connection, **parse.args_dict('connection', kwargs, remove=True)}

        if connection is not None:
            if isinstance(connection, dict):
                description = {**description, **connection}
            else:
                description['kind'] = connection

        connection_kind = description.get('kind', 'normal')
        if isinstance(connection_kind, Var):
            connection_kind = description['kind'] = connection_kind.value(
                name='kind', prefix='connection', context_level=context_level + 1)

        if connection_kind:
            for name, value in description.items():
                description[name] = value.value(
                    prefix='connection', name=name, context_level=context_level + 1) if isinstance(
                    value, Var) else value
        return description

    def __get_repeat_description(self, kwargs, repeat=None, context_level=1):
        if isinstance(repeat, Var):
            repeat = repeat.value('repeat', context_level=context_level + 1)
        assert repeat is None or isinstance(repeat, (dict, int, bool)), 'unknown value is specified for repeat'
        repeat_description = {**self._repeat, **parse.args_dict('repeat', kwargs, remove=True)}

        if repeat is not None:
            if isinstance(repeat, dict):
                repeat_description = {**repeat_description, **repeat}
            else:
                repeat_description['count'] = repeat
        repeat_count = repeat_description.get('count', False)
        if isinstance(repeat_count, Var):
            repeat_count = repeat_description['count'] = repeat_count.value(
                name='count', prefix='repeat', context_level=context_level + 1)

        if repeat_count:
            for name, value in repeat_description.items():
                repeat_description[name] = value.value(
                    prefix='repeat', name=name, context_level=context_level + 1) if isinstance(value, Var) else value

        return repeat_description

    def __get_parallel_description(self, kwargs, parallel=None, context_level=1):
        pass

    # instantiation
    def __instantiate_repeat(self, kwargs, defaults, init):
        self.repeat['tied'] = self.repeat.get('tied', False)
        self.repeat['tied'] = 1 if self.repeat['tied'] is False else (
            self.repeat['count'] if self.repeat['tied'] is True else self.repeat['tied'])
        self.repeat['num_blocks'] = (self.repeat['count'] if not self.repeat['tied'] else (
            self.repeat['count'] // self.repeat['tied'] if isinstance(self.repeat['tied'], int) else 1)) + (
                                        0 if self.repeat['count'] % self.repeat['tied'] == 0 else 1)

        template = Block(
            # class name
            name='Block',
            # description
            repeat=None,
            connection=self.repeat['connection'],
            init=init,
            # blocks
            block=self._block,
            # args & defaults
            defaults=self._defaults,
            args=self._args,
        )

        for i in range(self.repeat['num_blocks']):
            blocks_left = self.repeat['count'] - i * self.repeat['tied']
            block = template(init=init, defaults=defaults, **kwargs)
            name = f'block-{i}' if self.repeat['tied'] == 1 else \
                f'block-{i}-[{self.repeat["tied"] if blocks_left >= self.repeat["tied"] else blocks_left}]'
            setattr(self, f'block-{i}', block)

    def initialize_weights(self, context_level=1):
        pass

    @classmethod
    def is_active(cls, prefix=None, context_level=1):
        if isinstance(cls._active, Var):
            return cls._active.value(prefix=prefix, name='active', context_level=context_level + 1)
        return cls._active

    @classmethod
    def update_defaults(cls, defaults):
        cls._defaults = {**cls._defaults, **defaults}
        cls._args_table = cls.args_table()
        cls._translation_table = cls.translation_table()
        cls._block_args_table = cls.block_args_table()
        return cls

    def _forward(self, inputs):
        if self.repeat['count']:
            pass
