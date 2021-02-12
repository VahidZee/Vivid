import typing as th
from vivid.utilities.variables import Var, var_args_description
from .block import _Block
import torch
from collections import OrderedDict
from vivid.utilities import parse

try:
    # Literal is available in python > 3.8
    CONNECTION_KINDS = th.Literal["normal", "residual", "dense", "other"]
except AttributeError:
    CONNECTION_KINDS = th.Any


def Block(
        # class description
        name: th.Optional[str] = None,
        args: th.Optional[th.Dict[str, th.Any]] = None,
        active: th.Optional[th.Union[Var, bool]] = True,
        defaults: th.Optional[dict] = None,

        # initialization
        init: th.Optional[th.Union[th.Dict[str, th.Any], th.Any]] = None,
        init_blacklist: th.Optional[th.Union[th.Dict[str, th.Any], Var]] = None,

        # input/output
        inputs=None,
        outputs=None,

        # connection
        connection: th.Optional[th.Union[str, dict, Var]] = None,
        connection_kind: th.Optional[th.Union[CONNECTION_KINDS, Var]] = None,
        connection_link=None,
        connection_reduction=None,

        # repetition
        repeat: th.Optional[th.Union[int, dict, bool]] = None,
        repeat_count: th.Optional[th.Union[bool, int, Var]] = None,
        repeat_tied: th.Optional[th.Union[bool, int, Var]] = None,
        repeat_connection: th.Optional[CONNECTION_KINDS] = None,

        # parallel
        parallel: th.Optional[th.Union[dict, bool]] = None,
        parallel_args=None,
        parallel_names=None,
        parallel_reduction=None,

        # modules
        **blocks: th.OrderedDict[str, th.Union[_Block, th.Any]],
):
    # connection
    assert connection is None or isinstance(connection, (str, dict, Var)), 'unknown "connection" is specified'
    if connection is None or isinstance(connection, (dict, str)):
        connection = connection or dict()
        if isinstance(connection, str):
            assert connection_kind is not None, 'inconsistent values are provided for "connection"'
            connection = dict(kind=connection)
        connection['kind'] = connection_kind if connection_kind is not None else connection.get('kind', None)
        connection['link'] = connection_link if connection_link is not None else connection.get('link', None)
        connection['reduction'] = connection_reduction if connection_reduction is not None else connection.get(
            'reduction', None)
    else:
        assert (connection_kind is not None or connection_link is not None or
                connection_reduction is not None), 'inconsistent values are provided for "connection"'

    # repeat
    assert repeat is None or isinstance(repeat, (int, dict, Var)), 'unknown "repeat" is specified'
    if repeat is None or isinstance(repeat, (dict, int)):
        repeat = repeat or dict()
        if isinstance(repeat, int):
            assert repeat is not None, 'inconsistent values are provided for "repeat"'
            repeat = dict(count=repeat)
        repeat['count'] = repeat_count if repeat_count is not None else repeat.get('count', 0)
        repeat['tied'] = repeat_tied if repeat_tied is not None else repeat.get('tied', False)
        repeat['connection'] = repeat_connection if repeat_connection is not None else repeat.get(
            'connection', None)
    else:
        assert (repeat_count is not None or repeat_tied is not None or
                repeat_connection is not None), 'inconsistent values are provided for "repeat"'

    # process args & blocks
    args_dict = dict(args=args or dict())
    defaults_dict = dict(defaults=defaults or dict())
    blocks_dict = parse.unsqueeze_dict(blocks, defaults=defaults_dict, args=args_dict)
    defaults_dict = parse.squeeze_dict(defaults_dict, base_dict='defaults')

    cls = type(
        'Block' if name is None else name,
        (_Block, torch.nn.Module),
        {
            '_defaults': defaults_dict,
            '_active': active,
            '_block': blocks_dict,
            '_args': args_dict,
            '_init': init,
            '_connection': connection,
            '_repeat': repeat,
            '_inputs': inputs,
            '_outputs': outputs,
        }
    )
    cls._args_table = cls.args_table()
    cls._block_args_table = cls.block_args_table()
    cls._translation_table = cls.translation_table()
    return cls
