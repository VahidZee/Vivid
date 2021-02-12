def __repr_value(value):
    return str(value) if not isinstance(value, str) else f'"{value}"'


def __repr_arg_name(arg_name: str, description: dict):
    name = f'{arg_name}' if len(description.get('lookup', [])) <= 1 else f'[{arg_name}]'
    return f'{name}' if description.get('active', True) is True else f'{name}*'


def _repr_arg(arg_name: str, description: dict):
    name = __repr_arg_name(arg_name, description)
    return name if 'default' not in description else f'{name}={__repr_value(description["default"])}'


class _BlockRepr(type):
    def __repr__(self):
        cls_name = self.__name__
        immediate_args = self._translation_table
        # if 'default' not in value else f'{name}={value["default"]}'
        required_variables = [
            _repr_arg(name, value) for name, value in immediate_args.items() if 'default' not in value
        ]
        default_variables = [
            _repr_arg(name, value) for name, value in immediate_args.items() if 'default' in value
        ]

        return f'{cls_name}({", ".join(required_variables + default_variables)})'
