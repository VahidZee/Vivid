def args_dict(prefix, args_dict: dict, remove=False, short_hand=False):
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
