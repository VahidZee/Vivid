def args_dict(prefix, args_dict: dict, remove=False):
    result = {f'{key.replace(f"{prefix}_", "")}': value for key, value in args_dict.items() if
              key.startswith(f"{prefix}_")}
    if not remove:
        return result
    to_remove = [key for key in args_dict.keys() if key.startswith(f"{prefix}_")]
    for item in to_remove:
        del args_dict[item]
    return result
