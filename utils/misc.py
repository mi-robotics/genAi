

def flatten_dict(d, parent_key='', sep='_'):
    """
    Flatten a nested dictionary. Concatenate keys with underscore.

    :param d: Dictionary to flatten
    :param parent_key: String (optional) - A prefix to prepend to each key
    :param sep: String (optional) - Separator used to concatenate keys
    :return: A new flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)