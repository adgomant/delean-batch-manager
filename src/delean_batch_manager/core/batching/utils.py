# -*- coding: utf-8 -*-

import logging
import json

from ..utils.misc import mask_path


def read_json(path, encoding="utf-8"):
    """
    Reads a JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        dict or list: Parsed JSON content.
    """
    with open(path, 'r', encoding=encoding) as f:
        return json.load(f)


def write_json(data, path, indent=4, encoding="utf-8"):
    """
    Writes data to a JSON file.

    Args:
        data (dict or list): Data to write.
        path (str): Destination file path.
        indent (int): Indentation level for formatting.
    """
    with open(path, 'w', encoding=encoding) as f:
        json.dump(data, f, indent=indent)


def save_batch_id_map_to_file(batch_id_map, path, verbose=True):
    """
    Save a subfolder -> batch ID map to a JSON file.
    """
    path = str(path)
    if not path.endswith('.json'):
        raise ValueError("Path must end with .json")
    write_json(batch_id_map, path)
    if verbose:
        logging.info(f"Saved {len(batch_id_map)} batch IDs to {mask_path(path)}.")


def load_batch_id_map_from_file(path, verbose=True):
    """
    Load a subfolder -> batch ID map from a JSON file.
    """
    path = str(path)
    if not path.endswith('.json'):
        raise ValueError("Path must end with .json")
    batch_id_map = read_json(path)
    if verbose:
        logging.info(f"Loaded {len(batch_id_map)} batch IDs from {mask_path(path)}.")
    return batch_id_map


def save_batch_metadata(batch_metadata, path):
    """
    Save the full OpenAI batch job metadata object to JSON.
    """
    path = str(path)
    if not path.endswith('.json'):
        raise ValueError("Path must end with .json")
    write_json(batch_metadata, path)
    logging.info(f"Saved batch metadata to {mask_path(path)}.")

