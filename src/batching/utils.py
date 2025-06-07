# -*- coding: utf-8 -*-

import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


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


def save_batch_id_map_to_file(batch_ids, path):
    """
    Save a subfolder -> batch ID map to a JSON file.
    """
    write_json(batch_ids, path)
    logging.info(f" Saved {len(batch_ids)} batch IDs to {mask_path(path)}.")


def load_batch_id_map_from_file(path):
    """
    Load a subfolder -> batch ID map from a JSON file.
    """
    batch_ids = read_json(path)
    logging.info(f"Loaded {len(batch_ids)} batch IDs from {mask_path(path)}.")
    return batch_ids


def save_batch_metadata(batch_metadata, path):
    """
    Save the full OpenAI batch job metadata object to JSON.
    """
    write_json(batch_metadata.model_dump(), path)
    logging.info(f"Saved batch metadata to {mask_path(path)}.")

