# -*- coding: utf-8 -*-

import polars as pl
from pathlib import Path

from .misc import read_jsonl


def check_jsonl_source_data_keys(source_data_file):
    """Check if each object inside JSONL file has the required keys 'prompt' and 'custom_id'."""
    lines = read_jsonl(source_data_file)
    wrong_keys = {}
    for i, item in enumerate(lines):
        if 'prompt' not in item or 'custom_id' not in item:
            wrong_keys[i] = list(item.keys())
    if wrong_keys:
        return False, wrong_keys
    return True, None


def check_tabular_source_data_columns(source_data_file):
    """Check if the tabular file (CSV or PARQUET) has the required columns 'prompt' and 'custom_id'."""
    if source_data_file.endswith('.csv'):
        df = pl.read_csv(source_data_file)
    elif source_data_file.endswith('.parquet'):
        df = pl.read_parquet(source_data_file)
    else:
        raise ValueError("Source data file must be a CSV or PARQUET file.")
    if 'prompt' not in df.columns or 'custom_id' not in df.columns:
        return False
    return True


def read_source_data_jsonl(source_data_file, as_map=False):
    """Read source data from a JSONL file."""
    lines = read_jsonl(source_data_file)
    data = []
    for i, item in enumerate(lines):
        if 'prompt' not in item or 'custom_id' not in item:
            raise KeyError(f"Expected 'prompt' and 'custom_id' keys not found in line {i} of {source_data_file}.")
        data.append({
            'prompt': item['prompt'],
            'custom_id': item['custom_id']
        })

    if as_map:
        data = {item['custom_id']: item['prompt'] for item in data}

    return data


def read_source_data_tabular(source_data_file,  as_map=False):
    """Read source data from a CSV or PARQUET file."""
    source_data_file = Path(source_data_file)
    if source_data_file.suffix == '.csv':
        df = pl.read_csv(source_data_file)
    elif source_data_file.suffix == '.parquet':
        df = pl.read_parquet(source_data_file)
    else:
        raise ValueError('Source data file must be either CSV or PARQUET')

    if 'prompt' not in df.columns or 'custom_id' not in df.columns:
        raise KeyError(f"Expected 'prompt' and 'custom_id' columns not found in {source_data_file}.")

    df = df.select(['prompt', 'custom_id'])
    data = df.to_dicts()

    if as_map:
        data = {item['custom_id']: item['prompt'] for item in data}

    return data


def read_source_data(source_data_file, as_map=False):
    """Read source data from a JSONL, CSV or PARQUET file."""
    source_data_file = Path(source_data_file)
    if source_data_file.suffix == '.jsonl':
        return read_source_data_jsonl(source_data_file, as_map=as_map)
    if source_data_file.suffix in ['.csv', '.parquet']:
        return read_source_data_tabular(source_data_file, as_map=as_map)
    raise ValueError("Source data file must be a JSONL, CSV or PARQUET file.")


def read_only_prompts_from_source_data_jsonl(source_data_file):
    """Read only prompts from a JSONL source data file."""
    lines = read_jsonl(source_data_file)
    prompts = []
    for i, item in enumerate(lines):
        if 'prompt' not in item:
            raise KeyError(f"Expected 'prompt' key not found in line {i} of {source_data_file}.")
        prompts.append(item['prompt'])
    return prompts


def read_only_prompts_from_source_data_tabular(source_data_file):
    """Read only prompts from a CSV or PARQUET source data file."""
    source_data_file = Path(source_data_file)
    if source_data_file.suffix == '.csv':
        df = pl.read_csv(source_data_file)
    elif source_data_file.suffix == '.parquet':
        df = pl.read_parquet(source_data_file)
    else:
        raise ValueError("Source data file must be a CSV or PARQUET file.")
    return df['prompt'].to_list()


def read_only_prompts_from_source_data(source_data_file):
    """Read only prompts from a JSONL or CSV source data file."""
    source_data_file = Path(source_data_file)
    if source_data_file.suffix == '.jsonl':
        return read_only_prompts_from_source_data_jsonl(source_data_file)
    if source_data_file.suffix in ['.csv', '.parquet']:
        return read_only_prompts_from_source_data_tabular(source_data_file)
    raise ValueError("Source data file must be a JSONL, CSV or PARQUET file.")
