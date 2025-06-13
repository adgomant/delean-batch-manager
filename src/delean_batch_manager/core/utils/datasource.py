# -*- coding: utf-8 -*-

import json
import polars as pl


def check_jsonl_source_data_keys(source_data_file):
    """Check if each object inside JSONL file has the required keys 'prompt' and 'custom_id'."""
    wrong_keys = {}
    with open(source_data_file, 'r') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
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


def read_source_data_jsonl(source_data_file):
    """Read source data from a JSONL file."""
    with open(source_data_file, 'r') as f:
        lines = [json.loads(line) for line in f]
    data = []
    for i, item in enumerate(lines):
        if 'prompt' not in item or 'custom_id' not in item:
            raise KeyError(f"Expected 'prompt' and 'custom_id' keys not found in line {i} of {source_data_file}.")
        data.append({
            'prompt': item['prompt'],
            'custom_id': item['custom_id']
        })
    return data


def read_source_data_tabular(source_data_file):
    """Read source data from a CSV or PARQUET file."""
    if source_data_file.endswith(".csv"):
        df = pl.read_csv(source_data_file)
    elif source_data_file.endswith('.parquet'):
        df = pl.read_parquet(source_data_file)
    else:
        raise ValueError('Source data file must be either CSV or PARQUET')
    if 'prompt' not in df.columns or 'custom_id' not in df.columns:
        raise KeyError(f"Expected 'prompt' and 'custom_id' columns not found in {source_data_file}.")
    df = df.select(['prompt', 'custom_id'])
    df = df.to_dicts()
    return df


def read_source_data(source_data_file):
    """Read source data from a JSONL, CSV or PARQUET file."""
    if source_data_file.endswith('.jsonl'):
        return read_source_data_jsonl(source_data_file)
    elif source_data_file.endswith('.csv') or source_data_file.endswith('.parquet'):
        return read_source_data_tabular(source_data_file)
    else:
        raise ValueError("Source data file must be a JSONL, CSV or PARQUET file.")


def read_only_prompts_from_source_data_jsonl(source_data_file):
    """Read only prompts from a JSONL source data file."""
    prompts = []
    with open(source_data_file, 'r') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            prompt = item.get('prompt')
            if prompt is not None:
                prompts.append(item['prompt'])
            else:
                raise KeyError(f"Missing 'prompt' in line {i}")
    return prompts


def read_only_prompts_from_source_data_tabular(source_data_file):
    """Read only prompts from a CSV or PARQUET source data file."""
    if source_data_file.endswith('.csv'):
        df = pl.read_csv(source_data_file)
    elif source_data_file.endswith('.parquet'):
        df = pl.read_parquet(source_data_file)
    return df['prompt'].to_list()


def read_only_prompts_from_source_data(source_data_file):
    """Read only prompts from a JSONL or CSV source data file."""
    if source_data_file.endswith('.jsonl'):
        return read_only_prompts_from_source_data_jsonl(source_data_file)
    elif source_data_file.endswith('.csv') or source_data_file.endswith('.parquet'):
        return read_only_prompts_from_source_data_tabular(source_data_file)
    else:
        raise ValueError("Source data file must be a JSONL, CSV or PARQUET file.")
