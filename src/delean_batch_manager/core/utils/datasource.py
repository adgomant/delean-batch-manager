# -*- coding: utf-8 -*-

import json
import polars as pl


def check_source_data_jsonl_keys(source_data_file):
    """Check if each object inside JSONL file has the required keys 'prompt' and 'idx'."""
    wrong_keys = {}
    with open(source_data_file, 'r') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            if 'prompt' not in item or 'idx' not in item:
                wrong_keys[i] = list(item.keys())
    if wrong_keys:
        return False, wrong_keys
    return True, None


def check_source_data_csv_columns(source_data_file):
    """Check if the CSV file has the required columns 'prompt' and 'idx'."""
    df = pl.read_csv(source_data_file)
    if 'prompt' not in df.columns or 'idx' not in df.columns:
        return False
    return True


def read_source_data_jsonl(source_data_file):
    """Read source data from a JSONL file."""
    with open(source_data_file, 'r') as f:
        lines = [json.loads(line) for line in f]
    data = []
    for i, item in enumerate(lines):
        if 'prompt' not in item or 'idx' not in item:
            raise KeyError(f"Expected 'prompt' and 'idx' keys not found in line {i} of {source_data_file}.")
        data.append({
            'prompt': item['prompt'],
            'idx': item['idx']
        })
    return data


def read_source_data_csv(source_data_file):
    """Read source data from a CSV file."""
    df = pl.read_csv(source_data_file)
    if 'prompt' not in df.columns or 'idx' not in df.columns:
        raise KeyError(f"Expected 'prompt' and 'idx' columns not found in {source_data_file}.")
    df = df.select(['prompt', 'idx'])
    df = df.to_dicts()
    return df


def read_source_data(source_data_file):
    """Read source data from a JSONL or CSV file."""
    if source_data_file.endswith('.jsonl'):
        return read_source_data_jsonl(source_data_file)
    elif source_data_file.endswith('.csv'):
        return read_source_data_csv(source_data_file)
    else:
        raise ValueError("Source data file must be a JSONL or CSV file.")


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


def read_only_prompts_from_source_data_csv(source_data_file):
    """Read only prompts from a CSV source data file."""
    df = pl.read_csv(source_data_file)
    return df['prompt'].to_list()


def read_only_prompts_from_source_data(source_data_file):
    """Read only prompts from a JSONL or CSV source data file."""
    if source_data_file.endswith('.jsonl'):
        return read_only_prompts_from_source_data_jsonl(source_data_file)
    elif source_data_file.endswith('.csv'):
        return read_only_prompts_from_source_data_csv(source_data_file)
    else:
        raise ValueError("Source data file must be a JSONL or CSV file.")
