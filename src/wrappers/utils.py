# -*- coding: utf-8 -*-

import os
import logging
import json
import openai
import polars as pl


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

#=======================================================================
# Authentication and API Clients Utilities
#=======================================================================

def create_openai_client(api_key=None):
    """
    Create an OpenAI client for API calls.

    Args:
        api_key (str): The OpenAI API key. If not provided, it will be fetched from the environment variable.
    """
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("No OpenAI API key provided or found in environment.")

    client = openai.OpenAI(api_key=api_key)
    logging.info("OpenAI client created successfully.")
    return client


def create_azure_openai_client(api_key=None, endpoint=None):
    """
    Create an Azure OpenAI client for API calls.

    Args:
        api_key (str): The Azure OpenAI API key. If not provided, it will be fetched from the environment variable.
        endpoint (str): The Azure OpenAI endpoint. If not provided, it will be fetched from the environment variable.
    """
    if api_key is None:
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("No Azure OpenAI API key provided or found in environment.")

    if endpoint is None:
        endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    if endpoint is None:
        raise ValueError("No Azure OpenAI endpoint provided or found in environment.")

    client = openai.AzureOpenAI(
        api_key=api_key,
        api_version="2025-03-01-preview",
        azure_endpoint=endpoint,
    )
    return client


#=======================================================================
# Source Data Utilities
#=======================================================================


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
            raise Exception(f"Expected 'prompt' and 'idx' keys not found in line {i} of {source_data_file}.")
        data.append({
            'prompt': item['prompt'],
            'idx': item['idx']
        })
    return data


def read_source_data_csv(source_data_file):
    """Read source data from a CSV file."""
    df = pl.read_csv(source_data_file)
    if 'prompt' not in df.columns or 'idx' not in df.columns:
        raise Exception(f"Expected 'prompt' and 'idx' columns not found in {source_data_file}.")
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
        for line in f:
            item = json.loads(line)
            prompts.append(item['prompt'])
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
