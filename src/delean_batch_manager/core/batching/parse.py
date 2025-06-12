# -*- coding: utf-8 -*-

import re
import json
import os
import logging
from pathlib import Path
from typing import Optional, Literal
from collections import defaultdict
import polars as pl

from ..utils.misc import mask_path


def extract_demand_level_from_response(response: str) -> float:
    """
    Extract the demand level from the response string.

    Args:
        conclusion (str): The response string from
            which to extract the demand level.

    Returns:
        float: The extracted demand level, or NaN if extraction fails.
    """
    *cot_steps, conclusion = response.split('\n\n')
    try:
        # Extract the last number from the conclusion
        match = re.findall(r'\d+', conclusion)
        demand_level = float(match[-1])
        if demand_level < 0 or demand_level > 5:
            # If the demand level is outside the expected range, 
            # result is considered invalid
            return float('nan'), False
        if len(match) == 1 and conclusion.startswith(str(demand_level)):
            # Avoid cases where the only number present in the final statement
            # is a leading section number (yes, this could happen)
            # e.g., "4. Conclusion: Thus, the level of Attention and Search
            # demanded by the given TASK INSTANCE is: **Not Applicable**"
            return float('nan'), False
        return demand_level, True
    except IndexError:
        return float('nan'), False


def parse_subdomain_batch_output_files(
        base_folder: str | Path,
        only_levels: bool = False,
        verbose: bool = False
    ) -> dict:
    """
    Parse the output files from the subdomain batch processing to extract demand levels.

    Args:
        base_folder (str): Path to the folder containing subdomain directories with the output files.
        only_levels (bool, optional): If True, only returns the demand levels without additional information.
        verbose (bool, optional): If True, logs warnings for any issues encountered when extracting demand levels.

    Returns:
        dict: A dictionary containing the parsed outputs with prompt custom id 
            as keys. It includes subdomain, finish_reason, model response and 
            demand level.
    """
    results = {}
    for file in Path(base_folder).glob('*/output.jsonl'):
        subdomain = Path(file).parent.name.split('_')[0]
        data = [json.loads(line) for line in open(file, 'r')]
        for item in data:
            custom_id = item['custom_id']
            finish_reason = item['response']['body']['choices'][0]['finish_reason']
            response = item['response']['body']['choices'][0]['message']['content']
            if finish_reason == 'stop':
                demand_level, ok = extract_demand_level_from_response(response)
                if not ok:
                    if verbose:
                        logging.warning(f"Error extracting demand level from item '{custom_id}' in subdomain '{subdomain}'. Response: {repr(response)}")
                    demand_level = float('nan')
                    failed = True
                failed = False
            else:
                demand_level = float('nan')
                failed = True
            if custom_id not in results:
                results[custom_id] = {
                    'demands': {}
                }
            results[custom_id]['demands'][subdomain] = {
                'level': demand_level
            }
            if not only_levels:
                results[custom_id]['demands'][subdomain]['finish_reason'] = finish_reason
                results[custom_id]['demands'][subdomain]['model_response'] = response
    return results


def save_parsed_results_jsonl(
        results: dict,
        output_path: Optional[str] = None,
    ):
    """
    Parse the results from the subdomain batch output files and save them to a JSONL file.

    Args:
        results (dict): Parsed results from the subdomain batch output files.
        output_path (str, optional): Path to save the results. If None, results are not saved.
            Path can be a file or a directory. If a directory is provided, the results will be saved as 'annotations.jsonl' in that directory.
    """
    lines = []
    for custom_id, demands in results.items():
        demands['idx'] = custom_id
        lines.append(demands)
    if output_path is not None:
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, 'annotations.jsonl')
        with open(output_path, 'w') as f:
            for line in lines:
                f.write(json.dumps(line) + '\n')
        logging.info(f"Results saved to {mask_path(output_path)}")


def save_parsed_results_csv(
        results: dict,
        output_path: Optional[str] = None,
        format: str = 'long',
    ):
    """
    Parse the results from the subdomain batch output files
    and save them to a CSV file.

    Args:
        results (dict): Parsed results from the subdomain batch output files.
        output_path (str, optional): Path to save the results.
            If None, results are not saved. Path can be a file or a directory.
            If a directory is provided, the results will be saved as
            'annotations.csv' in that directory.
        format (str, optional): Format of the output CSV file. Can be 'long' or 'wide'.
            - 'long' format has columns: idx, demand, level, finish_reason,
                model_response. If only_levels is True, it has idx, demand, level.
            - 'wide' format has prompt custom id as index and demands as
                columns with their levels as values. Note that this format won't
                include finish_reason or model_response information.
    """
    if format not in ['long', 'wide']:
        raise ValueError("format must be either 'long' or 'wide'")
    rows = defaultdict(list)
    for custom_id, demands in results.items():
        for subdomain, data in demands['demands'].items():
            rows['idx'].append(custom_id)
            rows['demand'].append(subdomain)
            for k, v in data.items():
                rows[k].append(v)
    df = pl.DataFrame(rows)
    if format == 'wide':
        df = df.select(['idx', 'demand', 'level'])
        df = df.pivot(
            index='idx',
            columns='demand',
            values='level',
        )
    if output_path is not None:
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, 'annotations.csv')
        df.write_csv(output_path)
        logging.info(f"Results saved to {mask_path(output_path)}")


def save_parsed_results(
        results: dict,
        file_type: Literal['jsonl', 'csv'] = 'jsonl',
        output_path: Optional[str] = None,
        csv_format: str = 'long',
    ):
    """
    Parse the results from the subdomain batch output files and save them to a file.

    Args:
        results (dict): Parsed results from the subdomain batch output files.
        file_type (str): Type of file to save the results. Can be 'jsonl' or 'csv'.
        output_path (str, optional): Path to save the results. If None, results are not saved.
            Path can be a file or a directory. If a directory is provided, the results will be saved as 'annotations.jsonl' or 'annotations.csv' in that directory.
        csv_format (str, optional): Format of the output CSV file. Only used if file_type is 'csv'. Can be 'long' or 'wide'.
            - If 'long', the CSV will have columns: idx, demand, level, finish_reason, model_response. 
                If only_levels is True, it has idx, demand, level.
            - If 'wide', the CSV will have prompt custom id as index and demands as columns with their levels as values.
                Note that the 'wide' format won't include finish_reason or model_response information.
    Raises:
        ValueError: If file_type is not 'jsonl' or 'csv'.
    """
    if file_type == 'jsonl':
        save_parsed_results_jsonl(
            results=results,
            output_path=output_path,
        )
    elif file_type == 'csv':
        save_parsed_results_csv(
            results=results,
            output_path=output_path,
            format=csv_format,
        )
    else:
        raise ValueError("file_type must be either 'jsonl' or 'csv'")
