# -*- coding: utf-8 -*-

import os
import logging
import json
import math
import re
from collections import defaultdict
import pandas as pd
import polars as pl
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Optional, Union

from ..utils import mask_path


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


ACR2FULL = {
    'AS' : 'Attention and Search',
    'CEc': 'Comprehension',
    'CEe': 'Expression',
    'CL' : 'Conceptualization, Learning, and Abstraction',
    'KNn': 'Knowledge in Natural Sciences',
    'KNa': 'Knowledge in Applied Sciences and Professions',
    'KNc': 'Customary Everyday Knowledge',
    'KNf': 'Knowledge in Formal Sciences',
    'KNs': 'Knowledge in Social Sciences and Humanities',
    'MCt': 'Critical Thinking Processes',
    'MCu': 'Calibrating Knowns and Unknowns',
    'MCr': 'Identifying Relevant Information',
    'MS' : 'Mind Modelling and Social Cognition',
    'QLq': 'Quantitative Reasoning',
    'QLl': 'Logical Reasoning',
    'SNs': 'Spatial-physical Reasoning',
    'VO' : 'Volume',
    'AT' : 'Atypicality'
}


def read_demand_levels_rubric_files(rubrics_folder: str) -> dict:
    """
    Read rubric files from the specified folder.

    Args:
        rubrics_folder (str): Path to the folder containing rubric files.

    Returns:
        dict: Dictionary with rubric names as keys and their content as values.
    """
    rubrics = {}
    # Read all rubric files
    for filename in os.listdir(rubrics_folder):
        if filename.endswith('.txt'):
            filepath = os.path.join(rubrics_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                # Store the content with the rubric name as key
                # Remove 'rubric_' prefix and '.txt' suffix to get clean key
                key = filename.replace('.txt', '')
                rubrics[key] = f.read()
    return rubrics


def get_full_instruction(subdomain, rubric_content, prompt, max_tokens):
    """
    Generate the full instruction for a given subdomain, rubric content, and task instance.

    Args:
        subdomain (str): The subdomain for which the prompt is generated.
        rubric_content (str): The content of the rubric for the subdomain.
        prompt (str): The task instance prompt.
        max_tokens (int): Maximum number of tokens for the completion.

    Returns:
        str: The full prompt string.
    """
    instruction = (
        'INSTRUCTION: '
        f'Score the level of *{subdomain}* demanded by the given TASK INSTANCE using a discrete value from 0 to 5. '
        'Use CHAIN-OF-THOUGHTS REASONING to reason step by step before assigning the score. '
        #f'Ensure you always provide a score in less than {max_tokens} tokens, even if that implies less CHAIN-OF-THOUGHTS REASONING STEPS. '
        'After the CHAIN-OF-THOUGHTS REASONING STEPS, conclude your assessment with the statement: '
        f'"Thus, the level of *{subdomain}* demanded by the given TASK INSTANCE is: SCORE"'
        ', where SCORE is an integer score you have determined.')

    full_prompt = (
        f"The following rubric describes six distinct levels of *{subdomain}* required by different tasks:"
        f"\n{rubric_content}\n\nTASK INSTANCE: {prompt}"
        f"\n\n{instruction}"
        f'\n\nCHAIN-OF-THOUGHTS REASONING STEPS to score the level of *{subdomain}* demanded by the given TASK INSTANCE above:'
    )

    return full_prompt


def create_subdomain_batch_input_files(
        prompt_data: List[dict],
        rubrics: dict,
        output_dir: str,
        max_completion_tokens: int = 1000,
        openai_model: str = "gpt-4o",
        body_url: str = "/chat/completions",
        max_lines_per_file: Optional[int] = None,
        max_bytes_per_file: Optional[int] = None
    ) -> List[str]:
    """
    Create JSONL batch files for each subdomain based on the provided prompt data and rubrics.

    Args:
        prompt_data (list of dicts): List of dictionaries containing the prompts and their indices.
        rubrics (dict): Dictionary containing the rubrics for each subdomain.
        output_dir (str): Directory to save subdomain directories containing the batch input files.
        max_completion_tokens (int): Maximum number of tokens for the completion. The larger the better but may become costly.
        openai_model (str): OpenAI model name.
        body_url (str): URL for the request dict. Defaults to "/chat/completions".
        max_lines_per_file (int): Maximum number of lines per file.
        max_bytes_per_file (int): Maximum size of each file in bytes.

    Returns:
        List of paths to the created batch input files.
    """
    # System message
    system_message = {
        "role": "system",
        "content": ""
    }

    final_paths = []
    for subdomain, rubric_content in tqdm(rubrics.items(), desc="Subdomains completed: "):
        subdomain_dir = os.path.join(output_dir, subdomain)
        os.makedirs(subdomain_dir, exist_ok=True)
        output_file = os.path.join(subdomain_dir, 'input.jsonl')

        with open(output_file, 'w') as f:
            for row in prompt_data:
                # Combine rubric with base prompt and any row-specific content
                prompt = row['prompt']

                full_prompt = get_full_instruction(ACR2FULL[subdomain],
                                                   rubric_content,
                                                   prompt,
                                                   max_completion_tokens)

                request = {
                    "custom_id": row['idx'],
                    "method": "POST",
                    "url": body_url,
                    "body": {
                        "model": openai_model,
                        "messages": [
                            system_message,
                            {
                                "role": "user",
                                "content": full_prompt
                            }
                        ],
                        "max_tokens": max_completion_tokens,
                        "temperature": 0
                    }
                }

                f.write(json.dumps(request) + '\n')

        need_to_split = need_to_split_file(
            n_lines=len(rows),
            n_bytes=os.path.getsize(output_file),
            max_lines_per_file=max_lines_per_file,
            max_bytes_per_file=max_bytes_per_file
        )
        if need_to_split:
            part_paths = split_large_file(
                output_file,
                max_bytes_per_file=max_bytes_per_file,
                max_lines_per_file=max_lines_per_file
            )
            final_paths.extend(part_paths)
            logging.info(f"{subdomain} batch file splitted into {len(part_paths)} parts")
        else:
            final_paths.append(output_file)
            logging.info(f"Created batch file for subdomain {subdomain}")

    logging.info(f"Total files created: {len(final_paths)}")

    return final_paths


def parse_subdomain_batch_output_files(
        base_folder,
        output_path=None,
        return_pandas=False,
        verbose=False
    ) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Parse the output files from the subdomain batch processing to extract demand levels.

    Args:
        base_folder (str): Path to the folder containing subdomain directories with the output files.
        output_path (str, optional): Path to save the results. Can be folder or file path. 
            If folder, it will be saved with default name. If None, results are not saved.
        return_pandas (bool, optional): If True, returns a pandas DataFrame instead of polars.
        verbose (bool, optional): If True, logs warnings for any issues encountered when extracting demand levels.

    Returns:
        A DataFrame containing the parsed outputs. 
        It includes subdomain, custom_id, finish_reason, COT steps, conclusion, and demand level.
    """
    demand_levels = defaultdict(list)
    for file in Path(base_folder).glob('*/output.jsonl'):
        subdomain = Path(file).parent.name.split('_')[0]
        data = [json.loads(line) for line in open(file, 'r')]
        for output in data:
            custom_id = output['custom_id']
            finish_reason = output['response']['body']['choices'][0]['finish_reason']
            res = output['response']['body']['choices'][0]['message']['content']
            *cot_steps, conclusion = res.split('\n\n')
            if finish_reason == 'stop':
                try:
                    demand_level = float(re.findall(r'\d', conclusion)[-1])
                except IndexError:
                    if verbose:
                        logging.warning(f"Error extracting demand level from item '{custom_id}' in subdomain '{subdomain}'. Conclusion: '{conclusion}'")
                    demand_level = float('nan')
            else:
                demand_level = float('nan')
            demand_levels['subdomain'].append(subdomain)
            demand_levels['idx'].append(custom_id)
            demand_levels['finish_reason'].append(finish_reason)
            demand_levels['cot'].append("\n\n".join(cot_steps))
            demand_levels['conclusion'].append(conclusion)
            demand_levels['demand_level'].append(demand_level)

    demand_levels_data = pl.DataFrame(demand_levels)

    if output_path is not None:
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, 'annotations.csv')
        demand_levels_data.write_csv(output_path)
        logging.info(f"Demand Levels Annotation results saved to {mask_path(output_path)}")

    if return_pandas:
        demand_levels_data = demand_levels_data.to_pandas()

    return demand_levels_data


def need_to_split_file(n_lines, n_bytes, max_lines_per_file, max_bytes_per_file):
    """
    Determines if a file needs to be split based on the number of lines and bytes.
    """
    if max_lines_per_file is not None and n_lines > max_lines_per_file:
        return True
    if max_bytes_per_file is not None and n_bytes > max_bytes_per_file:
        return True
    return False


def split_large_file(input_file, max_bytes_per_file=None, max_lines_per_file=None):
    """
    Splits a large JSONL file based on one or both of:
      - max_bytes_per_file: maximum size in bytes per part
      - max_lines_per_file: maximum number of lines per part

    You must specify at least one of these arguments.
    If both are provided, the function first computes the number of parts
    needed to satisfy the size constraint, then adjusts if that would
    violate the line constraint.

    Args:
        input_file (str): Path to the JSONL file.
        max_bytes_per_file (int, optional): Maximum size in bytes per part.
        max_lines_per_file (int, optional): Maximum lines per part.

    Returns:
        list: List of paths to the split files.
    """
    if max_bytes_per_file is None and max_lines_per_file is None:
        raise ValueError("Please specify max_bytes_per_file, max_lines_per_file, or both")

    # Read all lines up front to count total_lines
    with open(input_file, 'r') as f:
        lines = f.readlines()
    total_lines = len(lines)

    # if size constraint is given, compute parts by size
    if max_bytes_per_file is not None:
        total_bytes = os.path.getsize(input_file)
        num_parts = math.ceil(total_bytes / max_bytes_per_file)

        # if both constraints, adjust parts to respect max_lines_per_file
        if max_lines_per_file is not None:
            base_lines = total_lines // num_parts
            if base_lines >= max_lines_per_file:
                num_parts = math.ceil(total_lines / max_lines_per_file)

    else:
        # only line constraint
        num_parts = math.ceil(total_lines / max_lines_per_file)

    # compute per-part line counts
    lines_per_part = total_lines // num_parts
    remainder = total_lines % num_parts

    base_subdomain_dir = Path(input_file).parent
    part_paths = []

    idx = 0
    # write out each part
    for i in range(num_parts):
        part_size = lines_per_part + (1 if i < remainder else 0)
        part_dir = f"{base_subdomain_dir}_part{i+1}"
        os.makedirs(part_dir, exist_ok=True)
        part_filename = os.path.join(part_dir, "input.jsonl")

        with open(part_filename, 'w') as out_f:
            for _ in range(part_size):
                if idx < total_lines:
                    # write the line to the part file
                    out_f.write(lines[idx])
                    idx += 1

        part_paths.append(part_filename)

    # remove the original file
    os.remove(input_file)
    if not os.listdir(base_subdomain_dir):
        os.rmdir(base_subdomain_dir)

    return part_paths


def halve_large_file(input_file):
    """
    Splits a large JSONL file into exactly two parts.

    Args:
        input_file: Path to the input JSONL file
        max_lines_per_file: Maximum number of lines per split file (ignored if specified)
    """
    # Read the total number of lines in the file
    with open(input_file, 'r') as f:
        total_lines = sum(1 for _ in f)

    # Always split into two equal parts
    lines_per_part = total_lines // 2

    base_name = os.path.splitext(input_file)[0]

    with open(input_file, 'r') as input_f:
        # Write first part
        with open(f"{base_name}_part1.jsonl", 'w') as output_f:
            for _ in range(lines_per_part):
                line = input_f.readline()
                output_f.write(line)

        # Write second part (remaining lines)
        with open(f"{base_name}_part2.jsonl", 'w') as output_f:
            for line in input_f:  # Read all remaining lines
                output_f.write(line)

    # Remove the original file
    os.remove(input_file)
    return 2
