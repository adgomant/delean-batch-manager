# -*- coding: utf-8 -*-

import os
import logging
import json
import math
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Optional


def get_full_instruction(subdomain, rubric_content, prompt):
    """
    Generate the full instruction for a given subdomain, rubric content, and task instance.

    Args:
        subdomain (str): The subdomain for which the prompt is generated.
        rubric_content (str): The content of the rubric for the subdomain.
        prompt (str): The task instance prompt.

    Returns:
        str: The full prompt string.
    """
    instruction = (
        'INSTRUCTION: '
        f'Score the level of *{subdomain}* demanded by the given TASK INSTANCE using a discrete value from 0 to 5. '
        'Use CHAIN-OF-THOUGHTS REASONING to reason step by step before assigning the score. '
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
        prompt_data (list of dicts): List of dictionaries containing 'prompt' and 'custom_id' keys.
        rubrics (dict): A dictionary of all rubrics with their full names and content indexed by acronym.
        output_dir (str): Directory to save subdomain directories containing the batch input files.
        max_completion_tokens (int): Maximum number of tokens for the completion. The larger the better but may become costly.
        openai_model (str): OpenAI model name or custom model identifier deployed on Azure OpenAI.
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
    for acronym, subdomain_dict in tqdm(rubrics.items(), desc="Subdomains completed: "):
        subdomain_dir = os.path.join(output_dir, acronym)
        os.makedirs(subdomain_dir, exist_ok=True)
        output_file = os.path.join(subdomain_dir, 'input.jsonl')

        with open(output_file, 'w', encoding='utf-8') as f:
            for row in prompt_data:
                # Combine rubric with base prompt and any row-specific content
                prompt = row['prompt']

                full_prompt = get_full_instruction(
                    subdomain=subdomain_dict['full_name'],
                    rubric_content=subdomain_dict['content'],
                    prompt=prompt,
                )

                request = {
                    "custom_id": row['custom_id'],
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
            n_lines=len(prompt_data),
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
            logging.info(f"{acronym} batch input file splitted into {len(part_paths)} parts")
        else:
            final_paths.append(output_file)
            logging.info(f"Created batch file for subdomain {acronym}")

    logging.info(f"Total files created: {len(final_paths)}")

    return final_paths


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
