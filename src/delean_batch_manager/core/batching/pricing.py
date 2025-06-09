# -*- coding: utf-8 -*-

import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from typing import List, Dict, Optional

from .files import get_full_instruction
from ..utils.misc import resolve_n_jobs


MODEL2PRICE = {
    # Pricing in USD per 1M tokens for Batch API
    # (https://platform.openai.com/docs/pricing)
    'gpt-4.1'            : {'input': 1,     'output': 4   },
    'gpt-4.1-mini'       : {'input': 0.2,   'output': 0.8 },
    'gpt-4.1-nano'       : {'input': 0.05,  'output': 0.2 },
    'gpt-4o'             : {'input': 1.25,  'output': 5   },
    'gpt-4o-mini'        : {'input': 0.075, 'output': 0.3 },
    'o1'                 : {'input': 7.5,   'output': 30  },
    'o1-mini'            : {'input': 0.55,  'output': 2.2 },
    'o3'                 : {'input': 5,     'output': 20  },
    'o3-mini'            : {'input': 0.55,  'output': 2.2 },
    'o4-mini'            : {'input': 0.55,  'output': 2.2 },
    'gpt-4'              : {'input': 15,    'output': 30  },
    'gpt-4-turbo'        : {'input': 5,     'output': 15  },
    'gpt-3.5-turbo'      : {'input': 0.25,  'output': 0.75},
}


def get_model_pricing(model):
    """
    Get the pricing for a specific OpenAI model.

    Args:
        model (str): The OpenAI model name.

    Returns:
        dict: A dictionary with input and output pricing.
    """
    for model_name, pricing in MODEL2PRICE.items():
        if model_name in model:
            return pricing, True
    return {"input": 0, "output": 0}, False


def get_instruction_tokens(instruction, encoding):
    """
    Count the number of tokens in the instruction.

    Args:
        instruction (str): Instruction string.
        encoding (tiktoken.core.Encoding): Encoding object for the OpenAI model.

    Returns:
        int: Number of tokens in the instruction.
    """
    instruction_tokens = len(encoding.encode(instruction))
    return instruction_tokens


def get_subdomain_tokens(rubrics, encoding):
    """
    Count the number of tokens in the subdomain names and rubric content.

    Args:
        rubrics (dict): Dictionary containing the rubrics for each subdomain.
        encoding (tiktoken.core.Encoding): Encoding object for the OpenAI model.

    Returns:
        dict: Dictionary with subdomain names as keys and their token counts as values.
    """
    subdomain_tokens = {}
    for acronym, subdomain_dict in rubrics.items():
        # Count tokens in the rubric content
        subdomain_tokens[acronym] = {
            'full_name': len(encoding.encode(subdomain_dict['full_name'])),
            'rubric': len(encoding.encode(subdomain_dict['content']))
        }
    return subdomain_tokens


def get_prompt_tokens(prompts, encoding):
    """
    Count the number of tokens in the prompts.

    Args:
        prompts (list): List of prompts.
        encoding (tiktoken.core.Encoding): Encoding object for the OpenAI model.

    Returns:
        int: Total number of tokens in the prompts.
    """
    total_tokens = sum(len(encoding.encode(prompt)) for prompt in prompts)
    return total_tokens


def get_total_tokens_aprox(prompts, rubrics, encoding):
    """
    Calculate the approximate total number of tokens for the given prompt data and rubrics.
    Based on the static structure of the full prompt across all texts and subdomains.

    Args:
        prompts (list): List of prompts.
        rubrics (dict): Dictionary containing the rubrics for each subdomain.
        max_completion_tokens (int): Maximum number of tokens for the completion.
        encoding (tiktoken.core.Encoding): Encoding object for the OpenAI model.

    Returns:
        int: Total number of input tokens.
    """
    total_input_tokens = 0
    n_prompts = len(prompts)
    prompt_tokens = get_prompt_tokens(prompts, encoding)
    subdomain_tokens = get_subdomain_tokens(rubrics, encoding)
    empty_instuction = get_full_instruction('', ' ', '')
    instruction_tokens = get_instruction_tokens(empty_instuction, encoding)
    for subdomain, tokens_info in subdomain_tokens.items():
        total_input_tokens += (
            prompt_tokens +
            n_prompts * 4 * tokens_info['full_name'] +
            n_prompts * tokens_info['rubric'] +
            n_prompts * instruction_tokens
        )
    return total_input_tokens


def get_total_tokens_exact_serial(prompts, rubrics, encoding):
    """
    Calculate the exact total number of tokens for the given prompt data and rubrics.

    Args:
        prompts (list): List of prompts.
        rubrics (dict): Dictionary containing the rubrics for each subdomain.
        max_completion_tokens (int): Maximum number of tokens for the completion.
        encoding (tiktoken.core.Encoding): Encoding object for the OpenAI model.

    Returns:
        int: Total number of input tokens.
    """
    total_input_tokens = 0
    for acronym, subdomain_dict in rubrics.items():
        for prompt in tqdm(prompts, total=len(prompts), desc=f"Tokenizing subdomain {acronym:<3}"):
            full_prompt = get_full_instruction(
                subdomain_dict['full_name'], subdomain_dict['content'], prompt
            )
            total_input_tokens += get_instruction_tokens(full_prompt, encoding)
    return total_input_tokens


def get_total_tokens_exact_parallel(prompts, rubrics, encoding, max_workers=-1):
    """
    Calculate the exact total number of tokens for the given prompt data and rubrics.
    Parallel computation: each subdomain is processed in a separate thread.
    """
    def subdomain_token_sum(subdomain, rubric_content):
        total = 0
        for prompt in prompts:
            full_prompt = get_full_instruction(subdomain, rubric_content, prompt)
            total += get_instruction_tokens(full_prompt, encoding)
        return total

    total_input_tokens = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for acronym, subdomain_dict in rubrics.items():
            futures.append(
                executor.submit(
                    subdomain_token_sum,
                    subdomain_dict['full_name'],
                    subdomain_dict['content']
                )
            )
        for future in tqdm(as_completed(futures), total=len(futures), desc="Tokenizing per subdomain"):
            total_input_tokens += future.result()
    return total_input_tokens


def get_encoding(openai_model):
    """
    Get the encoding for the specified OpenAI model.

    Args:
        openai_model (str): OpenAI model name.

    Returns:
        tiktoken.core.Encoding: Encoding object for the OpenAI model.
    """
    try:
        encoding = tiktoken.encoding_for_model(openai_model)
    except KeyError:
        if openai_model in ["o1", "o1-mini", "o3", "o3-mini", "o4-mini"]:
            encoding = tiktoken.get_encoding("o200k_base")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")
    return encoding


def get_batch_api_pricing(
        prompts: List[str],
        rubrics: Dict[str, str],
        max_completion_tokens: int,
        openai_model: str = "gpt-4o",
        estimation: str = 'aprox',
        n_jobs: Optional[int] = -1,
    ) -> float:
    """
    Calculate the cost of using the OpenAI Batch API for the given prompt data and rubrics.
    Take into account that the obtained cost is actually an upper bound of the real cost since
    the model may not use all the tokens in all the completions.

    Args:
        prompts (list): List of benchmark prompts (not the actual full prompts given to the model).
        rubrics (dict): Dictionary containing the rubrics for each subdomain.
        max_completion_tokens (int): Maximum number of tokens for the completion.
        openai_model (str): OpenAI model name.
        estimation (str): Type of estimation to use ('aprox' or 'exact').
            - 'aprox': Approximate cost (99.95% accurate). Extremely much faster. Recommended for large datasets.
            - 'exact': Exact cost. Slower but accurate.
        n_jobs (int or None): Number of parallel jobs for exact estimation.
            Default is -1 (all available cores). If None, runs in serial mode.

    Returns:
        float: Estimated cost in USD.
    """
    # Check if the model is valid    
    pricing, found = get_model_pricing(openai_model)
    if not found:
        raise ValueError(f"Model {openai_model} not found in pricing data. Supported models are: {', '.join(MODEL2PRICE.keys())}")

    encoding = get_encoding(openai_model)

    # Calculate input cost
    if estimation == 'aprox':
        total_input_tokens = get_total_tokens_aprox(prompts, rubrics, encoding)
    elif estimation == 'exact':
        if n_jobs is None:
            total_input_tokens = get_total_tokens_exact_serial(prompts, rubrics, encoding)
        else:
            max_workers = resolve_n_jobs(n_jobs, verbose=False)
            total_input_tokens = get_total_tokens_exact_parallel(prompts, rubrics, encoding, max_workers=max_workers)
    else:
        raise ValueError("Invalid estimation type. Use 'aprox' or 'exact'.")
    cost_per_1M_input_tokens = pricing['input']
    estimated_input_cost = (total_input_tokens / 1_000_000) * cost_per_1M_input_tokens

    # Calculate output cost
    total_output_tokens = len(prompts) * len(rubrics) * max_completion_tokens
    cost_per_1M_output_tokens = pricing['output']
    estimated_output_cost = (total_output_tokens / 1_000_000) * cost_per_1M_output_tokens

    # Calculate total estimated cost
    estimated_cost = estimated_input_cost + estimated_output_cost

    return estimated_cost
