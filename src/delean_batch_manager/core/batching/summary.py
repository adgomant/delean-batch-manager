# -*- coding: utf-8 -*-

import json
import math
import logging
from datetime import datetime
from collections import Counter
from statistics import mean, stdev
from pathlib import Path
from typing import List, Optional

from .pricing import get_model_pricing
from ..utils.misc import mask_path


def get_batch_summary_dict(batch_metadata: dict, output_file_path: str):
    """
    Generate a summary dictionary from batch metadata and output file.

    Args:
        batch_metadata (dict): JSON content of the batch metadata.
        output_file_path (str): Path to the output.jsonl file.

    Returns:
        dict: The formatted summary dictionary.
    """
    # Extract metadata info
    batch_id = batch_metadata["id"]
    status = batch_metadata["status"]
    created_at = datetime.fromtimestamp(batch_metadata["created_at"]).isoformat()
    completed_at = (
        datetime.fromtimestamp(batch_metadata.get("completed_at")).isoformat()
        if batch_metadata.get("completed_at") else "N/A"
    )
    duration = batch_metadata.get("completed_at", 0) - batch_metadata.get("created_at", 0)
    total = batch_metadata["request_counts"]["total"]
    completed = batch_metadata["request_counts"]["completed"]
    failed = batch_metadata["request_counts"]["failed"]
    errors = batch_metadata['errors']

    # Aggregate usage from output.jsonl
    prompt_tokens = 0
    completion_tokens = 0
    completion_tokens_list = []
    finish_reason_counter = Counter()
    with open(output_file_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            usage = obj.get("response", {}).get("body", {}).get("usage", {})
            model = obj.get("response", {}).get("body", {}).get("model", None)
            prompt_tokens += usage.get("prompt_tokens", 0)
            completion_tokens += usage.get("completion_tokens", 0)
            completion_tokens_list.append(usage.get("completion_tokens", 0))
            choice = obj.get("response", {}).get("body", {}).get("choices", [{}])[0]
            finish_reason = choice.get("finish_reason", "unknown")
            finish_reason_counter[finish_reason] += 1
    total_tokens = prompt_tokens + completion_tokens

    # Compute completion tokens stats
    avg_completion_tokens = mean(completion_tokens_list)
    std_completion_tokens = stdev(completion_tokens_list)
    max_completion_tokens = max(completion_tokens_list)
    min_completion_tokens = min(completion_tokens_list)

    # Compute cost
    cost_rates, _ = get_model_pricing(model)
    prompt_cost = (prompt_tokens / 1_000_000) * cost_rates["input"]
    completion_cost = (completion_tokens / 1_000_000) * cost_rates["output"]
    total_cost = prompt_cost + completion_cost

    # Finsh reason counts
    finish_reason_stop = finish_reason_counter.get('stop', 0)
    finish_reason_length = finish_reason_counter.get('length', 0)
    finish_reason_other = sum(count for reason, count in finish_reason_counter.items() if reason not in ['stop', 'length'])
    finish_reason_total = finish_reason_counter.total()

    # Generate summary dictionary
    subdomain_name = Path(output_file_path).parent.name
    summary = {
            "batch_id": batch_id,
            "model": model,
            "status": status,
            "created_at": created_at,
            "completed_at": completed_at,
            "duration": duration,
            "subdomain": subdomain_name,
            "requests": {
                "total": total,
                "completed": completed,
                "failed": failed,
            },
            "finish_reasons": {
                "stop": finish_reason_stop,
                "length": finish_reason_length,
                "other": finish_reason_other,
                "total": finish_reason_total
            },
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "avg_completion": avg_completion_tokens,
                "std_completion": std_completion_tokens,
                "max_completion": max_completion_tokens,
                "min_completion": min_completion_tokens,
                "total": total_tokens,
            },
            "costs": {
                "prompt": prompt_cost,
                "completion": completion_cost,
                "total": total_cost,
            }
        }
    if errors:
        summary['errors'] = {
            "error_file_id": batch_metadata.get('error_file_id', None),
            "errors": errors
        }
    return summary


def save_batch_summary(
    batch_metadata: dict,
    output_file_path: str,
    summary_path: Optional[str] = None,
    return_as: Optional[str] = None,
    save_dict: bool = False
):
    """
    Generate a summary text from batch metadata and output file.

    Args:
        batch_metadata (dict): JSON content of the batch metadata.
        output_file_path (str): Path to the output.jsonl file.
        summary_path (str): Path to save the summary text. If None, saves in the same folder as output.jsonl.
        return_as (str, optional): If 'dict', returns the summary as a dictionary; if 'print', prints the summary to console.
        save_dict (bool): If True, saves the summary dictionary as a JSON file alongside the summary text.

    Returns:
        dict or None: The summary dictionary if return_as == 'dict', else None.
    """
    summary_dict = get_batch_summary_dict(batch_metadata, output_file_path)

    # Generate summary text
    summary_lines = [
        f"Subdomain    : {summary_dict['subdomain']}",
        f"Batch ID     : {summary_dict['batch_id']}",
        f"Model        : {summary_dict['model']}",
        f"Status       : {summary_dict['status']}",
        f"Created at   : {summary_dict['created_at']}",
        f"Completed at : {summary_dict['completed_at']}",
        f"Duration     : {summary_dict['duration']} seconds",
        "",
    ]
    batch_jobs_total = summary_dict['requests']['total']
    batch_jobs_completed = summary_dict['requests']['completed']
    batch_jobs_failed = summary_dict['requests']['failed']
    summary_lines += [
        "=== Request Counts ===",
        f"Total     : {batch_jobs_total}",
        f"Completed : {batch_jobs_completed} ({(batch_jobs_completed / batch_jobs_total * 100) if batch_jobs_total else 0:.2f}%)",
        f"Failed    : {batch_jobs_failed} ({(batch_jobs_failed / batch_jobs_total * 100) if batch_jobs_total else 0:.2f}%)",
        "",
    ]
    finish_reasons_total = summary_dict['finish_reasons']['total']
    finish_reasons_stop = summary_dict['finish_reasons']['stop']
    finish_reasons_length = summary_dict['finish_reasons']['length']
    finish_reasons_other = summary_dict['finish_reasons']['other']
    summary_lines += [
        "=== Finish Reasons ===",
        f"Stop   : {finish_reasons_stop} ({(finish_reasons_stop / finish_reasons_total * 100) if finish_reasons_total else 0:.2f}%)",
        f"Length : {finish_reasons_length} ({(finish_reasons_length / finish_reasons_total * 100) if finish_reasons_total else 0:.2f}%)",
        f"Other  : {finish_reasons_other} ({(finish_reasons_other / finish_reasons_total * 100) if finish_reasons_total else 0:.2f}%)",
        "",
        "=== Token Usage ===",
        f"Total prompt tokens        : {summary_dict['tokens']['prompt']:,}",
        f"Total completion tokens    : {summary_dict['tokens']['completion']:,}",
        f"  - Avg. completion tokens : {summary_dict['tokens']['avg_completion']:.2f}",
        f"  - Std. completion tokens : {summary_dict['tokens']['std_completion']:.2f}",
        f"  - Max. completion tokens : {summary_dict['tokens']['max_completion']:.2f}",
        f"  - Min. completion tokens : {summary_dict['tokens']['min_completion']:.2f}",
        f"Total tokens               : {summary_dict['tokens']['total']:,}",
        "",
        "=== Estimated Costs (USD) ===",
        f"Prompt cost     : ${summary_dict['costs']['prompt']:.4f}",
        f"Completion cost : ${summary_dict['costs']['completion']:.4f}",
        f"Total cost      : ${summary_dict['costs']['total']:.4f}",
    ]

    if summary_dict.get('errors'):
        summary_lines.append("")
        summary_lines.append("=== Errors ===")
        summary_lines.append(f"Error file ID: {summary_dict['errors']['error_file_id']}")
        for error in summary_dict['errors']['errors']:
            summary_lines.append(f"- {error}")

    summary = "\n".join(summary_lines)

    # Save to file
    if summary_path is None:
        summary_path = Path(output_file_path).parent / "summary.txt"

    with open(summary_path, "w") as f:
        f.write(summary)

    logging.info(f"Batch summary saved to {mask_path(summary_path)}")

    # Save dict as JSON if requested
    if save_dict:
        json_path = Path(summary_path).with_suffix('.json')
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(summary_dict, jf, indent=2, ensure_ascii=False)
        logging.info(f"Batch summary dict saved to {mask_path(json_path)}")

    # Print to console if requested
    if return_as == 'print':
        print(summary)

    if return_as == 'dict':
        return summary_dict


def aggregate_completion_tokens_stats(batch_summary_list):
    """
    Aggregate completion tokens statistics from a list of batch summaries.
    All summary dicts in the list must contain a 'tokens' key containing a dict with 
    'completion', 'avg_completion', 'std_completion', 'max_completion', 'min_completion' keys.

    Args:
        batch_summary_list (List[dict]): List of summary dictionaries, one per batch.

    Returns:
        tuple: A tuple containing:
            - total_n (int): Total number of completion tokens across all batches.
            - overall_mean (float): Weighted mean of completion tokens.
            - pooled_std (float): Pooled standard deviation of completion tokens.
            - overall_max (float): Maximum completion tokens across all batches.
            - overall_min (float): Minimum completion tokens across all batches.
    """
    counts = [s['tokens']['completion'] for s in batch_summary_list]
    means = [s['tokens']['avg_completion'] for s in batch_summary_list]
    stds = [s['tokens']['std_completion'] for s in batch_summary_list]
    total_n = sum(counts)
    if total_n == 0:
        return 0, 0, 0, 0, 0

    n_requests = [s['requests']['total'] for s in batch_summary_list]

    # Global mean
    overall_mean = sum(n * m for n, m in zip(n_requests, means)) / sum(n_requests) 

    # Global std (within + between)
    sum_sq_diff = sum((n - 1) * (std ** 2) for n, std in zip(n_requests, stds))
    sum_sq_mean_diff = sum(n * (m - overall_mean) ** 2 for n, m in zip(n_requests, means))
    
    pooled_var = (sum_sq_diff + sum_sq_mean_diff) / (sum(n_requests) - 1) if sum(n_requests) > 1 else 0
    pooled_std = math.sqrt(pooled_var)

    overall_max = max(s['tokens']['max_completion'] for s in batch_summary_list)
    overall_min = min(s['tokens']['min_completion'] for s in batch_summary_list)

    return total_n, overall_mean, pooled_std, overall_max, overall_min


def get_general_summary_dict(batch_summary_list: List[dict]) -> dict:
    """
    Generate a general summary dictionary from a list of individual batch summaries.

    Args:
        batch_summary_list (List[dict]): List of summary dictionaries, one per batch.

    Returns:
        dict: The aggregated summary dictionary.
    """
    models = set(summary['model'] for summary in batch_summary_list)
    subdomains = set(summary['subdomain'] for summary in batch_summary_list)
    status_counts = Counter(summary['status'] for summary in batch_summary_list)
    status_completed = status_counts['completed']
    status_failed = status_counts['failed']
    status_others = sum(count for status, count in status_counts.items() if status not in ['completed', 'failed'])

    durations = [summary['duration'] for summary in batch_summary_list]
    mean_duration = mean(durations) if durations else 0
    std_duration = stdev(durations) if len(durations) > 1 else 0

    # Aggregate request counts
    total_requests = sum(s['requests']['total'] for s in batch_summary_list)
    completed_requests = sum(s['requests']['completed'] for s in batch_summary_list)
    failed_requests = sum(s['requests']['failed'] for s in batch_summary_list)

    # Aggregate finish reasons
    finish_reason_stop = sum(s['finish_reasons']['stop'] for s in batch_summary_list)
    finish_reason_length = sum(s['finish_reasons']['length'] for s in batch_summary_list)
    finish_reason_other = sum(s['finish_reasons']['other'] for s in batch_summary_list)
    finish_reason_total = finish_reason_stop + finish_reason_length + finish_reason_other

    # Aggregate tokens
    prompt_tokens = sum(s['tokens']['prompt'] for s in batch_summary_list)
    completion_tokens, *completion_tokens_stats = aggregate_completion_tokens_stats(batch_summary_list)
    avg_completion_tokens = completion_tokens_stats[0]
    std_completion_tokens = completion_tokens_stats[1]
    max_completion_tokens = completion_tokens_stats[2]
    min_completion_tokens = completion_tokens_stats[3]
    total_tokens = sum(s['tokens']['total'] for s in batch_summary_list)

    # Aggregate costs
    prompt_cost = sum(s['costs']['prompt'] for s in batch_summary_list)
    completion_cost = sum(s['costs']['completion'] for s in batch_summary_list)
    total_cost = sum(s['costs']['total'] for s in batch_summary_list)

    summary_dict = {
        "batches": len(batch_summary_list),
        "subdomains": list(subdomains),
        "models": list(models),
        "duration": {
            "mean": mean_duration,
            "std": std_duration
        },
        "status_counts": {
            "completed": status_completed,
            "failed": status_failed,
            "others": status_others,
            "total": len(batch_summary_list)
        },
        "request_counts": {
            "total": total_requests,
            "completed": completed_requests,
            "failed": failed_requests
        },
        "finish_reasons": {
            "stop": finish_reason_stop,
            "length": finish_reason_length,
            "other": finish_reason_other,
            "total": finish_reason_total
        },
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "avg_completion": avg_completion_tokens,
            "std_completion": std_completion_tokens,
            "max_completion": max_completion_tokens,
            "min_completion": min_completion_tokens,
            "total": total_tokens
        },
        "costs": {
            "prompt": prompt_cost,
            "completion": completion_cost,
            "total": total_cost
        }
    }

    return summary_dict


def save_general_summary(
    batch_summary_list: List[dict],
    summary_path: str,
    return_as: Optional[str] = None,
    save_dict: bool = False
):
    """
    Save a general summary file from a list of individual batch summaries.

    Args:
        batch_summary_list (List[dict]): List of summary dictionaries, one per batch.
        summary_path (str): Path to save the general summary text.
        return_as (str, optional): If 'dict', returns the summary as a dictionary; if 'print', prints the summary to console.
        save_dict (bool): If True, saves the summary dictionary as a JSON file alongside the summary text.
    """
    summary_dict = get_general_summary_dict(batch_summary_list)

    summary_lines = [
        "=== General Batch Summary ===",
        f"Number of batches      : {summary_dict['batches']}",
        f"Subdomains             : {list(summary_dict['subdomains'])}",
        f"Models                 : {list(summary_dict['models'])}",
        f"Avg. Duration          : {summary_dict['duration']['mean']:.2f} seconds (std: {summary_dict['duration']['std']:.2f})",
        "",
    ]

    batch_jobs_total = summary_dict['status_counts']['total']
    batch_jobs_completed = summary_dict['status_counts']['completed']
    batch_jobs_failed = summary_dict['status_counts']['failed']
    batch_jobs_others = summary_dict['status_counts']['others']
    summary_lines += [
        "=== Batch Job Statuses ===",
        f"Total     : {batch_jobs_total}",
        f"Completed : {batch_jobs_completed} ({(batch_jobs_completed / batch_jobs_total * 100) if batch_jobs_total else 0:.2f}%)",
        f"Failed    : {batch_jobs_failed} ({(batch_jobs_failed / batch_jobs_total * 100) if batch_jobs_total else 0:.2f}%)",
        f"Other     : {batch_jobs_others} ({(batch_jobs_others / batch_jobs_total * 100) if batch_jobs_total else 0:.2f}%)",
        "",
    ]
    request_counts_total = summary_dict['request_counts']['total']
    request_counts_completed = summary_dict['request_counts']['completed']
    request_counts_failed = summary_dict['request_counts']['failed']
    summary_lines += [
        "=== Individual Request Counts ===",
        f"Total     : {request_counts_total}",
        f"Completed : {request_counts_completed} ({(request_counts_completed / request_counts_total * 100) if request_counts_total else 0:.2f}%)",
        f"Failed    : {request_counts_failed} ({(request_counts_failed / request_counts_total * 100) if request_counts_total else 0:.2f}%)",
        "",
    ]
    finish_reasons_total = summary_dict['finish_reasons']['total']
    finish_reasons_stop = summary_dict['finish_reasons']['stop']
    finish_reasons_length = summary_dict['finish_reasons']['length']
    finish_reasons_other = summary_dict['finish_reasons']['other']
    summary_lines += [
        "=== Finish Reasons ===",
        f"Stop   : {finish_reasons_stop} ({(finish_reasons_stop / finish_reasons_total * 100) if finish_reasons_total else 0:.2f}%)",
        f"Length : {finish_reasons_length} ({(finish_reasons_length / finish_reasons_total * 100) if finish_reasons_total else 0:.2f}%)",
        f"Other  : {finish_reasons_other} ({(finish_reasons_other / finish_reasons_total * 100) if finish_reasons_total else 0:.2f}%)",
        "",
        "=== Token Usage ===",
        f"Total Prompt tokens        : {summary_dict['tokens']['prompt']:,}",
        f"Total Completion tokens    : {summary_dict['tokens']['completion']:,}",
        f"  - Avg. completion tokens : {summary_dict['tokens']['avg_completion']:.2f}",
        f"  - Std. completion tokens : {summary_dict['tokens']['std_completion']:.2f}",
        f"  - Max. completion tokens : {summary_dict['tokens']['max_completion']:.2f}",
        f"  - Min. completion tokens : {summary_dict['tokens']['min_completion']:.2f}",
        f"Total tokens               : {summary_dict['tokens']['total']:,}",
        "",
        "=== Estimated Costs (USD) ===",
        f"Prompt cost     : ${summary_dict['costs']['prompt']:.4f}",
        f"Completion cost : ${summary_dict['costs']['completion']:.4f}",
        f"Total cost      : ${summary_dict['costs']['total']:.4f}",
    ]

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines))

    logging.info(f"General summary saved to {mask_path(summary_path)}")

    # Save dict as JSON if requested
    if save_dict:
        json_path = Path(summary_path).with_suffix('.json')
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(summary_dict, jf, indent=2, ensure_ascii=False)
        logging.info(f"General summary dict saved to {mask_path(json_path)}")

    # Print to console if requested
    if return_as == 'print':
        print("\n".join(summary_lines))

    if return_as == 'dict':
        return summary_dict
