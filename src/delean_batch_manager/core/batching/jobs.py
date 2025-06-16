# -*- coding: utf-8 -*-
"""
This module provides functions to manage OpenAI Batch jobs, including launching,
checking statuses, downloading results, and managing batch jobs.
It supports both serial and parallel execution of tasks.
It includes retry logic for transient errors and supports parallel execution
for efficiency.
"""


import os
import time
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Literal

import openai
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)
from tqdm.auto import tqdm

from ..utils.misc import mask_path
from .summary import save_batch_summary, save_general_summary
from .utils import save_batch_metadata

LAUNCH_MAX_WORKERS = 3    # Conservative number of workers to avoid rate limits
DOWNLOAD_MAX_WORKERS = 5  # Moderate number of workers for downloading results
CHECK_MAX_WORKERS = 8     # Aggressive number of workers for checking statuses


retry_on_transient_openai_errors = retry(
    retry=retry_if_exception_type((
        openai.RateLimitError,
        openai.InternalServerError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.UnprocessableEntityError
    )),
    wait=wait_exponential(min=2, max=256),
    stop=stop_after_attempt(10),
    reraise=True
)


#=============================================================================
# Batch Job Launching
#=============================================================================

@retry_on_transient_openai_errors
def launch_batch_job(
        client: openai.OpenAI | openai.AzureOpenAI,
        input_file: str | Path,
        endpoint: str = '/chat/completions',
        metadata_path: str | Path = None
    ):
    """
    Launch a batch job using the OpenAI Batch API.

    Args:
        client: OpenAI API client.
        input_file (str): Path to the input JSONL file.
        metadata_path (str): Path to save the batch metadata. If None, saves in the same folder as input_file.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if os.path.getsize(input_file) == 0:
        raise ValueError(f"Input file is empty: {input_file}")

    logging.info("Launching batch job for %s...", mask_path(input_file))

    batch_file = client.files.create(
        file=open(input_file, 'rb'),
        purpose='batch'
    )

    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint=endpoint,
        completion_window="24h"
    )

    if metadata_path is None:
        metadata_path = Path(input_file).parent / "batch_metadata.json"
    else:
        metadata_path = Path(metadata_path)
        if metadata_path.is_dir():
            metadata_path = metadata_path / "batch_metadata.json"
    save_batch_metadata(batch_job.model_dump(), metadata_path)

    logging.info(f"Batch job created with ID: {batch_job.id}")

    return batch_job.id


def launch_multiple_batch_jobs(
        client: openai.OpenAI | openai.AzureOpenAI,
        input_files: list[str | Path],
        endpoint: str = "/chat/completions"
    ):
    """
    Launch all batch jobs for the provided list of input files.

    Args:
        client: OpenAI API client.
        input_files (list[str | Path]): List of input.jsonl file paths.
        endpoint (str): API endpoint for the batch job. Defaults to "/v1/chat/completions".

    Returns:
        dict: Mapping from subfolder path to batch ID.
    """
    batch_id_map = {}

    if not input_files:
        raise ValueError("No input files provided. Please ensure 'input.jsonl' files were specified.")

    logging.info(f"Launching {len(input_files)} batch jobs")

    for input_file in tqdm(input_files, desc="Launching batch jobs"):
        input_file = Path(input_file)
        subfolder_path = str(input_file.parent)

        try:
            batch_id = launch_batch_job(client, str(input_file), endpoint=endpoint)
            batch_id_map[subfolder_path] = batch_id
        except Exception as e:
            logging.error(f"Failed to launch job for {mask_path(subfolder_path)}: {e}")
            continue

        # Sleep to avoid hitting rate limits
        time.sleep(3)

    logging.info("All batch jobs launched successfully.")
    return batch_id_map


def launch_multiple_batch_jobs_parallel(
        client: openai.OpenAI | openai.AzureOpenAI,
        input_files: list[str | Path],
        endpoint: str = "/chat/completions",
        max_workers: int = LAUNCH_MAX_WORKERS
    ):
    """
    Launch all batch jobs in parallel with controlled concurrency.

    Args:
        client: OpenAI API client.
        input_files (list[str | Path]): List of input.jsonl file paths.
        endpoint (str): API endpoint for batch jobs.
        max_workers (int): Maximum concurrent launches (recommended: 3-5).

    Returns:
        dict: Mapping from subfolder path to batch ID.
    """
    batch_id_map = {}

    if not input_files:
        logging.warning("Cannot launch any job because no input files were provided. Please ensure 'input.jsonl' files were specified.")
        return {}

    logging.info(f"Launching {len(input_files)} batch jobs in parallel (max_workers={max_workers})...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(
                launch_batch_job,
                client,
                str(Path(input_file)),
                endpoint,
                str(Path(input_file).parent / "batch_metadata.json")
            ): str(Path(input_file).parent)
            for input_file in input_files
        }

        # Collect results as they complete
        for future in as_completed(future_to_path):
            subfolder_path = future_to_path[future]
            try:
                batch_id = future.result()
                batch_id_map[subfolder_path] = batch_id
            except Exception as e:
                logging.error(f"Failed to launch job for {mask_path(subfolder_path)}: {e}")

    logging.info("All batch jobs launched successfully.")
    return batch_id_map


#=============================================================================
# Batch Status Checking
#=============================================================================

@retry_on_transient_openai_errors
def check_batch_status(
        client: openai.OpenAI | openai.AzureOpenAI,
        batch_id: str,
        verbose: int = 2
    ):
    """
    Check the status of a batch job.

    Args:
        client: OpenAI API client.
        batch_id (str): The ID of the batch job.
        verbose (int): Verbosity level for logging:
            0 - minimal output,
            1 - basic status info,
            2 - detailed status info.

    Returns:
        str: The status of the batch job.
    """
    if verbose > 0:
        logging.info(f"Checking status for batch job {batch_id}")
    batch_job = client.batches.retrieve(batch_id)
    status = batch_job.status
    if verbose > 1:
        if status == "failed":
            logging.error(f"Batch {batch_id} failed with error: {batch_job.errors}")
        elif status == "in_progress":
            completed = batch_job.request_counts.completed
            percentage = completed / batch_job.request_counts.total * 100
            logging.info(f"Batch {batch_id} is in progress, {completed} requests completed ({percentage:.2f}%)")
        elif status == "finalizing":
            logging.info(f"Batch {batch_id} is finalizing, waiting for the output file ID")
        elif status == "completed":
            logging.info(f"Batch {batch_id} has completed with output file ID: {batch_job.output_file_id}")
        else:
            logging.info(f"Batch {batch_id} is in status: {status}")
    return status


def check_multiple_batch_status(
        client: openai.OpenAI | openai.AzureOpenAI,
        batch_id_list: List[str],
        verbose: int = 2
     ):
    """
    Check the status of all batch jobs in the provided list.

    Args:
        client: OpenAI API client.
        batch_id_list (list): List of batch job IDs.
        verbose (int): Verbosity level for logging:
            0 - minimal output,
            1 - basic status info,
            2 - detailed status info.

    Returns:
        dict: A dictionary with batch IDs as keys and their statuses as values.
        dict: A summary of the statuses with counts and percentages.
    """
    if verbose > 0:
        logging.info(f"Checking status for {len(batch_id_list)} batch jobs...")
    statuses = {}
    failed_checks = []

    for batch_id in tqdm(batch_id_list, desc="Checking batch statuses"):
        try:
            status = check_batch_status(client, batch_id, verbose=verbose)
            statuses[batch_id] = status
            if verbose > 1:
                logging.info(f"Status for {batch_id}: {status}")
        except Exception as e:
            failed_checks.append((batch_id, str(e)))
            statuses[batch_id] = 'error'

    # Generate summary
    counter = Counter(statuses.values())
    total = sum(counter.values())
    summary = {
        status: {'count': count, 'percentage': (count / total) * 100}
        for status, count in counter.items()
    }

    if verbose > 1:
        logging.info(f"{'='*25}")
        logging.info("Batch Job Status Summary:")
        for status, data in summary.items():
            logging.info(f"- {status}: {data['count']} ({data['percentage']:.2f}%)")

        if failed_checks:
            logging.warning("Failed status checks:")
            for batch_id, error in failed_checks:
                logging.warning(" %s: %s", batch_id, error)

    if verbose > 0:
        logging.info(f"Status checks complete: {len(statuses)} total, {len(failed_checks)} errors.")

    return statuses, summary


def check_multiple_batch_status_parallel(
        client: openai.OpenAI | openai.AzureOpenAI,
        batch_id_list: List[str],
        max_workers: int = CHECK_MAX_WORKERS,
        verbose: int = 2
    ):
    """
    Check the status of all batch jobs in parallel.

    Args:
        client: OpenAI API client.
        batch_id_list (list): List of batch job IDs.
        max_workers (int): Maximum concurrent status checks.
        verbose (int): Verbosity level for logging:
            0 - minimal output,
            1 - basic status info,
            2 - detailed status info.

    Returns:
        dict: A dictionary with batch IDs as keys and their statuses as values.
        dict: A summary of the statuses with counts and percentages.
    """
    if verbose > 0:
        logging.info(f"Checking status for {len(batch_id_list)} batch jobs in parallel (max_workers={max_workers})...")

    statuses = {}
    failed_checks = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all status check jobs
        future_to_batch_id = {
            executor.submit(check_batch_status, client, batch_id, verbose): batch_id
            for batch_id in batch_id_list
        }

        # Collect results as they complete
        for future in as_completed(future_to_batch_id):
            batch_id = future_to_batch_id[future]
            try:
                status = future.result()
                statuses[batch_id] = status
                if verbose > 1:
                    logging.info(f"Status for {batch_id}: {status}")
            except Exception as e:
                failed_checks.append((batch_id, str(e)))
                statuses[batch_id] = 'error'

    # Generate summary
    counter = Counter(statuses.values())
    total = sum(counter.values())
    summary = {
        status: {'count': count, 'percentage': (count / total) * 100}
        for status, count in counter.items()
    }

    if verbose > 1:
        logging.info(f"{'='*25}")
        logging.info(f"Batch Job Status Summary:")
        for status, data in summary.items():
            logging.info(f"- {status}: {data['count']} ({data['percentage']:.2f}%)")

        if failed_checks:
            logging.warning("Failed status checks:")
            for batch_id, error in failed_checks:
                logging.warning(f"  - {batch_id}: {error}")

    if verbose > 0:
        logging.info(f"Status checks complete: {len(statuses)} total, {len(failed_checks)} errors.")

    return statuses, summary


#==============================================================================
# Batch Result Downloading
#==============================================================================

@retry_on_transient_openai_errors
def download_batch_result(
        client: openai.OpenAI | openai.AzureOpenAI,
        batch_id: str,
        output_folder: str,
        return_as: Literal['dict', 'print'] | None = None,
        save_summary_dict: bool = False
    ):
    """
    Download the results JSONL file of a completed batch job
    and save it as output.jsonl in the given folder.

    Args:
        client: OpenAI API client.
        batch_id (str): The ID of the batch job.
        output_folder (str): The folder where output.jsonl will be saved.
        return_as (str, optional): "dict", "print", or None. How to return summary.
        save_dict (bool): Whether to save the summary as a dict file.

    Returns:
        dict or str or None: Summary according to return_as.

    Raises:
        Exception: If the batch job has no output file.
            This can happen if the job is still in progress or has failed.
    """
    logging.info(f"Downloading results for batch job {batch_id}...")
    batch_job = client.batches.retrieve(batch_id)

    # Download the output file if exists
    output_file_id = batch_job.output_file_id
    if output_file_id:
        result = client.files.content(output_file_id).content
        result_path = os.path.join(output_folder, "output.jsonl")
        with open(result_path, 'wb') as file:
            file.write(result)
        logging.info(f"Results downloaded to {mask_path(result_path)}.")
    else:
        logging.warning(f"Batch job {batch_id} has no output file. "
                        "It may still be in progress or has failed.")

    # Download the error file if it exists
    error_file_id = batch_job.error_file_id
    if error_file_id:
        error_result = client.files.content(error_file_id).content
        error_path = os.path.join(output_folder, "errors.jsonl")
        with open(error_path, 'wb') as file:
            file.write(error_result)
        logging.info(f"Error results downloaded to {mask_path(error_path)}.")

    # Save the batch summary
    batch_metadata = batch_job.model_dump()
    metadata_path = os.path.join(output_folder, "batch_metadata.json")
    save_batch_metadata(batch_metadata, metadata_path)
    summary_path = os.path.join(output_folder, "summary.txt")
    summary = save_batch_summary(
        batch_metadata,
        result_path,
        summary_path=summary_path,
        return_as=return_as,
        save_dict=save_summary_dict
    )

    return summary


def download_multiple_batch_results(
        client: openai.OpenAI | openai.AzureOpenAI,
        batch_id_map: dict[str, str],
        base_folder: str,
        return_as: Literal['dict', 'print'] | None = None,
        save_summary_dict: bool = False,
    ):
    """
    Download the results JSONL files of all batch jobs into their corresponding subfolders.

    Args:
        client: OpenAI API client.
        batch_id_map (dict): Mapping from subfolder name to batch ID.
        base_folder (str): Path to the folder containing subfolders (where results will be saved).
        return_as (str, optional): "dict", "print", or None. How to return summary.
        save_summary_dict (bool): Whether to save the summary as a dict file.
    """
    logging.info(f"Downloading results for {len(batch_id_map)} jobs...")

    batch_summary_list = []
    failed_downloads = []
    for subfolder, batch_id in tqdm(batch_id_map.items(), desc="Downloading batch results"):
        try:
            summary = download_batch_result(
                client,
                batch_id,
                subfolder,
                return_as='dict',
                save_summary_dict=save_summary_dict
            )
            batch_summary_list.append(summary)
        except Exception as e:
            failed_downloads.append((subfolder, batch_id, str(e)))

    summary_path = os.path.join(base_folder, "summary.txt")
    general_summary = save_general_summary(
        batch_summary_list,
        summary_path=summary_path,
        return_as=return_as,
        save_dict=save_summary_dict
    )

    logging.info(f"Downloads complete: {len(batch_summary_list)} successful, {len(failed_downloads)} failed.")

    if failed_downloads:
        logging.warning("Failed downloads:")
        for subfolder, batch_id, error in failed_downloads:
            logging.warning(f"{mask_path(subfolder)} ({batch_id}): {error}")

    return general_summary


def download_multiple_batch_results_parallel(
        client: openai.OpenAI | openai.AzureOpenAI,
        batch_id_map: dict[str, str],
        base_folder: str,
        max_workers: int = DOWNLOAD_MAX_WORKERS,
        return_as: Literal['dict', 'print'] | None = None,
        save_summary_dict: bool = False
    ):
    """
    Download all batch results in parallel.

    Args:
        client: OpenAI API client.
        batch_id_map (dict): Mapping from subfolder name to batch ID.
        base_folder (str): Base folder containing subfolders.
        max_workers (int): Maximum concurrent downloads.
        return_as (str, optional): "dict", "print", or None. How to return summary.
        save_summary_dict (bool): Whether to save the summary as a dict file.

    Returns:
        dict or str or None: Summary according to return_as.
    """
    logging.info(f"Downloading results for {len(batch_id_map)} jobs in parallel (max_workers={max_workers})...")

    batch_summary_list = []
    failed_downloads = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download jobs
        future_to_info = {
            executor.submit(
                download_batch_result,
                client,
                batch_id,
                subfolder,
                'dict',
                save_summary_dict
            ): (subfolder, batch_id)
            for subfolder, batch_id in batch_id_map.items()
        }

        # Collect results as they complete
        for future in as_completed(future_to_info):
            subfolder, batch_id = future_to_info[future]
            try:
                summary = future.result()
                batch_summary_list.append(summary)
                logging.info(f"Downloaded results for {mask_path(subfolder)}")
            except Exception as e:
                failed_downloads.append((subfolder, batch_id, str(e)))

    # Generate overall summary
    summary_path = os.path.join(base_folder, "summary.txt")
    general_summary = save_general_summary(
        batch_summary_list,
        summary_path=summary_path,
        return_as=return_as,
        save_dict=save_summary_dict
    )

    logging.info(f"Downloads complete: {len(batch_summary_list)} successful, {len(failed_downloads)} failed.")

    if failed_downloads:
        logging.warning("Failed downloads:")
        for subfolder, batch_id, error in failed_downloads:
            logging.warning(f"{mask_path(subfolder)} ({batch_id}): {error}")

    return general_summary


#==============================================================================
# Batch Job Tracking and Looping
#==============================================================================

def track_and_download_batch(client, batch_id, output_folder, verbose=False):
    """
    Track the status of a batch job and download the results if completed.

    Args:
        client: OpenAI API client.
        batch_id (str): The ID of the batch job.
        output_folder (str): The folder to save the results.
        verbose (bool): If True, print detailed status information.

    Returns:
        str: The status of the batch job after checking.
    """
    status = check_batch_status(client, batch_id, verbose=verbose)
    if status == 'completed':
        summary_dict = download_batch_result(
            client, batch_id, output_folder,
            return_as='dict', save_summary_dict=False
        )
        return 'completed', summary_dict
    elif status == 'failed':
        return 'failed', {}
    else:
        return 'pending', {}


def track_and_download_multiple_batch_jobs_parallel_loop(
        client: openai.OpenAI | openai.AzureOpenAI,
        batch_id_map: dict[str, str],
        base_folder: str,
        check_interval: int = 1800,
        max_workers: int = 5
    ):
    """
    Track and download all batch jobs in parallel until all are completed or failed.

    Args:
        client: OpenAI API client.
        batch_id_map (dict): Mapping from subfolder name to batch ID.
        base_folder (str): Folder containing all subfolders for individual batch jobs.
        check_interval (int): Time interval (in seconds) to wait before checking the status again.
        max_workers (int): Number of parallel workers for tracking and downloading.
    """
    pending_map = dict(batch_id_map)
    completed_ids = set()
    failed_ids = set()
    batch_summary_list = []

    while pending_map:
        logging.info(f"Checking {len(pending_map)} pending jobs...")
        still_pending = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_subfolder = {
                executor.submit(
                    track_and_download_batch,
                    client,
                    batch_id,
                    os.path.join(base_folder, subfolder)
                ): subfolder
                for subfolder, batch_id in pending_map.items()
            }

            for future in as_completed(future_to_subfolder):
                subfolder = future_to_subfolder[future]
                batch_id = pending_map[subfolder]
                status, summary_dict = future.result()
                if status == 'completed':
                    completed_ids.add(batch_id)
                    batch_summary_list.append(summary_dict)
                elif status == 'failed':
                    failed_ids.add(batch_id)
                    logging.warning(f"Batch job {batch_id} failed")
                else:
                    still_pending[subfolder] = batch_id

        pending_map = still_pending

        if pending_map:
            logging.info(f"{len(completed_ids)} completed, {len(failed_ids)} failed, {len(pending_map)} still pending. Waiting {check_interval} seconds before retrying...")
            time.sleep(check_interval)

    # Save the general summary
    summary_path = os.path.join(base_folder, "summary.txt")
    save_general_summary(batch_summary_list, summary_path=summary_path)

    logging.info(f"All jobs done. Final summary: {len(completed_ids)} completed, {len(failed_ids)} failed.")

    return completed_ids, failed_ids


#==============================================================================
# Batch Job Management
#==============================================================================

@retry_on_transient_openai_errors
def list_batch_jobs(client, batch_id_map=None, status=None):
    """
    List all batch jobs and their statuses.

    Args:
        client: OpenAI API client.
        batch_id_map (dict): Optional mapping of subfolder names to batch IDs.
            If not provided, all jobs related to the client will be listed.
        status (str): Filter jobs by status (e.g., "completed", "failed", etc.).
            If None, list all jobs.
    Returns:
        list: A list of dictionaries containing metadata for each job matching the status.
    """
    if status is None:
        logging.info("Listing all batch jobs...")
    else:
        logging.info(f"Listing jobs with status: {status}...")
    jobs = client.batches.list()
    jobs_metadata = []
    count = 0
    for job in jobs:
        created_at = datetime.fromtimestamp(job.created_at).isoformat()
        if batch_id_map:
            for subfolder, bid in batch_id_map.items():
                subdomain = Path(subfolder).name
                if bid == job.id and (status is None or job.status == status):
                    logging.info(f"- Demand: {subdomain:<3}, Batch ID: {job.id}, Status: {job.status}, Created at: {created_at}")
                    jobs_metadata.append(job.model_dump())
                    count += 1
                    break
        else:
            if status is None or job.status == status:
                logging.info(f"- Batch ID: {job.id}, Status: {job.status}, Created at: {created_at}")
                jobs_metadata.append(job.model_dump())
                count += 1
    if status is None:
        logging.info(f"Found {len(jobs_metadata)} batch jobs.")
    else:
        if count == 0:
            logging.info(f"No jobs found matching status '{status}'.")
        else:
            logging.info(f"Found {count} jobs matching status '{status}'.")
    return jobs_metadata


@retry_on_transient_openai_errors
def cancel_batch_job(client, batch_id):
    """
    Cancel a batch job using its ID.

    Args:
        client: OpenAI API client.
        batch_id (str): The ID of the batch job to cancel.
    """
    logging.info(f"Cancelling batch job {batch_id}...")
    client.batches.cancel(batch_id)
    logging.info(f"Batch job {batch_id} cancelled.")


def cancel_multiple_batch_jobs(client, batch_id_list):
    """
    Cancel all batch jobs in the provided mapping.

    Args:
        client: OpenAI API client.
        batch_id_list (list): List of batch job IDs to cancel.
    """
    logging.info(f"Cancelling {len(batch_id_list)} batch jobs...")
    for batch_id in batch_id_list:
        cancel_batch_job(client, batch_id)
    logging.info("Cancelled all batch jobs.")
