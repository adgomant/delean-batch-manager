# -*- coding: utf-8 -*-

import os
import openai
import logging
import time
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential,
    stop_after_attempt
)

from .summary import save_batch_summary, save_general_summary
from .utils import save_batch_metadata
from ..utils import mask_path


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Decorator for retrying transient OpenAI errors ---

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


@retry_on_transient_openai_errors
def launch_batch_job(client, input_file, endpoint='/chat/completions', metadata_path=None):
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
    
    logging.info(f"Launching batch job for {mask_path(input_file)}...")
    batch_file = client.files.create(
        file=open(input_file, 'rb'),
        purpose='batch'
    )
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint=endpoint,
        completion_window="24h"
    )
    batch_id = batch_job.id
    if metadata_path is None:
        metadata_path = Path(input_file).parent / "batch_metadata.json"
    save_batch_metadata(batch_job, metadata_path)
    logging.info(f"Batch job created with ID: {batch_id}")
    return batch_id


def launch_all_batch_jobs(client, base_folder, endpoint="/chat/completions"):
    """
    Launch all batch jobs in subfolders of the input_folder.
    Each subfolder must contain a file named 'input.jsonl'.

    Args:
        client: OpenAI API client.
        base_folder (str): Path to the folder containing subfolders with input.jsonl files.
        endpoint (str): API endpoint for the batch job. Defaults to "/v1/chat/completions".

    Returns:
        dict: Mapping from folder name to batch ID.
    """
    batch_ids = {}

    logging.info(f"Scanning subfolders in {mask_path(base_folder)}")

    for subfolder in sorted(os.listdir(base_folder)):
        subfolder_path = os.path.join(base_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue  # skip non-directories

        input_file = os.path.join(subfolder_path, "input.jsonl")
        if not os.path.exists(input_file):
            logging.warning(f"Skipping {mask_path(subfolder_path)}: input.jsonl not found.")
            continue

        metadata_path = os.path.join(subfolder_path, "batch_metadata.json")
        batch_id = launch_batch_job(client, input_file, endpoint=endpoint, metadata_path=metadata_path)
        batch_ids[subfolder_path] = batch_id

        time.sleep(3)

    logging.info(f"Launched {len(batch_ids)} batch jobs.")
    return batch_ids


### --- Status Checking --- ###

@retry_on_transient_openai_errors
def check_batch_status(client, batch_id, verbose=True):
    """
    Check the status of a batch job.

    Args:
        client: OpenAI API client.
        batch_id (str): The ID of the batch job.
        verbose (bool): If True, print detailed status information.

    Returns:
        str: The status of the batch job.
    """
    logging.info(f"Checking status for batch job {batch_id}")
    batch_job = client.batches.retrieve(batch_id)
    status = batch_job.status
    if verbose:
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


def check_all_batch_status(client, batch_id_list, verbose=True):
    """
    Check the status of all batch jobs in the provided list.

    Args:
        client: OpenAI API client.
        batch_id_list (list): List of batch job IDs.
        verbose (bool): If True, print detailed status information per batch.

    Returns:
        dict: A dictionary with batch IDs as keys and their statuses as values.
        dict: A summary of the statuses with counts and percentages.
    """
    logging.info(f"Checking status for {len(batch_id_list)} batch jobs...")
    statuses = {}
    for batch_id in batch_id_list:
        status = check_batch_status(client, batch_id, verbose=verbose)
        statuses[batch_id] = status

    # Generate summary
    counter = Counter(statuses.values())
    total = sum(counter.values())
    summary = {
        status: {'count': count, 'percentage': (count / total) * 100}
        for status, count in counter.items()
    }

    if verbose:
        logging.info(f"{'='*25}\nBatch Job Status Summary:")
        for status, data in summary.items():
            logging.info(f"- {status}: {data['count']} ({data['percentage']:.2f}%)")

    return statuses, summary


### --- Output Downloading --- ###


@retry_on_transient_openai_errors
def download_batch_result(client, batch_id, output_folder, return_summary_dict=False):
    """
    Download the results JSONL file of a completed batch job and save it as output.jsonl in the given folder.

    Args:
        client: OpenAI API client.
        batch_id (str): The ID of the batch job.
        output_folder (str): The folder where output.jsonl will be saved.
    """
    logging.info(f"Downloading results for batch job {batch_id}...")
    batch_job = client.batches.retrieve(batch_id)
    output_file_id = batch_job.output_file_id
    result = client.files.content(output_file_id).content

    result_path = os.path.join(output_folder, "output.jsonl")
    with open(result_path, 'wb') as file:
        file.write(result)

    # Save the batch summary
    batch_metadata = batch_job
    metadata_path = os.path.join(output_folder, "batch_metadata.json")
    save_batch_metadata(batch_metadata, metadata_path)
    summary_path = os.path.join(output_folder, "summary.txt")
    summary_dict = save_batch_summary(batch_metadata.model_dump(), result_path, summary_path=summary_path, return_dict=return_summary_dict)

    logging.info(f"Results downloaded to {mask_path(result_path)}.")

    if summary_dict:
        return summary_dict


def download_all_batch_results(client, batch_id_map, base_folder, return_summary_dict=False):
    """
    Download the results JSONL files of all batch jobs into their corresponding subfolders.

    Args:
        client: OpenAI API client.
        batch_id_map (dict): Mapping from subfolder name to batch ID.
        base_folder (str): Path to the folder containing subfolders (where results will be saved).
    """
    logging.info(f"Downloading results for {len(batch_id_map)} jobs...")

    batch_summary_list = []
    for subfolder, batch_id in batch_id_map.items():
        summary_dict = download_batch_result(client, batch_id, subfolder, return_summary_dict=True)
        batch_summary_list.append(summary_dict)

    summary_path = os.path.join(base_folder, "summary.txt")
    summary_dict = save_general_summary(batch_summary_list, summary_path=summary_path, return_dict=return_summary_dict)

    logging.info("All downloads complete.")

    if return_summary_dict:
        return summary_dict


### --- Tracking and Looping --- ###

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
        summary_dict = download_batch_result(client, batch_id, output_folder, return_summary_dict=True)
        return 'completed', summary_dict
    elif status == 'failed':
        return 'failed', {}
    else:
        return 'pending', {}


def track_and_download_all_batch_jobs_parallel_loop(client, batch_id_map, base_folder, check_interval=1800, max_workers=5):
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


### --- Job Listing --- ###

@retry_on_transient_openai_errors
def list_batch_jobs(client, status=None):
    """
    List all batch jobs and their statuses.

    Args:
        client: OpenAI API client.
        status (str): Filter jobs by status (e.g., "completed", "failed", etc.). If None, list all jobs.

    Returns:
        list: A list of dictionaries containing metadata for each job matching the status.
    """
    logging.info(f"Listing jobs with status: {status}...")
    jobs = client.batches.list()
    jobs_metadata = []
    count = 0
    for job in jobs:
        if status is None or job.status == status:
            logging.info(f"- Batch ID: {job.id}, Status: {job.status}, Created at: {job.created_at}")
            jobs_metadata.append(job.model_dump())
            count += 1
    logging.info(f"Found {count} jobs matching status '{status}'.")
    return jobs_metadata


### --- Job cancellation --- ###

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


def cancel_all_batch_jobs(client, batch_id_list):
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
