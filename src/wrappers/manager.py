# -*- coding: utf-8 -*-

import os
import logging
import openai
from pathlib import Path
from typing import List, Literal


from .utils import (
    read_source_data,
    read_only_prompts_from_source_data
)
from ..batching.pricing import get_batch_api_pricing
from ..batching.files import (
    read_demand_levels_rubric_files,
    create_subdomain_batch_input_files,
    parse_subdomain_batch_output_files,
    save_parsed_subdomain_batch_output_files_results,
)
from ..batching.jobs import (
    launch_batch_job,
    launch_all_batch_jobs,
    launch_all_batch_jobs_parallel,
    check_batch_status,
    check_all_batch_status,
    check_all_batch_status_parallel,
    download_batch_result,
    download_all_batch_results,
    download_all_batch_results_parallel,
    track_and_download_all_batch_jobs_parallel_loop,
    list_batch_jobs,
    cancel_batch_job,
    cancel_all_batch_jobs
)
from ..batching.utils import (
    save_batch_id_map_to_file,
    load_batch_id_map_from_file
)
from ..utils import (
    assert_required_path,
    ensure_output_path,
    mask_path,
    resolve_n_jobs
)


class DeLeAnBatchManager:
    """A class to manage batch jobs for demand levels annotations using OpenAI Batch API."""

    def __init__(
        self,
        client: openai.OpenAI | openai.AzureOpenAI,
        base_folder: str | Path,
        source_data_path: str | Path,
        rubrics_folder: str | Path,
        max_completion_tokens: int = 1000,
        openai_model: str = "gpt-4o"
    ):
        self.client = client
        self._is_azure_client = isinstance(client, openai.AzureOpenAI)
        self.base_folder = base_folder
        self.source_data_path = source_data_path
        self.rubrics_folder = rubrics_folder
        self.max_tokens = max_completion_tokens
        self.openai_model = openai_model
        self.endpoint = "/chat/completions"
        self.batch_id_map = {}  # Maps subfolder -> batch_id
        self._finalized = set()

        self.__assert_required_paths()
        self.__ensure_output_paths()
        # Try to load existing batch ID map from file if it exists in the base folder
        self.load_batch_id_map(verbose=False)

    def __assert_required_paths(self):
        assert_required_path(self.source_data_path, description="Source data path")
        assert_required_path(self.rubrics_folder, description="Rubrics data path")

    def __ensure_output_paths(self):
        ensure_output_path(self.base_folder, description="Base folder")

    def set_openai_model(self, openai_model):
        self.openai_model = openai_model
        logging.info(f"OpenAI model set to {self.openai_model}")

    def get_batch_api_pricing(
            self,
            openai_model: str | list = "gpt-4o",
            max_completion_tokens: int | None = 1000,
            estimation: Literal['aprox', 'exact'] = 'aprox',
            n_jobs: int | None = None,
            verbose: bool = True
        ):
        """
        Estimate the cost of batch API calls for the given OpenAI model.

        Args:
            openai_model (str | list): OpenAI model name or list of model names.
            max_completion_tokens (int): Maximum number of tokens for completion.
            estimation (str): Type of estimation to perform ('aprox' or 'exact').
            n_jobs (int): Number of parallel jobs to run for pricing estimation.
            verbose (bool): Whether to log detailed information.

        Returns:
            dict: Estimated cost for each model.

        Raises:
            ValueError: If openai_model is not a string or a list of strings.
        """
        if isinstance(openai_model, str):
            openai_model = [openai_model]
        if not isinstance(openai_model, list):
            raise ValueError("openai_model must be a string or a list of strings.")
        prompts = read_only_prompts_from_source_data(self.source_data_path)
        rubrics = read_demand_levels_rubric_files(self.rubrics_folder)
        if verbose:
            logging.info("Calculating batch API pricing...")
        max_tokens = max_completion_tokens or self.max_tokens
        estimated_cost = {}
        for model in openai_model:
            estimated_cost[model] = get_batch_api_pricing(
                prompts=prompts,
                rubrics=rubrics,
                max_completion_tokens=max_tokens,
                openai_model=model,
                estimation=estimation,
                n_jobs=n_jobs
            )
            if verbose:
                logging.info(f"Estimated cost for model {model}: ${estimated_cost[model]:,.4f}")
        return estimated_cost

    def create_input_files(self, demand_level: str | list | None = None):
        """
        Create input files for batch jobs based on source data and rubrics.

        Args:
            demand_level (str | list | None): Demand level(s) to filter input files. 
                If None, creates input files for all demand levels.

        Raises:
            ValueError: If demand_level is not a string or list of strings.
            Exception: If no rubric files are found for the specified demand level(s).
        """
        max_bytes_per_file = 190 * 1024**2                                    # 190MB = 200MB (OpenAI Batch API limit) - 10MB (safety margin)
        max_lines_per_file = 100_000 if self._is_azure_client else 50_000     # Set 100K if using Azure OpenAI client, otherwise 50K         

        logging.info(f"Reading prompt data from {mask_path(self.source_data_path)}")
        prompt_data = read_source_data(self.source_data_path)

        logging.info(f"Reading rubrics from {mask_path(self.rubrics_folder)}")
        rubrics = read_demand_levels_rubric_files(self.rubrics_folder)
        if demand_level is not None:
            if isinstance(demand_level, str):
                demand_level = [demand_level]
            if not isinstance(demand_level, list):
                raise ValueError("demand_level must be a string or a list of strings.")
            rubrics = {k: v for k, v in rubrics.items() if k in demand_level}
            if not rubrics:
                raise Exception(f"No rubrics found for demand level(s): {demand_level}")

        logging.info(f"Creating batch files in {mask_path(self.base_folder)}")
        create_subdomain_batch_input_files(
            prompt_data=prompt_data,
            rubrics=rubrics,
            output_dir=self.base_folder,
            max_completion_tokens=self.max_tokens,
            openai_model=self.openai_model,
            body_url=self.endpoint,
            max_lines_per_file=max_lines_per_file,
            max_bytes_per_file=max_bytes_per_file
        )

    def parse_output_files(
            self,
            only_levels: bool = False,
            verbose: bool = True
        ) -> dict:
        """
        Parse all avilable output files in self.base folder from batch jobs.

        Args:
            only_levels (bool): If True, only extracts demand levels from the
                output files. If True, only returns the demand levels without
                additional information about finish reasons and model responses.
            verbose (bool): Whether to log detailed information per example
                when failing to extract annotations.

        Returns:
            dict: A dictionary containing the parsed outputs with prompt custom ids as keys.
            Each entry includes subdomain, finish_reason, model response and demand level.
        """
        logging.info(f"Parsing output files in {mask_path(self.base_folder)}")
        return parse_subdomain_batch_output_files(
            base_folder=self.base_folder,
            only_levels=only_levels,
            verbose=verbose
        )

    def parse_output_files_and_save_results(
            self,
            output_path: str | Path = None,
            file_type: Literal['jsonl', 'csv'] = 'jsonl',
            only_levels: bool = False,
            csv_format: Literal['long', 'wide'] = 'long',
            verbose: bool = True
        ):
        """
        Parse all available output files in self.base folder from batch jobs
        and save the results to a specified output path.

        Args:
            output_path (str | Path): Path to save the parsed output files results.
                Path can be a either a file or a directory. If a directory is provided, 
                the results will be saved as 'annotations.jsonl' or 'annotations.csv'
                in that directory.
            file_type (str): Type of file to save the results. Can be 'jsonl' or 'csv'.
            only_levels (bool): If True, only saves demand levels without
                additional information about finish reasons and model responses.
            csv_format (str): Format of the CSV file if file_type is 'csv'.
                Can be 'long' or 'wide'. "long" format will have one row per
                annotation, while "wide" format will have one row per prompt
                with all levels as columns. Note that "wide" will not include 
                finish reasons and completions, only levels, independently of 
                the --only-levels flag.
            verbose (bool): Whether to log detailed information per example
                when failing to extract annotations.
        """
        save_parsed_subdomain_batch_output_files_results(
            base_folder=self.base_folder,
            output_path=output_path,
            file_type=file_type,
            only_levels=only_levels,
            csv_format=csv_format,
            verbose=verbose
        )
        logging.info(f"Parsed output files results saved in {mask_path(output_path)}")

    def get_batch_input_files(self, demand_levels: str | list | None = None) -> List[Path]:
        """
        Get input files for specified demand levels.

        Args:
            demand_levels (str | list | None): Demand level(s) to filter input files. 
                If None, returns all input files.

        Returns:
            List[Path]: List of input file paths corresponding to the specified demand levels.

        Raises:
            ValueError: If demand_levels is not a string or list of strings.
            Exception: If no input files are found for the specified demand level(s).
        """
        input_files = Path(self.base_folder).glob("*/input.jsonl")

        if demand_levels is None:
            return list(input_files)
        if isinstance(demand_levels, str):
            demand_levels = [demand_levels]
        if not isinstance(demand_levels, list):
            raise ValueError("demand_levels must be a string or a list of strings.")

        filtered_files = []
        for demand in demand_levels:
            demand_files = [file for file in input_files if demand in file.name]
            if not demand_files:
                raise Exception(f"No input files found for demand level '{demand}'. Please ensure the demand level is correct and input files are created.")
            filtered_files.extend(demand_files)

        return filtered_files

    def get_batch_output_files(self, demand_levels: str | list | None = None) -> List[Path]:
        """
        Get output files for specified demand levels.

        Args:
            demand_levels (str | list | None): Demand level(s) to filter output files. 
                If None, returns all output files.

        Returns:
            List[Path]: List of output file paths corresponding to the specified demand levels.

        Raises:
            ValueError: If demand_levels is not a string or list of strings.
            Exception: If no output files are found for the specified demand level(s).
        """
        output_files = Path(self.base_folder).glob("*/output.jsonl")

        if demand_levels is None:
            return list(output_files)
        if isinstance(demand_levels, str):
            demand_levels = [demand_levels]
        if not isinstance(demand_levels, list):
            raise ValueError("demand_levels must be a string or a list of strings.")

        filtered_files = []
        for demand in demand_levels:
            demand_files = [file for file in output_files if demand in file.name]
            if not demand_files:
                raise Exception(f"No output files found for demand level '{demand}'. Please ensure the demand level is correct and output files are downloaded.")
            filtered_files.extend(demand_files)

        return filtered_files

    def launch_single(self, input_file: str | Path) -> str:
        """
        Launch a single batch job for the specified input file.

        Args:
            input_file (str | Path): Path to the input file.

        Returns:
            str: The batch ID of the launched job.
        """
        if not str(input_file).startswith(str(self.base_folder)):
            input_file = os.path.join(self.base_folder, input_file)
        subfolder = Path(input_file).parent
        batch_id = launch_batch_job(
            client=self.client,
            input_file=input_file,
            endpoint=self.endpoint
        )
        self.batch_id_map[subfolder] = batch_id
        self.save_batch_id_map_to_file()
        return batch_id

    def launch_all(self, parallel: bool = False):
        """
        Launch all batch jobs for input files in the base folder.

        Args:
            parallel (bool): Whether to launch jobs in parallel.
                Recommended for multiple large input files.

        Returns:
            dict: A dictionary mapping absolute subfolders to batch IDs.
        """
        if parallel:
            self.batch_id_map = launch_all_batch_jobs_parallel(
                client=self.client,
                base_folder=self.base_folder,
                endpoint=self.endpoint
            )
        else:
            self.batch_id_map = launch_all_batch_jobs(
                client=self.client,
                base_folder=self.base_folder,
                endpoint=self.endpoint
            )
        self.save_batch_id_map_to_file()
        return self.batch_id_map

    def launch(self, input_file: str | Path = None, parallel: bool = False) -> str | dict:
        """
        Launch batch jobs for specified input file 
        or all input files in the base folder.

        Args:
            input_file (str | Path | None): Path to a single input file.
                If None, launches all input files in the base folder.
            parallel (bool): Whether to launch jobs in parallel if launching all jobs.
                Recommended for multiple large input files.

        Returns:
            str | dict: Batch ID if a single input file is provided,
                otherwise a dictionary mapping absolute subfolders to batch IDs.
        """
        if input_file:
            return self.launch_single(input_file)
        else:
            return self.launch_all(parallel=parallel)

    def get_batch_ids(self, demand_level: str | list | None = None) -> List[str]:
        """
        Get batch IDs for specified demand levels.

        Args:
            demand_level (str | list | None): Demand level(s) to filter batch IDs. 
                If None, returns all batch IDs.

        Returns:
            List[str]: List of batch IDs corresponding to the specified demand levels.

        Raises:
            ValueError: If demand_level is not a string or list of strings.
            Exception: If no batch IDs are found for the specified demand level(s) 
                or if no batch IDs are stored in the manager.
        """
        if not self.batch_id_map:
            raise Exception("No batch IDs stored in manager. Launch jobs first.")

        if demand_level is None:
            return list(self.batch_id_map.values())
        if isinstance(demand_level, str):
            demand_level = [demand_level]
        if not isinstance(demand_level, list):
            raise ValueError("demand_level must be a string or a list of strings.")

        batch_ids = []
        for subfolder, batch_id in self.batch_id_map.items():
            demand_batch_ids = [batch_id for d in demand_level if d in subfolder.name]
            if not demand_batch_ids:
                raise Exception(f"No batch IDs found for demand level '{demand_level}'. Please ensure the demand level is correct and jobs are launched.")
            batch_ids.extend(demand_batch_ids)

        return batch_ids

    def check_single_status(self, batch_id: str, verbose: bool = True):
        """
        Check the status of a single batch job by its ID.

        Args:
            batch_id (str): The ID of the batch job to check.
            verbose (bool): Whether to print detailed status information.

        Returns:
            str: The status of the batch job ('pending', 'running', 'completed', 'failed', etc.).

        Raises:
            Exception: If the batch ID is not found in the manager.
        """
        if batch_id not in self.batch_id_map.values():
            raise Exception("Batch ID not found in manager. Launch job first.")
        status = check_batch_status(self.client, batch_id, verbose=verbose)
        if status in ['completed', 'failed']:
            self._finalized.add(batch_id)
        return status

    def check_all_status(self, parallel: bool = True, verbose: bool = True):
        """
        Check the status of all batch jobs stored in the manager.

        Args:
            parallel (bool): Whether to check the status of all jobs in parallel.
                 Recommended for multiple jobs.
            verbose (bool): Whether to print detailed status information.

        Returns:
            dict: A dictionary mapping batch IDs to their statuses.
            dict: A summary dictionary with counts of each status type.

        Raises:
            AssertionError: If no batch IDs are stored in the manager.
        """
        assert self.batch_id_map, "No batch IDs stored in manager. Launch jobs first."

        if parallel:
            statuses, summary = check_all_batch_status_parallel(
                self.client, list(self.batch_id_map.values()), verbose=verbose
            )
        else:
            statuses, summary = check_all_batch_status(
                self.client, list(self.batch_id_map.values()), verbose=verbose
            )

        for bid, status in statuses.items():
            if status in ['completed', 'failed']:
                self._finalized.add(bid)

        return statuses, summary

    def check(
            self,
            batch_id: str | None = None,
            parallel: bool = False,
            verbose: bool = True
        ):
        """
        Check the status of a single batch job or all batch jobs.

        Args:
            batch_id (str | None): The ID of the batch job to check.
                If None, checks the status of all batch jobs.
            parallel (bool): Whether to check the status of all jobs in
                parallel if cheking all jobs. Recommended for multiple jobs.
            verbose (bool): Whether to print detailed status information.

        Returns:
            str | dict: The status of the single batch job if batch_id is provided,
                otherwise a tuple containing a dictionary of batch IDs and their statuses,
                and a summary dictionary with counts of each status type.
        """
        if batch_id:
            return self.check_single_status(batch_id, verbose=verbose)
        else:
            return self.check_all_status(parallel=parallel, verbose=verbose)

    def download_single_result(
            self,
            batch_id: str,
            return_summary_dict: bool = False
        ):
        """
        Download results for a single batch job by its ID.
        If the job is not finalized, it will first check the status of the job.
        If the job is not 'completed' or 'failed', it will not download the
        results. If the job can be downloaded, it will override the existing 
        results in the subfolder.

        Args:
            batch_id (str): The ID of the batch job to download results for.
            return_summary_dict (bool): Whether to return a summary dictionary of the results.

        Returns:
            dict | None: A summary dictionary of the results if return_summary_dict is True,
                otherwise None.

        Raises:
            AssertionError: If the batch ID is not found in the manager.
        """
        assert batch_id in self.batch_id_map.values(), "Batch ID not found in manager. Launch job first."

        if batch_id not in self._finalized:
            logging.warning("Batch job does not appear as finalized. Checking status first.")
            status = self.check_single_status(batch_id, verbose=True)
            if batch_id not in self._finalized:
                logging.error(f"Batch job {batch_id} is not finalized. Status: {status}. Cannot download results until job is 'completed' or 'failed'.")
                return

        for subfolder, bid in self.batch_id_map.items():
            if bid == batch_id:
                
                summary_dict = download_batch_result(
                    client=self.client,
                    batch_id=batch_id,
                    output_folder=subfolder,
                    return_summary_dict=return_summary_dict
                )
                if return_summary_dict:
                    return summary_dict
                break

    def download_all_results(
            self,
            parallel: bool = False,
            return_summary_dict: bool = False
        ):
        """
        Download results for all finalized batch jobs stored in the manager.
        If no batch jobs are finalized, it will first check the status of all jobs.
        If no jobs are 'completed' or 'failed', it will not download results.
        If any batch jobs are finalized, it will download their results and
        return a summary dictionary. Note that this will override existing
        results in the subfolders.

        Args:
            parallel (bool): Whether to download results in parallel.
            Recommended for multiple dense jobs with large files to be downloaded.
            return_summary_dict (bool): Whether to return a summary dictionary of the results.

        Returns:
            dict | None: A summary dictionary of the results if return_summary_dict is True,
                otherwise None.

        Raises:
            AssertionError: If no batch IDs are stored in the manager or if no batch jobs appear as finalized.
        """
        assert self.batch_id_map, "No batch IDs stored in manager. Launch jobs first."

        if not self._finalized:
            logging.warning("No batch job appear as finalized. Checking status first.")
            self.check_all_status(verbose=True)
            if not self._finalized:
                logging.error("No batch jobs are finalized. Cannot download results until at least one job is 'completed' or 'failed'.")
                return

        finalized_batch_id_map = {k: v for k, v in self.batch_id_map.items() if v in self._finalized}
        if parallel:
            summary_dict = download_all_batch_results_parallel(
                client=self.client,
                batch_id_map=finalized_batch_id_map,
                base_folder=self.base_folder,
                return_summary_dict=return_summary_dict
            )
        else:
            summary_dict = download_all_batch_results(
                client=self.client,
                batch_id_map=finalized_batch_id_map,
                base_folder=self.base_folder,
                return_summary_dict=return_summary_dict
            )

        return summary_dict if summary_dict else None

    def download(
            self,
            batch_id: str | None = None,
            parallel: bool = False,
            return_summary_dict: bool = False
        ) -> dict | None:
        """
        Download results for a single batch job or all finalized batch jobs.
        Results will only be downloaded if the job(s) is/are finalized.
        If the result(s) can be downloaded, existing results in the
        subfolder(s) will be overridden.

        Args:
            batch_id (str | None): The ID of the batch job to download results
                for. If None, downloads results for all finalized batch jobs.
            parallel (bool): Whether to download results in parallel
                if downloading all jobs. Recommended for multiple dense jobs
                with large files to be downloaded
            return_summary_dict (bool): Whether to return a summary dictionary
                of the results.
        
        Returns:
            dict | None: A summary dictionary of the results if
                return_summary_dict is True, otherwise None.
        """
        if batch_id:
            return self.download_single_result(
                batch_id, return_summary_dict=return_summary_dict
            )
        else:
            return self.download_all_results(
                parallel=parallel, return_summary_dict=return_summary_dict
            )

    def track_and_download_loop(self, check_interval: int = 1800, n_jobs: int = 5):
        """
        Track and download all batch jobs in a loop until all jobs are finalized.

        Args:
            check_interval (int): Time in seconds to wait between checks for job completion.
            n_jobs (int): Number of parallel jobs to run for tracking and downloading.
                - If -1, uses all available CPU cores.
                - If 1, uses a single thread.
                - If >1, uses that many workers, capped at available cores.

        Returns:
            tuple: A tuple containing two dictionaries:
                - completed: Batch IDs of completed jobs.
                - failed: Batch IDs of failed jobs.

        Raises:
            AssertionError: If no batch IDs are stored in the manager.
            ValueError: If check_interval is not a positive integer.
        """
        assert self.batch_id_map, "No batch IDs stored in manager. Launch jobs first."

        if check_interval <= 0:
            raise ValueError("check_interval must be a positive integer.")

        max_workers = resolve_n_jobs(n_jobs, verbose=False)
        completed, failed = track_and_download_all_batch_jobs_parallel_loop(
            client=self.client,
            batch_id_map=self.batch_id_map,
            base_folder=self.base_folder,
            check_interval=check_interval,
            max_workers=max_workers
        )
        self._finalized.update(completed)
        self._finalized.update(failed)

        return completed, failed

    def save_batch_id_map_to_file(self):
        """
        Save the current batch ID map to a JSON file in the base folder.
        The file will be named "batch_id_map.json".
        """
        path = os.path.join(self.base_folder, "batch_id_map.json")
        save_batch_id_map_to_file(self.batch_id_map, path)

    def load_batch_id_map_from_file(self, verbose: bool = True):
        """
        Load the batch ID map from a JSON file in the base folder.
        The file should be named "batch_id_map.json".
        If the file does not exist, it will log an error and return.
        """
        path = os.path.join(self.base_folder, "batch_id_map.json")
        if not os.path.exists(path):
            if verbose:
                logging.error("Batch ID map file not found in base folder.")
            return
        self.batch_id_map = load_batch_id_map_from_file(path)

    def list(self, status: str | None = None):
        """
        List all batch jobs with their statuses.

        Args:
            status (str | None): If provided, filters jobs by the specified status.
                If None, lists all jobs regardless of status.

        Returns:
            List[dict]: A list of dictionaries containing job metadata.
        """
        return list_batch_jobs(self.client, status)

    def cancel_single_job(self, batch_id: str):
        """
        Cancel a single batch job by its ID.

        Args:
            batch_id (str): The ID of the batch job to cancel.

        Raises:
            AssertionError: If the batch ID is not found in the manager.
        """
        assert batch_id in self.batch_id_map.values(), "Batch ID not found in manager."
        cancel_batch_job(self.client, batch_id)
        self.batch_id_map.pop(batch_id, None)

    def cancel_all_jobs(self):
        """
        Cancel all batch jobs stored in the manager.
        This will clear the batch ID map after cancellation.

        Raises:
            AssertionError: If no batch IDs are stored in the manager.
        """
        assert self.batch_id_map, "No batch IDs stored in manager."
        cancel_all_batch_jobs(self.client, list(self.batch_id_map.values()))
        self.batch_id_map.clear()

    def cancel(self, batch_id: str | None = None):
        """
        Cancel a single batch job by its ID or all batch jobs if no ID is provided.

        Args:
            batch_id (str | None): The ID of the batch job to cancel. 
                If None, cancels all batch jobs.
        """
        if batch_id:
            self.cancel_single_job(batch_id)
        else:
            self.cancel_all_jobs()

    def run_full_pipeline(
            self,
            check_interval: int = 1800,
            parallel_launch: bool = False,
            max_workers_for_track: int = 5,
            demand_level: str | list | None = None
        ):
        """
        Run the full pipeline: create batch files, launch jobs, track and download results.
        This method will:
        1. Create batch files from the source data and specified rubrics.
           - If `demand_level` is provided, it will filter the input files accordingly.
           - If `demand_level` is None, it will create input files for all demand levels.
        2. Launch all batch jobs.
        3. Track and download all batch jobs in a loop until completion.

        Args:
            check_interval (int): Time in seconds to wait between checks for job completion.
            parallel_launch (bool): Whether to launch jobs in parallel.
            max_workers_for_track (int): Number of parallel jobs to run for tracking and downloading.
                - If -1, uses all available CPU cores.
                - If 1, uses a single thread.
                - If >1, uses that many workers, capped at available cores.
            demand_level (str | list | None): Demand level(s) to filter input files.
                If None, creates input files for all demand levels.
        """
        logging.info("Running full pipeline...")

        logging.info(f"Creating batch files from {mask_path(self.source_data_path)} and {mask_path(self.rubrics_folder)}...")
        self.create_input_files(demand_level=demand_level)
        logging.info(f"Batch files created in {mask_path(os.path.join(self.base_folder, '*', 'input.jsonl'))}.")

        logging.info("Launching all batch jobs...")
        self.launch(parallel=parallel_launch)

        logging.info("Tracking and downloading all batch jobs (this may take a while)...")
        self.track_and_download_loop(
            check_interval=check_interval, max_workers=max_workers_for_track
        )
        logging.info(f"All batch jobs tracked and downloaded in {mask_path(os.path.join(self.base_folder, '*', 'output.jsonl'))}.")

        logging.info("Pipeline complete.")
