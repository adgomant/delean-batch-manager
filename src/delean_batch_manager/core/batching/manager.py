# -*- coding: utf-8 -*-

import os
import logging
import openai
import polars as pl
from pathlib import Path
from typing import List, Literal

from ..batching.pricing import get_batch_api_pricing
from ..batching.files import create_subdomain_batch_input_files
from ..batching.parse import BatchOutputParser
from ..batching.jobs import (
    launch_batch_job,
    launch_multiple_batch_jobs,
    launch_multiple_batch_jobs_parallel,
    check_batch_status,
    check_multiple_batch_status,
    check_multiple_batch_status_parallel,
    download_batch_result,
    download_multiple_batch_results,
    download_multiple_batch_results_parallel,
    track_and_download_multiple_batch_jobs_parallel_loop,
    list_batch_jobs,
    cancel_batch_job,
    cancel_multiple_batch_jobs
)
from ..batching.utils import (
    save_batch_id_map_to_file,
    load_batch_id_map_from_file
)
from ..utils.rubrics import RubricsCatalog
from ..utils.datasource import (
    read_source_data,
    read_only_prompts_from_source_data,
)
from ..utils.clients import (
    create_azure_openai_client,
    create_openai_client
)
from ..utils.misc import (
    assert_required_path,
    ensure_output_path,
    mask_path,
    resolve_n_jobs,
    read_yaml
)


logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


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
        self.base_folder = Path(base_folder).resolve()
        self.source_data_path = Path(source_data_path).resolve()
        self.rubrics_folder = Path(rubrics_folder).resolve()
        self.rubrics_catalog = RubricsCatalog(self.rubrics_folder)
        self.max_tokens = max_completion_tokens
        self.openai_model = openai_model
        self.endpoint = "/chat/completions"

        self.batch_id_map = {}   # Maps subfolder -> batch_id
        self._completed = set()  # Set of completed batch IDs 
        self._input_files = []   # List of input files created for batch jobs
        self._output_files = []  # List of output files downloaded from batch jobs

        self.__assert_required_paths()
        self.__ensure_output_paths()

        # Try to load existing batch ID map from file if it exists in the base folder
        self.__try_load_all()

    @classmethod
    def from_config(cls, manager_file_path: str | Path):
        """
        Create a DeLeAnBatchManager instance from a configuration file.

        Args:
            manager_file_path (str | Path): Path to the configuration file.

        Returns:
            DeLeAnBatchManager: An instance of DeLeAnBatchManager initialized with the configuration.
        """
        config = read_yaml(manager_file_path)
        required = ['API', 'base_folder', 'source_data_file', 'rubrics_folder']
        if set(required) - set(config.keys()):
            raise ValueError(f"Configuration file must contain the following keys: {', '.join(required)}")
        if config['API'] == 'AzureOpenAI':
            client = create_azure_openai_client()
        else:
            client = create_openai_client()
        return cls(
            client=client,
            base_folder=config['base_folder'],
            source_data_path=config['source_data_file'],
            rubrics_folder=config['rubrics_folder'],
            max_completion_tokens=config.get('max_completion_tokens', 1000),
            openai_model=config.get('openai_model', 'gpt-4o')
        )

    def __try_load_all(self):
        try:
            self.load_batch_id_map_from_file(verbose=False)
            self._input_files = self.get_batch_files(which='input')
            self._output_files = self.get_batch_files(which='output')
            self._completed = set()
            if self._output_files:
                for outfile in self._output_files:
                    for subf, bid in self.batch_id_map.items():
                        if outfile.parent == Path(subf):
                            self._completed.add(bid)
        except:
            pass

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
            openai_model: str | list | None = "gpt-4o",
            max_completion_tokens: int | None = 1000,
            estimation: Literal['aprox', 'exact'] = 'aprox',
            n_jobs: int | None = None,
            verbose: bool = True
        ):
        """
        Estimate the cost of batch API calls for the given OpenAI model.

        Args:
            openai_model (str | list | None): OpenAI model name or list of model names.
                If None, uses the default model set in the manager.
            max_completion_tokens (int): Maximum number of tokens for completion.
                If None, uses the default max tokens set in the manager.
            estimation (str): Type of estimation to perform ('aprox' or 'exact').
                Default is 'aprox'.
            n_jobs (int): Number of parallel jobs to run for pricing estimation.
                - If -1, uses all available CPU cores.
                - If 1, uses a single thread.
                - If >1, uses that many workers, capped at available cores.
                - If None, fallbacks to serial execution.
            verbose (bool): Whether to log detailed information.

        Returns:
            dict: Estimated cost for each model.

        Raises:
            ValueError: If openai_model is not a string or a list of strings.
        """
        if openai_model is None:
            openai_model = self.openai_model
        if isinstance(openai_model, str):
            openai_model = [openai_model]
        if not isinstance(openai_model, list):
            raise ValueError("openai_model must be a string or a list of strings.")

        prompts = read_only_prompts_from_source_data(self.source_data_path)
        rubrics = self.rubrics_catalog.get_rubrics_dict()

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

    def create_input_files(self, demands: str | list | None = None):
        """
        Create input files for batch jobs based on source data and rubrics.

        Args:
            demands (str | list | None): Demand name(s) to create input files.
                If None, creates input files for all demand(s).

        Raises:
            ValueError: If demands is not a string or list of strings.
            Exception: If no rubric files are found for the specified demand name(s).
        """
        max_bytes_per_file = 190 * 1024**2                                    # 190MB = 200MB (OpenAI Batch API limit) - 10MB (safety margin)
        max_lines_per_file = 100_000 if self._is_azure_client else 50_000     # Set 100K if using Azure OpenAI client, otherwise 50K         

        prompt_data = read_source_data(self.source_data_path)
        rubrics = self.rubrics_catalog.get_rubrics_dict()

        if demands is not None:
            if isinstance(demands, str):
                demands = [demands]
            if not isinstance(demands, list):
                raise ValueError("demand_level must be a string or a list of strings.")
            rubrics = {k: v for k, v in rubrics.items() if k in demands}
            if not rubrics:
                raise Exception(f"No rubrics found for demand level(s): {demands}")

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

        self._input_files = self.get_batch_files(demands=demands, which='input')

    def parse_output_files(
            self,
            format: Literal['long', 'wide'] = 'long',
            return_as: Literal['jsonl', 'df'] = 'jsonl',
            only_levels: bool = False,
            only_succeed: bool = False,
            only_failed: bool = False,
            finish_reason: str | None = None,
            include_prompts: bool = False,
            verbose: bool = True,
            output_path: str | Path | None = None,
            file_type: Literal['jsonl', 'csv', 'parquet'] = 'jsonl',
            prefix: str = "",
            split_by_demand: bool = False
        ) -> dict | pl.DataFrame:
        """
        Parse all available output files in self.base folder from batch jobs
        and save the results to a specified output path if specified.

        Args:
            format (str): Format of the output. Can be "long" or "wide".
            return_as (str): Format to return the parsed output. Can be "jsonl" or "df".
            only_levels (bool): If True, only saves demand levels without
                additional information about finish reasons and model responses.
            only_succeed (bool): If True, only includes successful annotations in the output.
            only_failed (bool): If True, only includes failed annotations in the output.
            finish_reason (str | None): If specified, filters results by the finish reason.
            include_prompts (bool): If True, includes the original prompts in the output.
            verbose (bool): Whether to log detailed information per example
                when failing to extract annotations.
            output_path (str | Path | None): Path to save the parsed output files results.
                Path can be a either a file path or a directory. If a directory is provided, 
                the results will be saved as 'annotations.jsonl' or 'annotations.csv'
                in that directory. If not provided, the results will not be saved.
            file_type (str): Type of file to save the results. Can be 'jsonl', 'csv', or 'parquet'.
                Only used if `output_path` is provided and is a directory. Default is 'jsonl'.
            prefix (str): Prefix for output files. Only used if `output_path` is provided and
                is a directory.
            split_by_demand (bool): If True, splits the output files by demand level. Note that this
                will only work if the `output_path` is a directory.

        Returns:
            dict or pd.DataFrame: The parsed outputs as a dictionary (for jsonl) or DataFrame (for csv/parquet).
        """
        assert self._output_files, "No output files found. Please download batch jobs results first."

        if return_as not in ['jsonl', 'df']:
            raise ValueError("return_as must be either 'jsonl' or 'df'.")

        source_prompts = None
        if include_prompts:
            source_prompts = read_source_data(self.source_data_path, as_map=True)

        parser = BatchOutputParser(
            only_levels=only_levels,
            only_succeed=only_succeed,
            only_failed=only_failed,
            finish_reason=finish_reason,
            source_prompts=source_prompts,
            verbose=verbose,
            format=format
        )
        parser.parse(self._output_files)

        if output_path:
            output_path = Path(output_path).resolve()
            if output_path.suffix:
                file_type = output_path.suffix.lstrip('.')
            write_kwargs = {
                'path': output_path,
                'prefix': prefix,
                'split_by_demand': split_by_demand
            }
            if file_type == 'jsonl':
                parser.write_jsonl(**write_kwargs)
            elif file_type == 'csv':
                parser.write_csv(**write_kwargs)
            elif file_type == 'parquet':
                parser.write_parquet(**write_kwargs)
            else:
                raise Exception(f"Unsupported file_type: {file_type}")

        return parser.to_jsonl() if return_as == 'jsonl' else parser.to_df()

    def get_batch_files(
            self,
            demands: str | list | None = None,
            which: Literal['input', 'output'] = 'input'
        ) -> List[Path]:
        """
        Get input or output files for specified demand levels.

        Args:
            demands (str | list | None): Demand name(s) to filter output files. 
                If None, returns all output files.
            which (str): Type of files to return, either 'input' or 'output'.

        Returns:
            List[Path]: List of output file paths corresponding to the specified demand name(s).

        Raises:
            ValueError: If demands is not a string or list of strings.
            Exception: If no output files are found for the specified demand name(s).
        """
        files = Path(self.base_folder).rglob(f"{which}.jsonl")
        if not files:
            msg = "create input" if which == 'input' else "download output"
            raise Exception(f"No {which} files found in the base folder. "
                            f"Please {msg} files first.")

        if demands is None:
            return list(files)
        if isinstance(demands, str):
            demands = [demands]
        if not isinstance(demands, list):
            raise ValueError("demands must be a string or a list of strings.")

        filtered_files = []
        for demand in demands:
            demand_files = [file for file in files if demand in file.name]
            if not demand_files:
                raise Exception(f"No {which} files found for demand '{demand}'. "
                                "Please ensure the demand name is correct and output files are downloaded.")
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
        assert input_file in self._input_files, f"Input file not found. Please create input files first."
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

        Raises:
            AssertionError: If no input files are found in the base folder.
        """
        assert self._input_files, "No input files found. Please create input files first."
        input_files = self.get_batch_files(which='input')
        launch_func = (
            launch_multiple_batch_jobs_parallel if parallel
            else launch_multiple_batch_jobs
        )

        self.batch_id_map = launch_func(
            client=self.client,
            input_files=input_files,
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

    def get_batch_ids(self, demands: str | list | None = None) -> List[str]:
        """
        Get batch IDs for specified demand levels.

        Args:
            demands (str | list | None): Demand names(s) to filter batch IDs.
                If None, returns all batch IDs.

        Returns:
            List[str]: List of batch IDs corresponding to the specified demand(s).

        Raises:
            AssertionError: If no batch IDs are stored in the manager.
            ValueError: If demands is not a string or list of strings.
            Exception: If no batch IDs are found for the specified demand(s).
        """        
        assert self.batch_id_map, "No batch IDs stored in manager. Launch jobs first."

        if demands is None:
            return list(self.batch_id_map.values())
        if isinstance(demands, str):
            demands = [demands]
        if not isinstance(demands, list):
            raise ValueError("demand_level must be a string or a list of strings.")

        batch_ids = []
        for subfolder, batch_id in self.batch_id_map.items():
            demand_batch_ids = [batch_id for d in demands if d in subfolder.name]
            if not demand_batch_ids:
                raise Exception(f"No batch IDs found for demand(s) '{demands}'. Please ensure the demands names are correct and jobs are launched.")
            batch_ids.extend(demand_batch_ids)

        return batch_ids

    def check_single_status(self, batch_id: str, verbose: int = 2):
        """
        Check the status of a single batch job by its ID.

        Args:
            batch_id (str): The ID of the batch job to check.
            verbose (int): Verbosity level for logging:
                0 - minimal output,
                1 - basic status info,
                2 - detailed status info.

        Returns:
            str: The status of the batch job ('pending', 'running', 'completed', 'failed', etc.).

        Raises:
            AssertionError: If the batch ID is not found in the manager.
        """
        assert batch_id in self.batch_id_map.values(), "Batch ID not found in manager. Launch job first."
        status = check_batch_status(self.client, batch_id, verbose=verbose)
        if status == 'completed':
            self._completed.add(batch_id)
        return status

    def check_all_status(self, parallel: bool = True, verbose: int = 2):
        """
        Check the status of all batch jobs stored in the manager.

        Args:
            parallel (bool): Whether to check the status of all jobs in parallel.
                 Recommended for multiple jobs.
            verbose (int): Verbosity level for logging:
                0 - minimal output,
                1 - basic status info,
                2 - detailed status info.

        Returns:
            dict: A dictionary mapping batch IDs to their statuses.
            dict: A summary dictionary with counts of each status type.

        Raises:
            AssertionError: If no batch IDs are stored in the manager.
        """
        assert self.batch_id_map, "No batch IDs stored in manager. Launch jobs first."

        check_func = (
            check_multiple_batch_status_parallel if parallel 
            else check_multiple_batch_status
        )

        statuses, summary = check_func(
            self.client, list(self.batch_id_map.values()), verbose=verbose
        )

        for bid, status in statuses.items():
            if status == 'completed':
                self._completed.add(bid)

        return statuses, summary

    def check(
            self,
            batch_id: str | None = None,
            parallel: bool = False,
            verbose: int = 2
        ):
        """
        Check the status of a single batch job or all batch jobs.

        Args:
            batch_id (str | None): The ID of the batch job to check.
                If None, checks the status of all batch jobs.
            parallel (bool): Whether to check the status of all jobs in
                parallel if cheking all jobs. Recommended for multiple jobs.
            verbose (int): Verbosity level for logging:
                0 - minimal output,
                1 - basic status info,
                2 - detailed status info.

        Returns:
            str | dict: The status of the single batch job if batch_id is provided,
                otherwise a tuple containing a dictionary of batch IDs and their statuses,
                and a summary dictionary with counts of each status type.

        Raises:
            AssertionError: If no batch IDs are stored in the manager 
                or if the batch ID is not found in the manager.
            ValueError: If batch_id is not a string.
        """
        if batch_id:
            if not isinstance(batch_id, str):
                raise ValueError("batch_id must be a string.")
            return self.check_single_status(batch_id, verbose=verbose)
        else:
            return self.check_all_status(parallel=parallel, verbose=verbose)

    def download_single_result(
            self,
            batch_id: str,
            return_as: Literal['dict', 'print'] | None = 'dict',
            save_summary_dict: bool = False
        ):
        """
        Download results for a single batch job by its ID.
        If the job is not completed, it will first check the status of the job.
        If the job is not 'completed', it will not download the
        results. If the job can be downloaded, it will override the existing 
        results in the subfolder.

        Args:
            batch_id (str): The ID of the batch job to download results for.
            return_as (str, None): Whether to return the summary as a dictionary or print it.
                Can be 'dict' or 'print'. If None, no summary is returned.
            save_summary_dict (bool): Whether to save the summary as a dictionary apart from text.

        Returns:
            dict: A summary dictionary of the results if return_summary_dict is True
            None: If return_summary_dict is False or if the job is not completed.

        Raises:
            AssertionError: If the batch ID is not found in the manager.
        """
        assert batch_id in self.batch_id_map.values(), "Batch ID not found in manager. Launch job first."

        if batch_id not in self._completed:
            logging.warning("Batch job does not appear as completed. Checking status first.")
            status = self.check_single_status(batch_id, verbose=0)
            if batch_id not in self._completed:
                logging.error(f"Batch job {batch_id} is not completed. Status: {status}. Cannot download results.")
                return

        summary_dict = {}
        for subfolder, bid in self.batch_id_map.items():
            if bid == batch_id:
                summary_dict = download_batch_result(
                    client=self.client,
                    batch_id=batch_id,
                    output_folder=subfolder,
                    return_as=return_as,
                    save_summary_dict=save_summary_dict
                )
                break

        self._output_files = self.get_batch_files(which='output')
        return summary_dict

    def download_all_results(
            self,
            parallel: bool = False,
            return_as: Literal['dict', 'print'] | None = 'dict',
            save_summary_dict: bool = False
        ):
        """
        Download results for all completed batch jobs stored in the manager.
        If no batch jobs are completed, it will first check the status of all jobs.
        If no jobs are completed, it will not download results.
        If any batch jobs are completed, it will download their results and
        return a summary dictionary. Note that this will override existing
        results in the subfolders.

        Args:
            parallel (bool): Whether to download results in parallel.
                Recommended for multiple dense jobs with large files to be downloaded.
            return_as (str, None): Whether to return the general summary as a dictionary or print it.
                Can be 'dict' or 'print'. If None, no summary is returned.
            save_summary_dict (bool): Whether to save the general summary as a dictionary apart from text.

        Returns:
            dict: A summary dictionary of the results if return_summary_dict is True.
            None: If return_summary_dict is False or if no batch jobs are completed.

        Raises:
            AssertionError: If no batch IDs are stored in the manager.
        """
        assert self.batch_id_map, "No batch IDs stored in manager. Launch jobs first."

        if not self._completed:
            logging.warning("No batch job appear as completed. Checking status first.")
            self.check_all_status(verbose=0)
            if not self._completed:
                logging.error("No batch jobs are completed. Cannot download results until at least one job is 'completed' or 'failed'.")
                return

        finalized_batch_id_map = {
            k: v for k, v in self.batch_id_map.items()
            if v in self._completed
        }
        download_func = (
            download_multiple_batch_results_parallel if parallel 
            else download_multiple_batch_results
        )
        summary_dict = download_func(
            client=self.client,
            batch_id_map=finalized_batch_id_map,
            base_folder=self.base_folder,
            return_as=return_as,
            save_summary_dict=save_summary_dict
        )

        self._output_files = self.get_batch_files(which='output')
        return summary_dict

    def download(
            self,
            batch_id: str | None = None,
            parallel: bool = False,
            return_as: Literal['dict', 'print'] | None = 'dict',
            save_summary_dict: bool = False
        ) -> dict | None:
        """
        Download results for a single batch job or all completed batch jobs.
        Results will only be downloaded if the job(s) is/are completed.
        If the result(s) can be downloaded, existing results in the
        subfolder(s) will be overridden.

        Args:
            batch_id (str | None): The ID of the batch job to download results
                for. If None, downloads results for all completed batch jobs.
            parallel (bool): Whether to download results in parallel
                if downloading all jobs. Recommended for multiple dense jobs
                with large files to be downloaded
            return_as (str, None): Whether to return the summary as a dictionary or print it.
                Can be 'dict' or 'print'. If None, no summary is returned.
            save_summary_dict (bool): Whether to save the summary/ies as dictionary apart from text.

        Returns:
            dict: A summary dictionary of the results if return_summary_dict is True.
            None: If return_summary_dict is False or if the job(s) is/are not completed.

        Raises:
            AssertionError: If no batch IDs are stored in the manager
                or if the batch ID is not found in the manager.
            ValueError: If batch_id is not a string.
        """
        if batch_id:
            if not isinstance(batch_id, str):
                raise ValueError("batch_id must be a string.")
            return self.download_single_result(
                batch_id, return_as=return_as, save_summary_dict=save_summary_dict
            )
        else:
            return self.download_all_results(
                parallel=parallel, return_as=return_as, save_summary_dict=save_summary_dict
            )

    def track_and_download_loop(self, check_interval: int = 1800, n_jobs: int = 5):
        """
        Track and download all batch jobs in a loop until all jobs are
        finalized. Note that this method will only download results for
        jobs that are completed

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
        completed, failed = track_and_download_multiple_batch_jobs_parallel_loop(
            client=self.client,
            batch_id_map=self.batch_id_map,
            base_folder=self.base_folder,
            check_interval=check_interval,
            max_workers=max_workers
        )
        
        self._completed.update(completed)
        self._output_files = self.get_batch_files(which='output')

        return completed, failed

    def save_batch_id_map_to_file(self):
        """
        Save the current batch ID map to a JSON file in the base folder.
        The file will be named "batch_id_map.json".

        Raises:
            AssertionError: If no batch IDs are stored in the manager.
        """
        assert self.batch_id_map, "No batch IDs to save. Launch jobs first."
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

    def list_jobs(self, status: str | None = None):
        """
        List all batch jobs with their statuses.

        Args:
            status (str | None): If provided, filters jobs by the specified status.
                If None, lists all jobs regardless of status.

        Returns:
            List[dict]: A list of dictionaries containing job metadata.

        Raises:
            AssertionError: If no batch IDs are stored in the manager.
            ValueError: If status is not a string.
        """
        assert self.batch_id_map, "No batch IDs stored in manager. Launch jobs first."
        if status is not None and not isinstance(status, str):
            raise ValueError("status must be a string or None.")
        return list_batch_jobs(self.client, self.batch_id_map, status)

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
        cancel_multiple_batch_jobs(self.client, list(self.batch_id_map.values()))
        self.batch_id_map.clear()

    def cancel(self, batch_id: str | None = None):
        """
        Cancel a single batch job by its ID or all batch jobs if no ID is provided.

        Args:
            batch_id (str | None): The ID of the batch job to cancel. 
                If None, cancels all batch jobs.

        Raises:
            AssertionError: If no batch IDs are stored in the manager 
                or if the batch ID is not found in the manager.
            ValueError: If batch_id is not a string.
        """
        if batch_id:
            if not isinstance(batch_id, str):
                raise ValueError("batch_id must be a string.")
            self.cancel_single_job(batch_id)
        else:
            self.cancel_all_jobs()

    def run_full_pipeline(
            self,
            check_interval: int = 1800,
            parallel_launch: bool = False,
            max_workers_for_track: int = 5,
            demands: str | list | None = None
        ):
        """
        Run the full pipeline: create batch files, launch jobs, track and download results.
        This method will:
        1. Create batch files from the source data and specified rubrics.
           - If `demands` is provided, it will filter the input files accordingly.
           - If `demands` is None, it will create input files for all demand levels.
        2. Launch all batch jobs.
        3. Track and download all batch jobs in a loop until completion.

        Args:
            check_interval (int): Time in seconds to wait between checks for job completion.
            parallel_launch (bool): Whether to launch jobs in parallel.
            max_workers_for_track (int): Number of parallel jobs to run for tracking and downloading.
                - If -1, uses all available CPU cores.
                - If 1, uses a single thread.
                - If >1, uses that many workers, capped at available cores.
            demands (str | list | None): Demand name(s) to create input file(s).
                If None, creates input files for all demands.
        """
        logging.info("Running full pipeline...")

        logging.info(f"Creating batch files from {mask_path(self.source_data_path)} and {mask_path(self.rubrics_folder)}...")
        self.create_input_files(demands=demands)
        logging.info(f"Batch files created in {mask_path(Path(self.base_folder) / '*' / 'input.jsonl')}.")

        logging.info("Launching all batch jobs...")
        self.launch(parallel=parallel_launch)

        logging.info("Tracking and downloading all batch jobs (this may take a while)...")
        self.track_and_download_loop(
            check_interval=check_interval, n_jobs=max_workers_for_track
        )
        logging.info(f"All batch jobs tracked and downloaded in {mask_path(Path(self.base_folder) / '*' / 'output.jsonl')}.")

        logging.info("Pipeline complete.")
