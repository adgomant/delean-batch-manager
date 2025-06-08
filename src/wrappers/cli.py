# -*- coding: utf-8 -*-

import os
import click
import logging
import yaml
from pathlib import Path

from ..batching.pricing import get_batch_api_pricing
from ..batching.files import (
    read_demand_levels_rubric_files,
    create_subdomain_batch_input_files,
    parse_subdomain_batch_output_files
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
    load_batch_id_map_from_file,
)
from .utils import (
    create_openai_client,
    create_azure_openai_client,
    read_source_data,
    read_only_prompts_from_source_data,
    check_source_data_jsonl_keys,
    check_source_data_csv_columns,
)
from ..utils import resolve_n_jobs


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def validate_positive_integer_callback(ctx, param, value):
    """Validate that the provided value is a positive integer."""
    if value <= 0:
        raise click.BadParameter("Value must be a positive integer.")
    return value


@click.group()
@click.option(
    '--run-name', required=True, type=str, 
    help='Name of the run to work with.'
)
@click.pass_context
def cli(ctx, run_name):
    """
    CLI tool for managing OpenAI Batch Jobs to obtain Demand Levels Annotations.
    Ensure you have the API keys set in your environment variables.
    """
    ctx.ensure_object(dict)
    ctx.obj['run_name'] = run_name
    if ctx.command.name != 'setup':
        manager_file_path = os.path.abspath(f"./runs/batch_runs_{run_name}/ManagerFile.yaml")
        if not os.path.exists(manager_file_path):
            logging.error(f"Manager files does not exist for run '{run_name}'. "
                          + "Please execute 'setup' command first.")
            raise SystemExit(1)
        else:
            with open(manager_file_path, 'r') as f:
                manager_file = yaml.safe_load(f)

            for key, value in manager_file.items():
                ctx.obj[key] = value

            if ctx.obj['API'] == 'AzureOpenAI':
                ctx.obj['client'] = create_azure_openai_client()
            else:
                ctx.obj['client'] = create_openai_client()

            batch_id_map_file = f"{ctx.obj['base_folder']}/batch_id_map.json"
            if os.path.exists(batch_id_map_file):
                ctx.obj['subfolder2id'] = load_batch_id_map_from_file(batch_id_map_file)
            else:
                ctx.obj['subfolder2id'] = {}


@cli.command()
@click.option(
    '--azure/--no-azure', type=bool, default=False,
    help='Use Azure OpenAI API instead of OpenAI API.'
)
@click.option(
    '--data-to-annotate', type=str, default=None,
    help=(
        'JSONL or CSV file within "data/source/" directory containing the prompts to be annotated.'
        '\n    -If JSONL, each line should be a JSON object containing a "prompt" and "idx" key.'
        '\n    -If CSV, it should have "prompt" and "idx" columns.'
    )
)
@click.option(
    '--openai-model', type=str, default='gpt-4o',
    help='Expected OpenAI model to use.'
)
@click.option(
    '--max-completion-tokens', type=int, default=1000,
    callback=validate_positive_integer_callback,
    help='Maximum number of tokens for the completion.'
)
@click.option(
    '--base-folder', default=None,
    help='Path to the base folder for the manager to run everything.'
)
@click.option(
    '--source-data-file', type=click.Path(exists=True), default=None,
    help='Path to the source data file containing prompts to be annotated. '
)
@click.option(
    '--rubrics-folder', type=click.Path(exists=True), default=None,
    help='Path to the folder containing demand levels rubrics.'
)
@click.option(
    '--annotations-folder', default="./data/annotations",
    help='Path to the folder where annotations will be saved.'
)
@click.option('--force', is_flag=True, help='Skip confirmation prompts.')
@click.pass_context
def setup(ctx, azure, data_to_annotate, openai_model, max_completion_tokens,
          base_folder, source_data_file, rubrics_folder, annotations_folder, force):
    """Setup the batch run environment."""
    run_name = ctx.obj['run_name']

    if data_to_annotate and source_data_file:
        raise click.UsageError("Please provide either 'data-to-annotate' or 'source-data-file', not both.")

    if base_folder:
        base_folder = os.path.abspath(base_folder)
        msg_exists = (f"Base folder '{base_folder}' already exists."
                      + "Proceeding but results will be overwritten. "
                      + "To avoid this, re-setup using a different base folder.")
        msg_create = (f"Creating base folder for the run at '{base_folder}'.")
    else:
        base_folder = os.path.abspath(f"./runs/batch_runs_{run_name}")
        msg_exists = (f"Base folder associated to run '{run_name}' already exists. "
                      + "Proceeding but results will be overwritten. "
                      + "To avoid this, re-setup using a different run name.")
        msg_create = (f"Creating base folder for the run at 'runs/batch_runs_{run_name}/'.")
    if os.path.exists(base_folder) and not force:
        logging.warning(msg_exists)
        click.confirm("Do you want to proceed?", abort=True)
        logging.info("Proceeding with existing base folder.")
    else:
        logging.info(msg_create)
        os.mkdir(base_folder)

    if source_data_file:
        source_data_file = os.path.abspath(source_data_file)
        msg = (f"Source data file '{source_data_file}' does not exist. "
               + "Please provide a valid CSV or JSONL file.")
    else:
        source_data_file = os.path.abspath(f'./data/{data_to_annotate}')
        msg = (f"Source data file '{data_to_annotate}' does not exist. "
               + "Please provide a valid CSV or JSONL file in 'data/' directory.")
    if not os.path.exists(source_data_file):
        logging.error(msg)
        raise SystemExit(1)

    if source_data_file.endswith('.jsonl'):
        ok, wrong_keys = check_source_data_jsonl_keys(source_data_file)
        if not ok:
            found_lines = '\n'.join([f"Found: {keys} at line {i+1}." for i, keys in wrong_keys.items()])
            logging.error(f"Each line in the JSONL file must contain 'prompt' and 'idx' keys.\n{found_lines}")
            raise SystemExit(1)
    elif source_data_file.endswith('.csv'):
        ok = check_source_data_csv_columns(source_data_file)
        if not ok:
            logging.error("CSV file must contain 'prompt' and 'idx' columns.")
            raise SystemExit(1)
    else:
        logging.error("The data to annotate must be a JSONL or CSV file.")
        raise SystemExit(1)

    if rubrics_folder:
        rubrics_folder = os.path.abspath(rubrics_folder)
        msg = (f"Rubrics folder '{rubrics_folder}' does not exist. "
               + "Please provide a valid folder containing demand levels rubrics.")
    else:
        rubrics_folder = os.path.abspath('./data/rubrics')
        msg = ("Rubrics folder does not appear to be under expected location. "
               + "Please ensure you have the demand levels rubrics in 'data/rubrics/' directory.")
    if not os.path.exists(rubrics_folder):
        logging.error(msg)
        raise SystemExit(1)

    annotations_folder = os.path.abspath(annotations_folder)
    if not os.path.exists(annotations_folder):
        logging.info(f"Creating annotations folder at '{annotations_folder}'.")
        os.mkdir(annotations_folder)

    manager_file = {
        "run": run_name,
        "API": "AzureOpenAI" if azure else "OpenAI",
        "base_folder": base_folder,  # Base folder for the run
        "source_data_file": source_data_file,
        "rubrics_folder": rubrics_folder,
        "annotations_folder": annotations_folder,
        "openai_body_url": "/chat/completions",
        "openai_model": openai_model,
        "openai_max_completion_tokens": max_completion_tokens,
    }

    manager_file_path = os.path.join(base_folder, "ManagerFile.yaml")
    with open(manager_file_path, 'w') as f:
        yaml.dump(manager_file, f, default_flow_style=False)


@cli.command()
@click.option(
    '-m', '--openai-model', multiple=True, default=[],
    help='Expected OpenAI model(s) to use. Pass multiple models separated by spaces.'
)
@click.option(
    '--max-completion-tokens', type=int, default=None,
    callback=validate_positive_integer_callback,
    help='Maximum number of tokens for the completion.'
)
@click.option(
    '--estimation', type=click.Choice(['aprox', 'exact'], case_sensitive=False),
    help='Estimation type: "aprox" for approximate (99.8%) cost, "exact" for exact cost. '
)
@click.pass_context
def get_pricing(ctx, openai_model, max_completion_tokens, estimation):
    """Get OpenAI Batch API pricing for expected usage."""
    source_data_path = ctx.obj['source_data_file']
    prompts = read_only_prompts_from_source_data(source_data_path)

    rubrics_folder = ctx.obj['rubrics_folder']
    rubrics = read_demand_levels_rubric_files(rubrics_folder)

    logging.info("Calculating batch API pricing...")
    openai_model = openai_model or [ctx.obj['openai_model']]
    max_tokens = max_completion_tokens or ctx.obj['max_completion_tokens']
    for model in openai_model:
        estimated_cost = get_batch_api_pricing(
            prompts=prompts,
            rubrics=rubrics,
            max_completion_tokens=max_tokens,
            openai_model=model,
            estimation=estimation
        )
        logging.info(f"Estimated cost for model {model}: ${estimated_cost:.2f}")


@cli.command()
@click.pass_context
def create_input_files(ctx):
    """Create JSONL batch files for OpenAI Batch API."""
    logging.info("Reading prompt data...")
    source_data_path = ctx.obj['source_data_file']
    prompt_data = read_source_data(source_data_path)

    logging.info("Reading rubrics...")
    rubrics_folder = ctx.obj['rubrics_folder']
    rubrics = read_demand_levels_rubric_files(rubrics_folder)

    # OpenAI batch API limits: https://platform.openai.com/docs/guides/batch#rate-limits
    # Azure OpenAI batch API limits: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/batch?tabs=global-batch%2Cstandard-input%2Cpython-secure&pivots=programming-language-python
    max_bytes_per_file = 190 * 1024**2                                       # 190 MB (200-10 MB for overhead)
    max_lines_per_file = 100_000 if ctx.obj['API'] == "AzureOpenAI" else 50_000  # 100K or 50k lines

    logging.info("Creating batch files...")
    create_subdomain_batch_input_files(
        prompt_data=prompt_data,
        rubrics=rubrics,
        output_dir=ctx.obj['base_folder'],
        max_completion_tokens=ctx.obj['openai_max_completion_tokens'],
        openai_model=ctx.obj['openai_model'],
        body_url=ctx.obj['openai_body_url'],
        max_lines_per_file=max_lines_per_file,
        max_bytes_per_file=max_bytes_per_file
    )


@cli.command()
@click.option(
    '--file-type', type=click.Choice(['jsonl', 'csv'], case_sensitive=False), default='jsonl',
    help='Type of parsed output file. Default is "jsonl".'
)
@click.option(
    '--verbose', is_flag=True, default=True,
    help='If set, will print detailed information about failures during parsing.'
)
@click.pass_context
def parse_output_files(ctx, file_type, verbose):
    """Parse output JSONL files from OpenAI Batch Jobs. Creates a single JSONL or CSV file with all annotations."""
    base_folder = ctx.obj['base_folder']
    annotations_path = ctx.obj['annotations_folder']
    run_name = ctx.obj['run_name']
    output_path = f"{annotations_path}/{run_name}_annotations.{file_type}"

    logging.info(f"Parsing batch output files...")
    parse_subdomain_batch_output_files(
        base_folder=base_folder,
        output_path=output_path,
        file_type=file_type,
        return_pandas=False,
        verbose=verbose
    )


@cli.command()
@click.argument(
    'demands', nargs=-1,
    help=(
        'Demand(s) batch jobs to launch separated by spaces. '
        'If not provided, jobs will be launched for all batch input files created. '
        'Note that demands should match the subfolder names in the base folder. '
        'In case of multiple parts per demand you can either provide the full name ' 
        '(e.g., "KNs_part1") to launch that specific part, or just the demand name '
        '(e.g., "KNs") to lauch all parts related to that demand.')
)
@click.option(
    '--parallel', default=False, is_flag=True,
    help='Use parallel processing (recommended for multiple large input files)'
)
@click.pass_context
def launch(ctx, demands, parallel):
    """Launch OpenAI Batch Jobs."""
    base_folder = ctx.obj['base_folder']
    client = ctx.obj['client']
    endpoint = ctx.obj['endpoint']

    if demands:
        input_files, failed = __safe_get_input_files_from_demands(demands, base_folder)
        if failed:
            logging.error(f"Cannot launch any job because of demands: {failed}. "
                          + "Please ensure the input files are created for these demands or check if names are correct.")
            raise SystemExit(1)

        for input_file in input_files:
            subfolder = input_file.parent
            batch_id = launch_batch_job(client, input_file, endpoint)
            ctx.obj['subfolder2id'][str(subfolder)] = batch_id

    else:
        if parallel:
            ctx.obj['subfolder2id'] = launch_all_batch_jobs_parallel(client, base_folder, endpoint)
        else:
            ctx.obj['subfolder2id'] = launch_all_batch_jobs(client, base_folder, endpoint)

    batch_id_map_file = os.path.join(base_folder, "batch_id_map.json")
    save_batch_id_map_to_file(ctx.obj['subfolder2id'], batch_id_map_file)
    logging.info(f"Batch ID map correctly saved")


@cli.command()
@click.argument(
    'demands', nargs=-1,
    help=(
        'Demand(s) batch jobs to launch separated by spaces. '
        'If not provided, jobs will be launched for all batch input files created. '
        'Note that demands should match the subfolder names in the base folder. '
        'In case of multiple parts per demand you can either provide the full name ' 
        '(e.g., "KNs_part1") to launch that specific part, or just the demand name '
        '(e.g., "KNs") to lauch all parts related to that demand.')
)
@click.option(
    '--parallel', default=False, is_flag=True,
    help='Use parallel processing (recommended for multiple jobs).'
)
@click.pass_context
def check(ctx, demands, parallel):
    """Check the status of OpenAI Batch Jobs."""
    client = ctx.obj['client']
    base_folder = ctx.obj['base_folder']

    batch_id_map, demands_launched = __safe_get_batch_id_map_and_demands_launched(ctx)

    if demands:
        subfolders, batch_ids, failed = __safe_get_subfolders_and_batch_ids_from_demands(demands, base_folder, batch_id_map)
        if failed:
            logging.error(f"Cannot check status for any job because of demands: {failed}. "
                          + "Please ensure the jobs are launched or check if names are correct. "
                          + f"Current demand jobs launched: {demands_launched}")
            raise SystemExit(1)

        for subfolder, batch_id in zip(subfolders, batch_ids):
            check_batch_status(client, batch_id, verbose=True)

    else:
        batch_id_list = list(batch_id_map.values())
        if parallel:
            check_all_batch_status_parallel(client, batch_id_list, verbose=True)
        else:
            check_all_batch_status(client, batch_id_list, verbose=True)


@cli.command()
@click.argument(
    'demands', nargs=-1,
    help=(
        'Demand(s) batch jobs to launch separated by spaces. '
        'If not provided, jobs will be launched for all batch input files created. '
        'Note that demands should match the subfolder names in the base folder. '
        'In case of multiple parts per demand you can either provide the full name ' 
        '(e.g., "KNs_part1") to launch that specific part, or just the demand name '
        '(e.g., "KNs") to lauch all parts related to that demand.')
)
@click.option(
    '--parallel', default=False, is_flag=True,
    help='Use parallel processing (recommended for multiple dense jobs with large files to be downloaded).'
)
@click.pass_context
def download(ctx, demands, parallel):
    """Download results of OpenAI Batch Jobs. Use this command only if you are sure that the jobs to be downloaded are succesfully completed."""
    client = ctx.obj['client']
    base_folder = ctx.obj['base_folder']

    batch_id_map, demands_launched = __safe_get_batch_id_map_and_demands_launched(ctx)

    if demands:
        subfolders, batch_ids, failed = __safe_get_subfolders_and_batch_ids_from_demands(demands, base_folder, batch_id_map)
        if failed:
            logging.error(f"Cannot download any result because of demands: {failed}. "
                          + "Please ensure the jobs are launched or check if names are correct. "
                          + f"Current demand jobs launched: {demands_launched}")
            raise SystemExit(1)
        
        for subfolder, batch_id in zip(subfolders, batch_ids):
            download_batch_result(client, batch_id, subfolder)

    else:
        if parallel:
            download_all_batch_results_parallel(client, batch_id_map, base_folder)
        else:
            download_all_batch_results(client, batch_id_map, base_folder)


@cli.command()
@click.argument(
    'demands', nargs=-1,
    help=(
        'Demand(s) batch jobs to cancel, separated by spaces. '
        'If not provided, jobs will be cancelled for all batch input files created. '
        'Note that demands should match the subfolder names in the base folder. '
        'In case of multiple parts per demand you can either provide the full name ' 
        '(e.g., "KNs_part1") to launch that specific part, or just the demand name '
        '(e.g., "KNs") to lauch all parts related to that demand.')
)
@click.pass_context
def cancel(ctx, demands):
    """Cancel OpenAI Batch Jobs."""
    client = ctx.obj['client']
    base_folder = ctx.obj['base_folder']
    batch_id_map, demands_launched = __safe_get_batch_id_map_and_demands_launched(ctx)

    if demands:
        subfolders, batch_ids, failed = __safe_get_subfolders_and_batch_ids_from_demands(demands, base_folder, batch_id_map)
        if failed:
            logging.error(f"Cannot cancel any job because of demands: {failed}. "
                          + "Please ensure the jobs are launched or check if names are correct. "
                          + f"Current demand jobs launched: {demands_launched}")
            raise SystemExit(1)

        for subfolder, batch_id in zip(subfolders, batch_ids):
            cancel_batch_job(client, batch_id)
            del ctx.obj['subfolder2id'][subfolder]

    else:
        cancel_all_batch_jobs(client, batch_id_map.values())
        ctx.obj['subfolder2id'] = {}


@cli.command()
@click.option(
    '--check-interval', default=1800, type=int,
    callback=validate_positive_integer_callback,
    help='Interval (seconds) between attempts to download results of OpenAI Batch Jobs.'
)
@click.option(
    '--n-jobs', default=5, type=int,
    callback=validate_positive_integer_callback,
    help='Number of parallel jobs. -1 for all available cores.'
)
@click.pass_context
def track_and_download_loop(ctx, check_interval, n_jobs):
    """Track and download results of OpenAI Batch Jobs in a loop until completion."""
    base_folder = ctx.obj['base_folder']
    client = ctx.obj['client']
    batch_id_map = ctx.obj['subfolder2id']
    max_workers = resolve_n_jobs(n_jobs)

    track_and_download_all_batch_jobs_parallel_loop(
        client=client,
        batch_id_map=batch_id_map,
        base_folder=base_folder,
        check_interval=check_interval,
        max_workers=max_workers
    )


@cli.command()
@click.option(
    '-s', '--status', default=None,
    help='Filter jobs by status.'
)
@click.pass_context
def list(ctx, status):
    client = ctx.obj['client']
    list_batch_jobs(client, status)


#=======================================================================
# Private Safe Getters
#=======================================================================

def __safe_get_subfolders(base_folder):
    """
    Safely get the demand subfolders from the base folder.
    If no subfolders are found, the program ends.
    """
    subfolders = [f for f in Path(base_folder).iterdir() if f.is_dir()]
    if not subfolders:
        logging.error(f"No subfolders found in base folder '{base_folder}'. "
                      "Please create input files first using 'create-input-files' command.")
        raise SystemExit(1)
    return subfolders


def __safe_get_input_files_from_demands(demands, base_folder):
    """Get the subfolders for given demands."""
    subfolders = __safe_get_subfolders(base_folder)
    infiles = []
    failed = []
    for demand in demands:
        demand_subfolders = [subf for subf in subfolders if demand in subf.name]
        if not demand_subfolders:
            failed.append(demand)
        for subfolder in demand_subfolders:
            input_file = subfolder / "input.jsonl"
            if input_file.exists():
                infiles.append(input_file)
            else:
                failed.append(demand)
    return infiles, failed


def __safe_get_batch_id_map_and_demands_launched(ctx):
    """
    Safely get the batch ID map and launched demands from the context.
    If no jobs are found, the program ends.
    """
    batch_id_map = ctx.obj.get('subfolder2id', {})
    if not batch_id_map:
        logging.error("No jobs found. Please launch batch jobs first using 'launch' command.")
        raise SystemExit(1)
    demands_launched = [Path(subf).name for subf in batch_id_map.keys()]
    return batch_id_map, demands_launched


def __safe_get_subfolders_and_batch_ids_from_demands(demands, base_folder, batch_id_map):
    """Get the subfolder and batch ID for given demands."""
    subfolders = __safe_get_subfolders(base_folder)
    subfs = []
    bids = []
    failed = []
    for demand in demands:
        demand_subfolders = [subf.name for subf in subfolders if demand in subf.name]
        if not demand_subfolders:
            failed.append(demand)
        for subfolder in demand_subfolders:
            batch_id = batch_id_map.get(subfolder)
            if batch_id:
                subfs.append(subfolder)
                bids.append(batch_id)
            else:
                failed.append(demand)
    return subfs, bids, failed
