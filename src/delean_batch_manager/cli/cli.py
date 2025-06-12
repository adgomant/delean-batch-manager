# -*- coding: utf-8 -*-

import os
import sys
import click
import logging
from pathlib import Path

from ..core.batching.pricing import get_batch_api_pricing
from ..core.batching.files import (
    create_subdomain_batch_input_files,
    parse_subdomain_batch_output_files,
    save_parsed_results
)
from ..core.batching.jobs import (
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
from ..core.batching.utils import (
    save_batch_id_map_to_file,
    load_batch_id_map_from_file,
)
from ..core.utils.registry import get_registry
from ..core.utils.rubrics import RubricsCatalog
from ..core.utils.datasource import (
    read_source_data,
    read_only_prompts_from_source_data,
)
from ..core.utils.clients import (
    create_openai_client,
    create_azure_openai_client
)
from ..core.utils.misc import (
    resolve_n_jobs,
    mask_path,
)
from ..core.utils.environment import (
    validate_required_env_vars
)
from .utils import (
    setup_logging,
    _validate_positive_integer_callback,
    _handle_existing_run_setup,
    _handle_new_run_setup,
    _safe_get_subfolders,
    _safe_get_batch_id_map,
    _get_demand_input_files,
    _get_demand_subfolders_and_batch_ids,
    _get_launched_demands
)


@click.group()
@click.option(
    '-v', '--verbose', is_flag=True,
    help='Enable verbose (DEBUG) logging'
)
@click.option(
    '-q', '--quiet', is_flag=True,
    help='Only show warnings and errors'
)
@click.option(
    '-r', '--run-name', type=str,
    help='Name of the run to work with.'
)
@click.pass_context
def cli(ctx, verbose, quiet, run_name):
    """
    DeLeAn Batch Manager CLI - A sophisticated tool for managing OpenAI Batch
    Jobs to obtain Demand Levels Annotations.

    This CLI allows you to set up batch runs, create input files, launch jobs,
    check their status, download results, and parse output files. It supports
    both OpenAI and Azure OpenAI APIs.

    \b
    Ensure you have the appropriate API keys set in your environment variables:
    - OPENAI_API_KEY (for OpenAI)
    - AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT (for Azure OpenAI)
    """
    # Set up logging first
    setup_logging(verbose=verbose, quiet=quiet)

    ctx.ensure_object(dict)
    ctx.obj['run_name'] = run_name

    # Skip checks if --help/-h is requested
    if any(arg in sys.argv for arg in ['--help', '-h']):
        return

    if not run_name and ctx.invoked_subcommand not in ['list-runs', 'unregister-run']:
        logging.error("Please specify a run name using the -r or --run-name option.")
        raise click.UsageError("Run name is required except for 'list-runs' "
                               "and 'unregister-run' commands.")

    if ctx.invoked_subcommand not in ['setup', 'list-runs', 'unregister-run']:
        # Load configuration from registry
        registry = get_registry()
        run_config = registry.get_run_config(run_name)

        if not run_config:
            logging.error(f"No configuration found for run '{run_name}'. "
                          f"Please run 'deleanbm --run-name {run_name} setup' "
                          "first, or use 'deleanbm list-runs' to see "
                          "available runs.")
            raise SystemExit(1)

        # Load configuration into context
        for key, value in run_config.items():
            ctx.obj[key] = value

        # Validate environment variables for the chosen API
        missing_vars = validate_required_env_vars(ctx.obj['API'])
        if missing_vars:
            logging.error(f"Missing required environment variables for {api}: {missing_vars}")
            logging.info("Please set these environment variables or create a "
                         ".env file at the repo root directory with:")
            for var in missing_vars:
                logging.info(f"  {var}=your_key_here")
            logging.info("Or set them in your system environment global variables running the following commands:")
            for var in missing_vars:
                logging.info(f"$ export {var}=your_key_here")
            raise SystemExit(1)

        # Create API client
        try:
            if ctx.obj['API'] == 'AzureOpenAI':
                ctx.obj['client'] = create_azure_openai_client()
            else:
                ctx.obj['client'] = create_openai_client()
        except ValueError as e:
            logging.error(f"Error creating {ctx.obj['API']} client: {e}")
            raise SystemExit(1)

        # Load batch ID map if exists
        batch_id_map_file = Path(ctx.obj['base_folder']) / "batch_id_map.json"
        if batch_id_map_file.exists():
            ctx.obj['subfolder2id'] = load_batch_id_map_from_file(
                str(batch_id_map_file), verbose=verbose
            )
        else:
            ctx.obj['subfolder2id'] = {}

    else:
        # No need to load configuration for setup, list-runs, or unregister-run commands
        return


@cli.command()
@click.option(
    '--source-data-file', type=click.Path(exists=True), default=None,
    help=('Path to the source data file containing prompts to be annotated. '
          'Required for new runs.')
)
@click.option(
    '--rubrics-folder', type=click.Path(exists=True), default=None,
    help=('Path to the folder containing demand level rubrics. '
          'Required for new runs.')
)
@click.option(
    '--base-folder', type=click.Path(), default=None,
    help=('Path to where all batch processing files will be stored. '
          'Required for new runs.')
)
@click.option(
    '--annotations-folder', type=click.Path(), default=None,
    help=('Path to where final annotation results will be saved. '
          'For new runs, default is base-folder/annotations/')
)
@click.option(
    '--openai-model', type=str, default=None,
    help=('OpenAI model to use for batch processing. '
          'For new runs, default is \'gpt-4o\'.')
)
@click.option(
    '--max-completion-tokens', type=int, default=None,
    callback=_validate_positive_integer_callback,
    help=('Maximum number of tokens for each completion. '
          'For new runs, default is 1000.')
)
@click.option(
    '--azure/--no-azure', default=False,
    help=('Use Azure OpenAI API instead of OpenAI API. '
          'For new runs, default is --no-azure.')
)
@click.option(
    '--force', is_flag=True, default=False,
    help=('Skip confirmation prompts when updating existing configurations.')
)
@click.pass_context
def setup(ctx, source_data_file, rubrics_folder, base_folder,
          annotations_folder, openai_model, max_completion_tokens, azure, force):
    """
    Setup a new batch processing run or update an existing one.

    For new runs, all main options are required (see example below).
    For existing runs, you can omit options to keep current values and
    use current configuration, or provide specific options to update
    only those values.

    \b
    Examples:
        # New run (some options required)
        deleanbm -r experiment1 setup \\
            # Required options
            --source-data-file ~/data/prompts.jsonl \\
            --rubrics-folder ~/research/rubrics/ \\
            --base-folder ~/experiments/batch_runs/exp1/
            # Optional options
            --annotations-folder ~/data/annotations \\
            --openai-model gpt-4o \\
            ...

        # Load existing run configuration (no options needed)
        deleanbm -r experiment1 setup

        # Update only the model for existing run
        deleanbm -r experiment1 setup --openai-model gpt-4o-mini
    """
    run_name = ctx.obj['run_name']
    registry = get_registry()
    existing_config = registry.get_run_config(run_name)

    if existing_config:
        return _handle_existing_run_setup(
            run_name, existing_config, source_data_file,
            rubrics_folder, base_folder, annotations_folder,
            openai_model, max_completion_tokens, azure, force
        )
    else:
        return _handle_new_run_setup(
            run_name, source_data_file, rubrics_folder,
            base_folder, annotations_folder, openai_model,
            max_completion_tokens, azure, force
        )


@cli.command()
@click.pass_context
def list_runs(ctx):
    """List all registered runs and their status."""
    registry = get_registry()
    runs = registry.list_runs()

    if not runs:
        logging.info("No runs registered. Use 'setup' command to create a run.")
        return

    logging.info(f"Found {len(runs)} registered runs:")
    logging.info("")

    for run in runs:
        status = "exists" if run["config_exists"] else "missing"
        logging.info(f"  {run['name']}")
        logging.info(f"    ManagerFile Status: {status}")
        logging.info(f"    Base folder: {mask_path(run['base_folder'])}")
        logging.info(f"    Last used: {run['last_accessed'][:19].replace('T', ' ')}")
        logging.info("")


@cli.command()
@click.argument('run_name', required=False)
@click.option(
    '--cleanup-orphaned', is_flag=True,
    help='Remove runs whose config files no longer exist.'
)
@click.pass_context
def unregister_run(ctx, run_name, cleanup_orphaned):
    """Remove a run from the global registry."""
    registry = get_registry()

    if cleanup_orphaned:
        orphaned = registry.cleanup_orphaned_runs()
        if orphaned:
            logging.info(f"Removed {len(orphaned)} orphaned runs: {orphaned}")
        else:
            logging.info("No orphaned runs found.")
        return

    if not run_name:
        logging.error("Please specify a run name to unregister, or use --cleanup-orphaned")
        raise SystemExit(1)

    if registry.unregister_run(run_name):
        logging.info(f"Successfully unregistered run '{run_name}'")
        logging.info("Note: This only removes the registry entry, not the actual files.")
    else:
        logging.error(f"Run '{run_name}' not found in registry")
        raise SystemExit(1)


@cli.command()
@click.option(
    '-m', '--openai-model', multiple=True, default=[],
    help=('Expected OpenAI model(s) to use. '
          'Pass multiple models separated by spaces.')
)
@click.option(
    '--max-completion-tokens', type=int, default=None,
    callback=_validate_positive_integer_callback,
    help='Maximum number of tokens for the completion.'
)
@click.option(
    '--estimation', default="aprox",
    type=click.Choice(['aprox', 'exact'], case_sensitive=False),
    help=('Estimation type: "aprox" for approximate (99.8%) cost, '
          '"exact" for exact cost.')
)
@click.option(
    '--n-jobs', default=None, type=int,
    callback=_validate_positive_integer_callback,
    help=('Number of parallel jobs to use for \'exact\' estimation. -1 for '
          'all available cores. If None, runs in serial mode. Recommended '
          'when requiring exact estimation for large datasets.')
)
@click.pass_context
def get_pricing(ctx, openai_model, max_completion_tokens, estimation, n_jobs):
    """Get OpenAI Batch API pricing for expected usage."""
    source_data_path = ctx.obj['source_data_file']
    prompts = read_only_prompts_from_source_data(source_data_path)

    rubrics_folder = ctx.obj['rubrics_folder']
    try:
        handler = RubricsCatalog(rubrics_folder)
        rubrics = handler.get_rubrics_dict()
    except Exception as e:
        logging.error(f"Error reading rubrics from folder '{rubrics_folder}': {e}")
        raise SystemExit(1)

    logging.info("Calculating batch API pricing...")
    openai_model = openai_model or [ctx.obj['openai_model']]
    max_tokens = max_completion_tokens or ctx.obj['openai_max_completion_tokens']
    for model in openai_model:
        try:
            estimated_cost = get_batch_api_pricing(
                prompts=prompts,
                rubrics=rubrics,
                max_completion_tokens=max_tokens,
                openai_model=model,
                estimation=estimation,
                n_jobs=n_jobs
            )
            logging.info(f"Estimated cost for model {model}: ${estimated_cost:.2f}")
        except Exception as e:
            logging.error(f"Error calculating pricing for model '{model}': {e}")
            raise SystemExit(1)


@cli.command()
@click.pass_context
def create_input_files(ctx):
    """Create JSONL batch files for OpenAI Batch API."""
    logging.info("Reading prompt data...")
    source_data_path = ctx.obj['source_data_file']
    prompt_data = read_source_data(source_data_path)

    logging.info("Reading rubrics...")
    rubrics_folder = ctx.obj['rubrics_folder']
    try:
        handler = RubricsCatalog(rubrics_folder)
        rubrics = handler.get_rubrics_dict()
    except Exception as e:
        logging.error(f"Error reading rubrics from folder '{rubrics_folder}': {e}")
        raise SystemExit(1)

    # OpenAI batch API limits: https://platform.openai.com/docs/guides/batch#rate-limits
    # Azure OpenAI batch API limits: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/batch?tabs=global-batch%2Cstandard-input%2Cpython-secure&pivots=programming-language-python
    max_bytes_per_file = 190 * 1024**2                                           # 190 MB (200-10 MB for overhead)
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
    '--file-type', default='jsonl',
    type=click.Choice(['jsonl', 'csv', 'parquet'], case_sensitive=False),
    help='Type of parsed output file. Default is "jsonl".'
)
@click.option(
    '--format', default='long',
    type=click.Choice(['long', 'wide'], case_sensitive=False),
    help=('Format of the CSV output file. "long" format will have one row per '
          'annotation, while "wide" format will have one row per prompt '
          'with all levels as columns. Note that "wide" will not include '
          'finish reasons and completions, only levels, independently of the '
          '--only-levels flag.')
)
@click.option(
    '--only-levels', default=False, is_flag=True,
    help=('If set, will only include the levels in the output file, excluding '
          'finish reasons and completions.')
)
@click.option(
    '--only-success', default=False, is_flag=True,
    help=('If set, will only include the successful annotations in the output file. '
          'Note that this is mutually exclusive with --only-failures.')
)
@click.option(
    '--only-failures', default=False, is_flag=True,
    help=('If set, will only include the failed annotations in the output file.'
          'Note that this is mutually exclusive with --only-success.')
)
@click.option(
    '--split-by-demand', default=False, is_flag=True,
    help=('If set, will create separate output files for each demand. ')
)
@click.option(
    '--finish-reason', default=None,
    type=click.Choice(['stop', 'length', 'other'], case_sensitive=False),
    help=('Filter results by finish reason. '
          'If specified, only annotations with the specified finish reason will be included. '
          'If not specified, all finish reasons will be included.')
)
@click.option(
    '--include-prompts', default=False, is_flag=True,
    help=('If set, will include the original prompts in the output file. ')
)
@click.option(
    '--verbose', is_flag=True, default=True,
    help=('If set, logs warnings for any issues encountered when extracting '
          'demand levels at instance level.')
)
@click.pass_context
def parse_output_files(ctx, file_type, only_levels, csv_format, verbose):
    """
    Parse output JSONL files from OpenAI Batch Jobs. Creates a single JSONL
    or CSV file with all annotations.
    """
    base_folder = ctx.obj['base_folder']
    annotations_path = ctx.obj['annotations_folder']
    run_name = ctx.obj['run_name']

    logging.info("Parsing batch output files...")
    try:
        results = parse_subdomain_batch_output_files(
            base_folder=base_folder,
            only_levels=only_levels,
            verbose=verbose
        )
    except Exception as e:
        logging.error(f"Error parsing output files: {e}")
        raise SystemExit(1)

    output_path = f"{annotations_path}/{run_name}_annotations"
    if file_type == 'csv':
        output_path += f"_{csv_format}"
        if only_levels and csv_format == 'long':
            output_path += "_only_levels"
    else:
        if only_levels:
            output_path += "_only_levels"
    output_path += f".{file_type}"

    logging.info(f"Saving parsed results to {output_path}")
    try:
        save_parsed_results(
            results=results,
            output_path=output_path,
            file_type=file_type,
            csv_format=csv_format,
        )
    except Exception as e:
        logging.error(f"Error saving parsed results: {e}")
        raise SystemExit(1)


@cli.command()
@click.argument('demands', nargs=-1)
@click.option(
    '--parallel', default=False, is_flag=True,
    help='Use parallel processing (recommended for multiple large input files)'
)
@click.pass_context
def launch(ctx, demands, parallel):
    """
    Launch OpenAI Batch Jobs.

    \b
    DEMANDS:
      Demand(s) batch jobs to launch separated by spaces.
      If not provided, jobs will be launched for all batch input files created.
      Batch input files must be created first using the 'create-input-files' command.
      Note that demands should match the subfolder names in the base folder.
      In case of multiple parts per demand you can either provide the full name
      (e.g., "KNs_part1") to launch that specific part, or just the demand name
      (e.g., "KNs") to launch all parts related to that demand.
    """
    base_folder = ctx.obj['base_folder']
    client = ctx.obj['client']
    endpoint = ctx.obj['openai_body_url']

    # Get all demands subfolders,
    # exits if no subfolders found with valid input files in.
    all_subfolders = _safe_get_subfolders(base_folder)

    if demands:
        input_files, failed = _get_demand_input_files(demands, all_subfolders)
        if failed:
            logging.error("Cannot launch any job because of demands: "
                          f"{failed}. Please ensure the input files are "
                          "created for these demands or check if names "
                          "are correct.")
            raise SystemExit(1)

        for input_file in input_files:
            subfolder = input_file.parent
            batch_id = launch_batch_job(client, input_file, endpoint)
            ctx.obj['subfolder2id'][str(subfolder)] = batch_id

    else:
        if parallel:
            ctx.obj['subfolder2id'] = launch_all_batch_jobs_parallel(
                client, base_folder, endpoint
            )
        else:
            ctx.obj['subfolder2id'] = launch_all_batch_jobs(
                client, base_folder, endpoint
            )

    batch_id_map_file = os.path.join(base_folder, "batch_id_map.json")
    save_batch_id_map_to_file(ctx.obj['subfolder2id'], batch_id_map_file)
    logging.info(f"Batch ID map correctly saved")


@cli.command()
@click.argument('demands', nargs=-1)
@click.option(
    '--parallel', default=False, is_flag=True,
    help='Use parallel processing (recommended for multiple jobs).'
)
@click.pass_context
def check(ctx, demands, parallel):
    """
    Check the status of OpenAI Batch Jobs.

    \b
    DEMANDS:
      Demand(s) batch jobs to check their status separated by spaces.
      If not provided, status will be checked for all active and known batch jobs.
      Use 'list-jobs' command to see all known launched jobs.
      Note that demands should match the subfolder names in the base folder.
      In case of multiple parts per demand you can either provide the full name
      (e.g., "KNs_part1") to check that specific part, or just the demand name
      (e.g., "KNs") to check all parts related to that demand.
    """
    client = ctx.obj['client']
    base_folder = ctx.obj['base_folder']

    # Get all demands subfolders,
    # exits if no subfolders found with valid input files in.
    all_subfolders = _safe_get_subfolders(base_folder)

    # Get batch ID map, exits if no batch jobs found launched
    batch_id_map = _safe_get_batch_id_map(ctx)
    demands_launched = _get_launched_demands(batch_id_map)

    if demands:
        subfolders, batch_ids, failed = _get_demand_subfolders_and_batch_ids(
            demands, all_subfolders, batch_id_map
        )
        if failed:
            logging.error("Cannot check status for any job because of "
                          f"demands: {failed}. Please ensure the jobs "
                          "are launched or check if names are correct. "
                          f"Current demand jobs launched: {demands_launched}")
            raise SystemExit(1)

        for subfolder, batch_id in zip(subfolders, batch_ids):
            check_batch_status(client, batch_id, verbose=2)

    else:
        batch_id_list = list(batch_id_map.values())
        if parallel:
            check_all_batch_status_parallel(client, batch_id_list, verbose=2)
        else:
            check_all_batch_status(client, batch_id_list, verbose=2)


@cli.command()
@click.argument('demands', nargs=-1)
@click.option(
    '--parallel', default=False, is_flag=True,
    help=('Use parallel processing (recommended for multiple dense jobs with '
          + 'large files to be downloaded).')
)
@click.pass_context
def download(ctx, demands, parallel):
    """
    Download results of OpenAI Batch Jobs.

    \b
    DEMANDS:
      Demand(s) batch jobs to download results separated by spaces.
      If not provided, results will be downloaded for all batch jobs launched.
      Use 'list-jobs' command to see all known launched jobs.
      Note that demands should match the subfolder names in the base folder.
      In case of multiple parts per demand you can either provide the full name
      (e.g., "KNs_part1") to download that specific part, or just the demand name
      (e.g., "KNs") to download all parts related to that demand.

    WARNING:
      Use this command only if you are sure that the jobs to be downloaded
      are successfully completed.
    """
    client = ctx.obj['client']
    base_folder = ctx.obj['base_folder']

    # Get all demands subfolders,
    # exits if no subfolders found with valid input files in.
    all_subfolders = _safe_get_subfolders(base_folder)

    # Get batch ID map, exits if no batch jobs found launched
    batch_id_map = _safe_get_batch_id_map(ctx)
    demands_launched = _get_launched_demands(batch_id_map)

    if demands:
        subfolders, batch_ids, failed = _get_demand_subfolders_and_batch_ids(
            demands, all_subfolders, batch_id_map
        )
        if failed:
            logging.error("Cannot download any result because of demands: "
                          f"{failed}. Please ensure the jobs are launched or "
                          "check if names are correct. Current demand jobs "
                          f"launched: {demands_launched}")
            raise SystemExit(1)

        for subfolder, batch_id in zip(subfolders, batch_ids):
            download_batch_result(client, batch_id, subfolder)

    else:
        if parallel:
            download_all_batch_results_parallel(client, batch_id_map, base_folder)
        else:
            download_all_batch_results(client, batch_id_map, base_folder)


@cli.command()
@click.argument('demands', nargs=-1)
@click.pass_context
def cancel(ctx, demands):
    """
    Cancel OpenAI Batch Jobs. This will not delete the
    subfolders or input files, only the jobs themselves.
    Note that you need to wait until the jobs are validated
    before you can cancel them.

    \b
    DEMANDS:
      Demand(s) batch jobs to cancel, separated by spaces.
      If not provided, jobs will be cancelled for all batch jobs launched.
      Use 'list-jobs' command to see all known launched jobs.
      Note that demands should match the subfolder names in the base folder.
      In case of multiple parts per demand you can either provide the full name
      (e.g., "KNs_part1") to cancel that specific part, or just the demand name
      (e.g., "KNs") to cancel all parts related to that demand.
    """
    client = ctx.obj['client']
    base_folder = ctx.obj['base_folder']

    # Get all demands subfolders,
    # exits if no subfolders found with valid input files in.
    all_subfolders = _safe_get_subfolders(base_folder)

    # Get batch ID map, exits if no batch jobs found launched
    batch_id_map = _safe_get_batch_id_map(ctx)
    demands_launched = _get_launched_demands(batch_id_map)

    if demands:
        subfolders, batch_ids, failed = _get_demand_subfolders_and_batch_ids(
            demands, all_subfolders, batch_id_map
        )
        if failed:
            logging.error("Cannot cancel any job because of demands: "
                          f"{failed}. Please ensure the jobs are launched "
                          "or check if names are correct. Current demand "
                          f"jobs launched: {demands_launched}")
            raise SystemExit(1)

        for subfolder, batch_id in zip(subfolders, batch_ids):
            cancel_batch_job(client, batch_id)
            del ctx.obj['subfolder2id'][subfolder]

    else:
        cancel_all_batch_jobs(client, batch_id_map.values())
        ctx.obj['subfolder2id'] = {}

    # Update batch ID map file
    batch_id_map_file = os.path.join(base_folder, "batch_id_map.json")
    save_batch_id_map_to_file(
        ctx.obj['subfolder2id'], batch_id_map_file, verbose=False
    )


@cli.command()
@click.option(
    '--check-interval', default=1800, type=int,
    callback=_validate_positive_integer_callback,
    help=('Interval (seconds) between attempts to download results of '
          'OpenAI Batch Jobs.')
)
@click.option(
    '--n-jobs', default=5, type=int,
    callback=_validate_positive_integer_callback,
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
def list_jobs(ctx, status):
    """List all OpenAI Batch Jobs with their status."""
    client = ctx.obj['client']
    batch_id_map = _safe_get_batch_id_map(ctx)
    list_batch_jobs(client, batch_id_map, status)
