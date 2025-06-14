# -*- coding: utf-8 -*-

import logging
from datetime import datetime
from pathlib import Path
import click
import yaml
from typing import Literal

from ..core.utils.registry import get_registry
from ..core.utils.datasource import (
    check_jsonl_source_data_keys,
    check_tabular_source_data_columns
)
from ..core.utils.misc import (
    mask_path,
    ensure_output_path
)


def setup_logging(verbose=False, quiet=False):
    """Configure logging for CLI execution."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True  # Override any existing configuration
    )

    # Reduce noise from external libraries in non-verbose mode
    if not verbose:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("azure").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    if verbose:
        logger = logging.getLogger(__name__)
        logger.debug("CLI logging setup completed")


def _validate_positive_integer_callback(ctx, param, value):
    """Validate that the provided value is a positive integer."""
    if value is not None and value <= 0:
        raise click.BadParameter("Value must be a positive integer.")
    return value


#=======================================================================
# Setup Command Utilities
#=======================================================================

def _handle_existing_run_setup(
        run_name, existing_config, source_data_file, rubrics_folder,
        base_folder, annotations_folder, openai_model,
        max_completion_tokens, azure, force
    ):
    """Handle setup for existing runs. Update configuration file with provided options and
    """

    provided_options = _resolve_paths(
        source_data_file=source_data_file,
        rubrics_folder=rubrics_folder,
        base_folder=base_folder,
        annotations_folder=annotations_folder
    )

    if provided_options:
        logging.warning("Modifying paths (e.g., source data, rubrics folder, "
                        "base folder...) in an existing run is discouraged, "
                        "especially if jobs have already been launched or "
                        "results downloaded. This may lead to overwriting "
                        "existing files or annotations. We recommend setting "
                        "up a new run instead to ensure data integrity.")
        if not force:
            click.confirm(
                "Do you really want to proceed?",
                abort=True
            )

        # If proceeding, validate provided paths
        _validate_configuration_paths(provided_options, force)

    # Add non-path options
    if openai_model is not None:
        provided_options['openai_model'] = openai_model
    if max_completion_tokens is not None:
        provided_options['openai_max_completion_tokens'] = max_completion_tokens
    if azure is not None:
        provided_options['API'] = "AzureOpenAI" if azure else "OpenAI"

    # Calculate and confirm changes
    changes = _get_configuration_changes(existing_config, provided_options)

    if not _confirm_configuration_changes(run_name, changes, force):
        _display_current_config(existing_config)
        return

    # Apply updates with special handling
    updated_config = _apply_configuration_updates(
        existing_config, provided_options
    )

    # Save configuration
    _save_and_register_configuration(updated_config, run_name)

    logging.info(f"Configuration updated successfully for run '{run_name}'!")


def _handle_new_run_setup(
        run_name, source_data_file, rubrics_folder, base_folder, 
        annotations_folder, openai_model, max_completion_tokens, azure, force
    ):
    """
    Handle setup for new runs - require all main options.
    Creates a new run configuration file with the provided options 
    and registers it in the global registry.
    """
    # Check required options for new runs
    missing_options = []
    if source_data_file is None:
        missing_options.append('--source-data-file')
    if rubrics_folder is None:
        missing_options.append('--rubrics-folder')  
    if base_folder is None:
        missing_options.append('--base-folder')

    if missing_options:
        logging.error(f"For new runs, the following options are required: {', '.join(missing_options)}")
        logging.info(f"Example: deleanbm -r {run_name} setup --source-data-file data.jsonl --rubrics-folder ./rubrics --base-folder ./runs/{run_name}")
        raise SystemExit(1)

    # Set defaults for new runs
    if openai_model is None:
        openai_model = 'gpt-4o'
    if max_completion_tokens is None:
        max_completion_tokens = 1000
    if azure is None:
        azure = False
    if annotations_folder is None:
        annotations_folder = str(Path(base_folder) / "annotations")

    # Resolve and validate all paths
    resolved_paths = _resolve_paths(
        source_data_file=source_data_file,
        rubrics_folder=rubrics_folder,
        base_folder=base_folder,
        annotations_folder=annotations_folder
    )
    _validate_configuration_paths(resolved_paths, force=force)

    # Create configuration
    config = {
        "run": run_name,
        "API": "AzureOpenAI" if azure else "OpenAI",
        "base_folder": resolved_paths['base_folder'],
        "source_data_file": resolved_paths['source_data_file'],
        "rubrics_folder": resolved_paths['rubrics_folder'],
        "annotations_folder": resolved_paths['annotations_folder'],
        "openai_body_url": "/chat/completions",
        "openai_model": openai_model,
        "openai_max_completion_tokens": max_completion_tokens,
        "created_at": datetime.now().isoformat(),
    }

    # Save configuration
    _save_and_register_configuration(config, run_name)

    logging.info(f"Setup complete! You can now run other commands for run '{run_name}'")

    # Show next steps
    logging.info("Next steps:")
    logging.info(f"1. Create input files: deleanbm -r {run_name} create-input-files")
    logging.info(f"2. Launch batch jobs: deleanbm -r {run_name} launch [--parallel]")
    logging.info(f"3. Check status: deleanbm -r {run_name} check [--parallel]")


def _resolve_paths(**path_kwargs):
    """
    Resolve provided paths to absolute paths and validate their existence.

    Args:
        **path_kwargs: Keyword arguments with path values to resolve and validate.
                      Keys should match configuration keys (source_data_file, rubrics_folder, etc.)

    Returns:
        dict: Resolved paths that were provided (not None)
    """
    resolved = {}
    for key, path_value in path_kwargs.items():
        if path_value is not None:
            resolved_path = str(Path(path_value).resolve())
            resolved[key] = resolved_path
    return resolved


def _validate_source_data_file(source_data_file_path):
    """Validate source data file existence, format and content."""
    source_data_file = Path(source_data_file_path)

    if not source_data_file.exists():
        logging.error(f"Source data file not found: {mask_path(str(source_data_file))}")
        raise SystemExit(1)

    # Check file extension
    if source_data_file.suffix not in ['.jsonl', '.csv', '.parquet']:
        logging.error("Source data file must be a .jsonl, .csv or .parquet file.")
        raise SystemExit(1)

    # Validate file content based on format
    if source_data_file.suffix == '.jsonl':
        ok, wrong_keys = check_jsonl_source_data_keys(str(source_data_file))
        if not ok:
            logging.error("JSONL file must have 'prompt' and 'custom_id' "
                          f"keys on each line but found:")
            for i, keys in wrong_keys.items():
                logging.error(f"  Line {i+1}: {keys}")
            raise SystemExit(1)

    elif source_data_file.suffix in ['.csv', '.parquet']:
        ok = check_tabular_source_data_columns(str(source_data_file))
        if not ok:
            logging.error("Tabular files (CSV or PARQUET) must contain 'prompt' and 'custom_id' columns.")
            raise SystemExit(1)


def _validate_rubrics_folder(rubrics_folder_path):
    """Validate rubrics folder exists and contains .txt files."""
    rubrics_path = Path(rubrics_folder_path)

    if not rubrics_path.exists():
        logging.error(f"Rubrics folder not found: {mask_path(str(rubrics_path))}")
        raise SystemExit(1)

    # Check for rubric files
    rubric_files = list(rubrics_path.glob("*.txt"))
    if not rubric_files:
        logging.error(f"No .txt files found in {mask_path(str(rubrics_path))}")
        raise SystemExit(1)

    rubric_names = [rf.stem for rf in rubric_files]
    logging.info(f"Found {len(rubric_files)} rubric files: {rubric_names}")


def _validate_configuration_paths(config, force):
    """Validate all paths in configuration exist and are accessible."""

    # Validate source data file if present
    if 'source_data_file' in config:
        _validate_source_data_file(config['source_data_file'])

    # Validate rubrics folder if present
    if 'rubrics_folder' in config:
        _validate_rubrics_folder(config['rubrics_folder'])

    # Ensure base folder is accessible (create if needed)
    if 'base_folder' in config:
        _prepare_base_folder(config['base_folder'], force=force)

    # Ensure annotations folder is accessible (create if needed)
    if 'annotations_folder' in config:
        ensure_output_path(config['annotations_folder'], "Annotations folder")


def _prepare_base_folder(base_folder_path, force=False):
    """
    Prepare base folder with proper confirmation if it exists and is not empty.

    Args:
        base_folder_path (str): Path to base folder
        force (bool): Skip confirmation if True

    Returns:
        Path: Resolved base folder path
    """
    base_folder = Path(base_folder_path)

    # Check if folder exists and is not empty
    if base_folder.exists() and not force:
        if any(base_folder.iterdir()):  # Check if folder is not empty
            logging.warning(f"Base folder {mask_path(str(base_folder))} already exists "
                            "and is not empty.")
            click.confirm(
                "Do you want to proceed? Existing files may be overwritten.",
                abort=True
            )

    # Ensure the folder exists
    ensure_output_path(str(base_folder), "Base folder")

    return base_folder


def _get_configuration_changes(existing_config, provided_options):
    """
    Get what changes will be made to configuration.

    Args:
        existing_config (dict): Current configuration
        provided_options (dict): New options provided by user

    Returns:
        list: List of change dictionaries with 'option', 'old', 'new' keys
    """
    changes = []

    for key, new_value in provided_options.items():
        old_value = existing_config.get(key, 'not set')
        if str(old_value) != str(new_value):
            # Mask paths for display
            display_old = mask_path(str(old_value)) if 'folder' in key or 'file' in key else old_value
            display_new = mask_path(str(new_value)) if 'folder' in key or 'file' in key else new_value

            changes.append({
                'option': key,
                'old': display_old,
                'new': display_new
            })

    return changes


def _confirm_configuration_changes(run_name, changes, force=False):
    """
    Show changes and get user confirmation.

    Args:
        run_name (str): Name of the run being updated
        changes (list): List of changes from _calculate_configuration_changes
        force (bool): Skip confirmation if True

    Returns:
        bool: True if user confirmed or no changes, False otherwise
    """
    if not changes:
        logging.info(f"No changes detected for run '{run_name}'")
        return False

    # Show what will change
    logging.info(f"Run '{run_name}' already exists. The following changes will be made:")
    for change in changes:
        logging.info(f"  {change['option']}: '{change['old']}' → '{change['new']}'")

    # Ask for confirmation unless --force
    if not force:
        click.confirm(
            f"Do you want to update the configuration for run '{run_name}'?",
            abort=True
        )

    return True


def _apply_configuration_updates(existing_config, provided_options):
    """
    Apply updates to existing configuration with special handling for related options.

    Args:
        existing_config (dict): Current configuration
        provided_options (dict): New options to apply
        annotations_folder (str, optional): Explicitly provided annotations folder

    Returns:
        dict: Updated configuration
    """
    updated_config = existing_config.copy()
    updated_config.update(provided_options)

    if 'API' in provided_options:
        old_api = existing_config.get('API', 'OpenAI')
        new_api = provided_options['API']
        if old_api != new_api:
            current_model = existing_config['openai_model']
            logging.warning(f"API changed from {old_api} to {new_api}. "
                            f"Make sure the model is consistent with the new API: {current_model}"
                            "Consider updating --openai-model if needed.")

    return updated_config


def _display_current_config(config):
    """Display current configuration in a readable format."""
    logging.info("Current configuration:")
    important_keys = [
        'source_data_file', 'rubrics_folder', 'base_folder', 
        'annotations_folder', 'openai_model', 'openai_max_completion_tokens', 'API'
    ]

    for key in important_keys:
        if key in config:
            value = config[key]
            if 'folder' in key or 'file' in key:
                value = mask_path(str(value))
            logging.info(f"  {key}: {value}")


def _save_and_register_configuration(config, run_name):
    """Save the configuration to file and registry."""
    base_folder = Path(config['base_folder'])

    # Update timestamp
    config['updated_at'] = datetime.now().isoformat()

    # Save to ManagerFile.yaml
    config_file = base_folder / "ManagerFile.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Update registry
    registry = get_registry()
    registry.register_run(run_name, str(config_file), str(base_folder))

    logging.info(f"Configuration saved to {mask_path(str(config_file))}")


#=======================================================================
# Batching Commands Utilities
#=======================================================================

def _safe_get_subfolders(base_folder: str | Path) -> list[Path]:
    """
    Safely get the all the demand subfolders from the base folder.
    If no subfolders are found, the program ends.
    This help to prevent attempting to do operations that
    requires the subfolders and input files to be created first.
    """
    subfolders = [
        f for f in Path(base_folder).iterdir()
        if f.is_dir() and (f/"input.jsonl").exists()
    ]
    if not subfolders:
        logging.error(f"No subfolders found in base folder '{base_folder}'. "
                      "Please create input files first using "
                      "'create-input-files' command.")
        raise SystemExit(1)
    return subfolders


def _get_demand_files(
        demands: list,
        all_subfolders: list[Path],
        which: Literal['input', 'output'] = 'input'
    ) -> tuple[list[Path], list[str]]:
    """Get the subfolders for given demands."""
    files = []
    failed = []
    for demand in demands:
        demand_subfolders = [
            subf for subf in all_subfolders
            if subf.name.startswith(demand)
        ]
        if not demand_subfolders:
            failed.append(demand)
        for subfolder in demand_subfolders:
            input_file = subfolder / f"{which}.jsonl"
            if input_file.exists():
                files.append(input_file)
            else:
                failed.append(demand)
    return files, failed


def _safe_get_batch_id_map(ctx) -> dict:
    """
    Safely get the batch ID map from the context.
    If no jobs are found, the program ends.
    This helps to prevent attempting operations
    that requires the jobs be launched first.
    """
    batch_id_map = ctx.obj.get('subfolder2id', {})
    if not batch_id_map:
        logging.error("No jobs found. Please launch batch jobs first "
                      "using 'launch' command.")
        raise SystemExit(1)
    return batch_id_map


def _get_launched_demands(batch_id_map: dict) -> list[str]:
    """Get the list of demands launched from the batch ID map."""
    return [Path(subf).name for subf in batch_id_map.keys()]


def _get_demand_subfolders_and_batch_ids(
        demands: list[str],
        all_subfolders: list[Path],
        batch_id_map: dict[str, str]
    ) -> tuple[list[Path], list[str], list[str]]:
    """Get the subfolder and batch ID for given demands."""
    subfs = []
    bids = []
    failed = []
    for demand in demands:
        demand_subfolders = [
            subf for subf in all_subfolders
            if subf.name.startswith(demand)
        ]
        if not demand_subfolders:
            failed.append(demand)
        for subfolder in demand_subfolders:
            batch_id = batch_id_map.get(str(subfolder))
            if batch_id:
                subfs.append(subfolder)
                bids.append(batch_id)
            else:
                failed.append(demand)
    return subfs, bids, failed
