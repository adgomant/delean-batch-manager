# -*- coding: utf-8 -*-

import os
import json
import logging
from pathlib import Path


#=======================================================================
# JSON Lines Utilities
#=======================================================================

def write_jsonl(lines, path):
    """
    Write a list of dictionaries to a JSON Lines file.
    Each dictionary is written as a separate line in the file.

    Args:
        lines (list): List of dictionaries to write.
        path (str): Path to the output file.
    """
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(json.dumps(line) + '\n')
    return


def read_jsonl(path):
    """
    Read a JSON Lines file and return a list of dictionaries.

    Args:
        path (str): Path to the input file.

    Returns:
        list: List of dictionaries read from the file.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


#=======================================================================
# Parallel Processing Utilities
#=======================================================================

def resolve_n_jobs(n_jobs: int, verbose: bool = True) -> int:
    """
    Resolve the number of parallel workers to use based on n_jobs and available CPU cores.

    Args:
        n_jobs (int): Desired number of jobs.
            -1 means use all available cores minus one.
             1 means serial execution.
             >1 means use that many workers, capped at available cores.
        verbose (bool): If True, log the number of workers being used. Defaults to True.

    Returns:
        int: Number of workers to use.
    """
    available_cores = os.cpu_count() or 1
    if n_jobs == 1:
        return 1
    elif n_jobs == -1:
        num_workers = max(1, available_cores - 1)
        if verbose:
            logging.info(f"Using {num_workers} workers (all available cores minus one).")
        return num_workers
    elif n_jobs > 1:
        if n_jobs > available_cores:
            logging.warning(f"n_jobs={n_jobs} exceeds available cores. Using {available_cores} workers instead.")
            return available_cores
        if verbose:
            logging.info(f"Using {n_jobs} workers.")
        return n_jobs
    else:
        raise ValueError(f"Invalid n_jobs value: {n_jobs}. Must be >= 1 or -1.")
    

#=======================================================================
# Path Utilities
#=======================================================================

def mask_path(path, base_dir=None):
    """
    Masks or simplifies a path for logging.

    Args:
        path (str): The full path to mask.
        base_dir (str, optional): The base directory to make the path relative to.

    Returns:
        str: The masked or simplified path.
    """
    path = Path(path)

    # Use base_dir if provided, otherwise fallback to PROJECT_DIR from environment
    if base_dir is None:
        base_dir = os.getenv('PROJECT_DIR')

    if base_dir:
        # Make the path relative to the base directory
        base_dir = Path(base_dir)
        try:
            return str(path.relative_to(base_dir))
        except ValueError:
            pass  # If path is not under base_dir, fall back to absolute path

    # Replace home directory with "~"
    if str(path).startswith(str(Path.home())):
        return f"~/{path.relative_to(Path.home())}"


def assert_required_path(path, description="Path"):
    """
    Ensures that a required file or directory exists.

    Args:
        path (str): The path to check.
        description (str): Description of the resource for error messages.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if not os.path.exists(path):
        logging.error(f"{description} not found at: {mask_path(path)}")
        raise FileNotFoundError(f"{description} not found: {path}")


def ensure_output_path(path, description="Output folder"):
    """
    Verifica que un directorio de salida existe; si no, lo crea.

    Args:
        path (str): Ruta al directorio.
        description (str): Descripción del recurso (para logging).
    """
    if not os.path.exists(path):
        logging.info(f"{description} does not exist. Creating it at: {mask_path(path)}")
        os.makedirs(path, exist_ok=True)