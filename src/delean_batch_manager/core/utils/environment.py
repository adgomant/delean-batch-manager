# -*- coding: utf-8 -*-

"""
Environment configuration management.
"""

import os
import logging
from pathlib import Path
from typing import Optional
import dotenv


def load_environment_variables(env_file: Optional[str] = None, verbose: bool = False) -> bool:
    """
    Load environment variables from .env file with smart path resolution.

    Args:
        env_file: Specific .env file path. If None, searches for .env files.
        verbose: Whether to log environment loading details.

    Returns:
        True if .env file was found and loaded, False otherwise.
    """
    if env_file:
        # Use specific file
        env_path = Path(env_file)
        if env_path.exists():
            dotenv.load_dotenv(env_path)
            if verbose:
                logging.debug(f"Loaded environment from: {env_path}")
            return True
        else:
            if verbose:
                logging.warning(f"Specified .env file not found: {env_path}")
            return False

    # Search for .env files in common locations
    search_paths = [
        Path.cwd() / '.env.local',
        Path.cwd() / '.env',
    ]

    # Also search in project root (if we can detect it)
    try:
        import delean_batch_manager
        package_root = Path(delean_batch_manager.__file__).parent.parent.parent
        search_paths.extend([
            package_root / '.env.local',
            package_root / '.env',
        ])
    except Exception:
        pass  # Can't detect project root, skip

    # Try each path
    for env_path in search_paths:
        if env_path.exists():
            dotenv.load_dotenv(env_path)
            if verbose:
                logging.debug(f"Loaded environment from: {env_path}")
            return True

    if verbose:
        logging.debug("No .env file found in search paths")
    return False


def validate_required_env_vars(api_type: str = "OpenAI") -> list:
    """
    Validate that required environment variables are set.

    Args:
        api_type: Either "OpenAI" or "AzureOpenAI"

    Returns:
        List of missing environment variables (empty if all present)
    """
    missing = []

    if api_type == "OpenAI":
        if not os.getenv('OPENAI_API_KEY'):
            missing.append('OPENAI_API_KEY')

    elif api_type == "AzureOpenAI":
        if not os.getenv('AZURE_OPENAI_API_KEY'):
            missing.append('AZURE_OPENAI_API_KEY')
        if not os.getenv('AZURE_OPENAI_ENDPOINT'):
            missing.append('AZURE_OPENAI_ENDPOINT')

    return missing


def setup_environment(verbose: bool = False, env_file: Optional[str] = None) -> bool:
    """
    Set up environment for the package.

    Args:
        verbose: Whether to log environment setup details
        env_file: Optional specific .env file to load

    Returns:
        True if environment setup was successful
    """
    # Load environment variables
    env_loaded = load_environment_variables(env_file, verbose)

    if verbose and not env_loaded:
        logging.debug("No .env file loaded. Relying on system environment variables.")
        logging.debug("Expected .env file locations:")
        logging.debug("  - ./.env (current directory)")
        logging.debug("  - ./.env.local (current directory)")
        logging.debug("  - <proyect_root>/.env (proyect root directory)")
        logging.debug("  - <proyect_root>/.env.local (proyect root directory)")

    return True  # Always return True since .env is optional