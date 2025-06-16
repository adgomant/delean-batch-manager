"""
CLI entry point for DeLeAn Batch Manager.

This module handles environment setup, logging configuration, and launches 
the CLI interface. It ensures proper initialization before CLI commands execute.
"""

import sys
import logging


def __setup_main_logging(verbose=False, quiet=False):
    """
    Configure logging for entry point execution. 
    Allows logging when setting up environment variables.
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Configure logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True  # Override any existing configuration
    )


def __setup_cli_environment(verbose=False):
    """Set up environment variables for CLI usage."""
    logger = logging.getLogger(__name__)

    try:
        # Import environment setup from core
        from ..core.utils.environment import setup_environment
        setup_environment(verbose=verbose)

        if verbose:
            logger.debug("Environment setup completed successfully")

        # Optionally validate required environment variables
        from ..core.utils.environment import validate_required_env_vars
        missing = validate_required_env_vars()
        if missing:
            logger.warning(f"Missing required environment variables: {', '.join(missing)}")

    except Exception as e:
        logger.warning(f"Environment setup failed: {e}")
        if verbose:
            logger.debug("Environment variables might be set system-wide", exc_info=True)
        # Don't fail - environment variables might be set system-wide


def main():
    """Main CLI entry point with full setup."""
    # Parse flags early for setup configuration
    verbose = '-v' in sys.argv or '--verbose' in sys.argv
    quiet = '-q' in sys.argv or '--quiet' in sys.argv

    # 1. Configure logging FIRST
    __setup_main_logging(verbose=verbose, quiet=quiet)

    # 2. Set up environment variables (now logging is available)
    __setup_cli_environment(verbose=verbose)

    # 3. Import and run CLI (after all setup is complete)
    logger = logging.getLogger(__name__)

    try:
        logger.debug("Starting CLI execution")
        from .cli import cli
        cli()

    except KeyboardInterrupt:
        logger.info("CLI interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT

    except Exception as e:
        if verbose:
            logger.exception("CLI execution failed")
        else:
            logger.error(f"CLI execution failed: {e}")
            raise e
        sys.exit(1)


if __name__ == '__main__':
    main()
