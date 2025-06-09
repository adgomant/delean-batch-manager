"""
DeLeAn Batch Manager - Demand Level Annotation through OpenAI Batch Jobs

A comprehensive toolkit for managing OpenAI Batch API jobs to obtain demand level 
annotations from large language models. This package provides both a programmatic 
API and a command-line interface for batch processing workflows.

Key Features:
    - Rubrics-based demand level annotation
    - OpenAI and Azure OpenAI Batch API support
    - Parallel job management and monitoring
    - Flexible configuration and run management
    - Comprehensive result parsing and analysis

Package Structure:
    batching: Core batch processing operations (files, jobs, monitoring)
    rubrics:  Rubrics management and validation system
    utils:    Shared utilities (clients, data handling)

Example Usage:

    Basic Workflow:
        import delean_batch_manager as dbm

        # Setup rubrics and data
        catalog = dbm.utils.rubrics.RubricsCatalog('./rubrics/')
        data = dbm.utils.datasource.read_source_data('./data.jsonl')

        # Create batch input files
        dbm.batching.create_subdomain_batch_input_files(
            prompt_data=data,
            rubrics=catalog.get_rubrics_dict(),
            output_dir='./batch_files/'
        )

        # Launch and monitor jobs
        client = dbm.utils.clients.create_openai_client()
        job_id = dbm.batching.launch_batch_job(client, './batch_files/AS/input.jsonl')
        status = dbm.batching.check_batch_status(client, job_id)

    High-Level Interface:
        # Use the manager class for simplified workflows
        manager = dbm.DeLeAnBatchManager(
            client=dbm.utils.create_azure_openai_client(),
            base_folder='./batch_runs/',
            source_data='./data.jsonl',
            rubrics_folder='./rubrics/',
            max_completion_tokens=1000,
            openai_model='my-custom-gpt-model-deployed-at-azure',
        )
        manager.create_input_files()
        manager.launch_jobs(parallel=True)
        manager.track_and_download_loop(check_interval=120)
        manager.parse_output_files_and_save_results(output_path='./results/')

    CLI Usage:
        # Command-line interface for complete workflows
        $ deleanbm -r exp1 setup --source-data-file ./data.jsonl --rubrics-folder ./rubrics/
        $ deleanbm -r exp1 create-input-files
        $ deleanbm -r exp1 launch --parallel
        $ ... (rest of commands)

Environment Setup:
    Required environment variables:
    - OPENAI_API_KEY (for OpenAI API)
    - AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT (for Azure OpenAI)

    These can be set via .env files in:
    - Current working directory (.env, .env.local)
    - Project root directory
"""

__version__ = "0.1.0"
__author__ = "Alvar"

# Load environment on package import
from .core.utils.environment import setup_environment
setup_environment()

# Export core API modules
from . import core
batching = core.batching
utils = core.utils
DeLeAnBatchManager = core.DeLeAnBatchManager

__all__ = [
    '__version__',
    '__author__',
    'batching',        # dbm.batching.*
    'utils',           # dbm.utils.*
    'DeLeAnBatchManager',  # dbm.DeLeAnBatchManager()
]

# Clean up namespace
del setup_environment, core
