"""
Command-line interface for DeLeAn Batch Manager.

This module provides a comprehensive CLI for managing OpenAI Batch Jobs
to obtain demand level annotations. The CLI is organized into logical
command groups for different workflow stages.

Command Categories:
    Configuration:
        - setup: Configure a new batch run
        - list-runs: Show available runs
        - unregister-run: Remove run configuration

    File Operations:
        - create-input-files: Generate JSONL batch input files

    Job Management:
        - launch: Start batch jobs (single or all)
        - check: Check batch job progress and status (single or all)
        - cancel: Cancel running jobs

    Results:
        - download: Download completed job results (single or all)
        - parse-output-files: Parse results into structured format
            and saves to disk as CSV or JSONL

Environment Requirements:
    - OPENAI_API_KEY (for OpenAI API)
    - AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT (for Azure OpenAI)

Example Workflow:
    # 1. Set up run configuration
    $ deleanbm -r exp1 setup 
       --source-data-file ./data.jsonl 
       --rubrics-folder ./rubrics/ 
       --base-folder ./batch_runs/ 
       --annotations-folder ./annotations/
       --max-completion-tokens 1000
       --openai-model my-custom-gpt-model-deployed-at-azure
       --azure

    # 2. Create batch input files
    $ deleanbm -r exp1 create-input-files

    # 3. Launch jobs
    $ deleanbm -r exp1 launch --parallel

    # 4. Monitor progress
    $ deleanbm -r exp1 check --parallel

    # 5. Download results when complete
    $ deleanbm -r exp1 download --parallel

    # 6. Parse into final format
    $ deleanbm -r exp1 parse-output --file-type csv --csv-format long

The CLI provides extensive help for each command:
    $ deleanbm --help
    $ deleanbm setup --help
    $ deleanbm launch --help
"""

from .cli import cli

__all__ = [
    'cli',  # Main CLI interface (Click command group)
]
