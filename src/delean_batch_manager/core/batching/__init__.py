"""
Batch processing operations for DeLeAn Batch Manager.

This module contains all batch processing functionality organized into submodules:

Submodules:
    files:   File operations (create input files, parse output files)
    jobs:    Job management (launch, monitor, download results)
    utils:   Batch-specific utilities (ID mapping, validation)
    pricing: Cost estimation and pricing information
    manager: High-level batch manager class (future)

Example Usage:
    import delean_batch_manager as dbm

    # File operations
    dbm.batching.files.create_subdomain_batch_input_files(...)
    dbm.batching.files.parse_subdomain_batch_output_files(...)

    # Job management
    batch_id = dbm.batching.jobs.launch_batch_job(client, input_file)
    status = dbm.batching.jobs.check_batch_status(client, batch_id)
    dbm.batching.jobs.download_batch_result(client, batch_id, output_file)

    # Utilities
    batch_map = dbm.batching.utils.load_batch_id_map_from_file(map_file)
    pricing = dbm.batching.pricing.get_batch_api_pricing(...)
"""

# Import submodules (not individual functions)
from . import files
from . import jobs
from . import utils
from . import pricing
from . import manager

__all__ = [
    'files',      # dbm.batching.files.*
    'jobs',       # dbm.batching.jobs.*
    'utils',      # dbm.batching.utils.*
    'pricing',    # dbm.batching.pricing.*
    'manager',    # dbm.batching.manager.*
]
