"""
Shared utilities for DeLeAn Batch Manager.

This module provides essential utilities organized into focused submodules:

Submodules:
    clients:     API client creation and management (OpenAI, Azure OpenAI)
    rubrics:     Rubrics management and validation system  
    datasource:  Data handling and validation utilities
    registry:    Run configuration registry (internal)
    misc:        Internal utilities (internal)
    environment: Environment configuration (internal)

Example Usage:
    import delean_batch_manager as dbm
    
    # API client management
    client = dbm.utils.clients.create_openai_client()
    azure_client = dbm.utils.clients.create_azure_openai_client()
    
    # Rubrics management
    catalog = dbm.utils.rubrics.RubricsCatalog('./rubrics/')
    improvements = dbm.utils.rubrics.suggest_rubric_improvements(catalog)
    
    # Data handling
    data = dbm.utils.datasource.read_source_data('./data.jsonl')
    dbm.utils.datasource.check_source_data_jsonl_keys(data)
"""

# Import modules to export
from . import clients     # Client creation utilities
from . import rubrics     # Full rubrics module
from . import datasource  # Full datasource module

__all__ = [
    'clients',      # dbm.utils.clients.* (OpenAI/Azure client functions)
    'rubrics',      # dbm.utils.rubrics.* (RubricsCatalog, validation, etc.)
    'datasource',   # dbm.utils.datasource.* (data reading, validation, etc.)
]

# Internal modules not exported:
# - registry (internal run management)
# - misc (internal utilities, only clients exported via clients module)
# - environment (internal environment setup)