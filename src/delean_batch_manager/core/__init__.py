"""
Core functionality for DeLeAn Batch Manager.

This module contains the complete backend business logic for batch processing
operations, organized into focused areas of responsibility.

Architecture:
    batching/   - Batch processing operations
      ├── files/     - Input/output file operations
      ├── jobs/      - Job lifecycle management  
      ├── utils/     - Batch-specific utilities
      ├── pricing/   - Cost estimation
      └── manager/   - High-level batch orchestration (future)

    utils/      - Shared utilities and infrastructure
      ├── clients/   - API client creation (OpenAI, Azure)
      ├── rubrics/   - Rubrics management system
      ├── datasource/- Data validation and processing
      ├── registry/  - Run configuration management (internal)
      ├── misc/      - General utilities (internal)
      └── environment/ - Environment setup (internal)

Public API Structure:
    The core module exports organized namespaces that provide:

    batching.files.*     - File creation and parsing operations
    batching.jobs.*      - Job launch, monitoring, and results
    batching.utils.*     - Batch ID mapping and utilities
    batching.pricing.*   - Cost estimation and pricing info

    utils.clients.*      - OpenAI and Azure OpenAI client creation
    utils.rubrics.*      - Rubrics loading, validation, management
    utils.datasource.*   - Source data reading and validation
"""

# Core module exports
from . import batching
from . import utils

# High-level manager interface
from .batching.manager import DeLeAnBatchManager

__all__ = [
    'batching',    # Comprehensive batch processing operations
    'utils',       # Essential utilities and infrastructure
    'DeLeAnBatchManager',  # High-level orchestration interface
]

