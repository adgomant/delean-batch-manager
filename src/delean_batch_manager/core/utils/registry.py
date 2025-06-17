# -*- coding: utf-8 -*-

"""
Centralized run registry for managing run configurations across the system.
"""

import yaml
import platformdirs
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import logging

from .misc import read_yaml, write_yaml


# Global registry instance
_registry = None


class RunRegistry:
    """Centralized registry for managing batch run configurations."""

    def __init__(self):
        self.registry_path = self._get_registry_path()
        self._ensure_registry_exists()

    def _get_registry_path(self) -> Path:
        """Get the platform-specific registry path."""
        config_dir = platformdirs.user_config_dir("delean-batch-manager", "delean")
        return Path(config_dir) / "runs_registry.yaml"

    def _ensure_registry_exists(self):
        """Ensure the registry directory and file exist."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._save_registry({"runs": {}})

    def _load_registry(self) -> Dict:
        """Load the registry from disk."""
        try:
            config = read_yaml(self.registry_path)
            return config or {"runs": {}}
        except Exception as e:
            logging.warning(f"Error loading registry: {e}. Creating new registry.")
            return {"runs": {}}

    def _save_registry(self, registry: Dict):
        """Save the registry to disk."""
        try:
            write_yaml(registry, self.registry_path)
        except Exception as e:
            logging.error(f"Error saving registry: {e}")
            raise

    def register_run(self, run_name: str, config_path: str, base_folder: str):
        """Register a new run in the registry."""
        registry = self._load_registry()

        # Convert to absolute paths
        config_path = str(Path(config_path).resolve())
        base_folder = str(Path(base_folder).resolve())

        registry["runs"][run_name] = {
            "config_path": config_path,
            "base_folder": base_folder,
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat()
        }

        self._save_registry(registry)
        logging.debug(f"Registered run '{run_name}' in global registry")

    def get_run_config(self, run_name: str) -> Optional[Dict]:
        """Get configuration for a specific run."""
        registry = self._load_registry()
        run_info = registry["runs"].get(run_name)

        if not run_info:
            return None

        # Verify config file still exists
        config_path = Path(run_info["config_path"])
        if not config_path.exists():
            logging.warning(f"Config file for run '{run_name}' no longer exists: {config_path}")
            return None

        # Update last accessed time
        run_info["last_accessed"] = datetime.now().isoformat()
        self._save_registry(registry)

        # Load and return the actual config
        try:
            config = read_yaml(config_path)
            return config
        except Exception as e:
            logging.error(f"Error loading config for run '{run_name}': {e}")
            return None

    def list_runs(self) -> List[Dict]:
        """List all registered runs with their info."""
        registry = self._load_registry()
        runs = []

        for run_name, run_info in registry["runs"].items():
            config_exists = Path(run_info["config_path"]).exists()
            runs.append({
                "name": run_name,
                "base_folder": run_info["base_folder"],
                "created_at": run_info["created_at"],
                "last_accessed": run_info["last_accessed"],
                "config_exists": config_exists
            })

        return sorted(runs, key=lambda x: x["last_accessed"], reverse=True)

    def unregister_run(self, run_name: str) -> bool:
        """Remove a run from the registry."""
        registry = self._load_registry()

        if run_name in registry["runs"]:
            del registry["runs"][run_name]
            self._save_registry(registry)
            logging.info(f"Unregistered run '{run_name}' from global registry")
            return True

        return False

    def cleanup_orphaned_runs(self) -> List[str]:
        """Remove runs whose config files no longer exist."""
        registry = self._load_registry()
        orphaned = []

        for run_name, run_info in list(registry["runs"].items()):
            if not Path(run_info["config_path"]).exists():
                orphaned.append(run_name)
                del registry["runs"][run_name]

        if orphaned:
            self._save_registry(registry)
            logging.info(f"Cleaned up {len(orphaned)} orphaned runs: {orphaned}")

        return orphaned


def get_registry() -> RunRegistry:
    """Get the global run registry instance."""
    global _registry
    if _registry is None:
        _registry = RunRegistry()
    return _registry
