# -*- coding: utf-8 -*-

"""
Rubrics management system for flexible demand level annotations.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RubricInfo:
    """Information about a single rubric."""
    acronym: str
    full_name: str
    content: str
    file_path: str


class RubricsCatalog:
    """
    Manages a catalog of demand level rubrics from file folders.

    Rubric files should follow this format:
    - Filename: {acronym}.txt (e.g., AS.txt, MyCustom.txt)
    - First line: Full name of the demand (e.g., "Attention and Search")  
    - Rest of file: Rubric content with level descriptions
    """

    def __init__(self, rubrics_folder: str):
        """
        Initialize the rubrics handler.

        Args:
            rubrics_folder: Path to folder containing rubric files.
        """
        self.rubrics_folder = Path(rubrics_folder)
        self._rubrics_cache: Dict[str, RubricInfo] = {}
        self._load_rubrics()

    def _load_rubrics(self):
        """Load all rubrics from the specified folder."""
        self._rubrics_cache.clear()

        if not self.rubrics_folder.exists():
            logging.error(f"Rubrics folder does not exist: {self.rubrics_folder}")
            return

        if not self.rubrics_folder.is_dir():
            logging.error(f"Rubrics path is not a directory: {self.rubrics_folder}")
            return

        loaded_count = 0
        for file_path in self.rubrics_folder.glob("*.txt"):
            try:
                acronym, full_name, content = self._parse_rubric_file(file_path)

                # Validate the rubric
                is_valid, validation_msg = validate_rubric_file(content)
                if not is_valid:
                    logging.warning(f"Invalid rubric in {file_path}: {validation_msg}")
                    continue

                self._rubrics_cache[acronym] = RubricInfo(
                    acronym=acronym,
                    full_name=full_name,
                    content=content,
                    file_path=str(file_path)
                )
                loaded_count += 1

            except Exception as e:
                logging.warning(f"Error loading rubric from {file_path}: {e}")

        logging.info(f"Loaded {loaded_count} rubrics from {self.rubrics_folder}")

    def _parse_rubric_file(self, file_path: Path) -> Tuple[str, str, str]:
        """Parse a rubric file according to the standard format."""
        # Acronym is the .txt file name
        acronym = file_path.stem

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            raise ValueError("Empty rubric file")

        # First line is the full name, preceeded by '#'
        full_name = lines[0].strip()
        if not full_name:
            raise ValueError("Missing full name on first line")
        if not full_name.startswith('#'):
            raise ValueError("First line must contain full name starting with '#'")
        full_name = full_name[1:].strip()  # Remove the leading '#'

        # The rest is the content
        if len(lines) < 2:
            raise ValueError("Missing rubric content")
        content = ''.join(lines[1:]).strip()

        return acronym, full_name, content

    def get_rubric(self, acronym: str) -> Optional[RubricInfo]:
        """Get a specific rubric by acronym."""
        return self._rubrics_cache.get(acronym)

    def get_rubrics_dict(self) -> Dict[str, dict]:
        """Get a dictionary of all rubrics with their full names and content indexed by acronym."""
        return {
            acronym: {
                "full_name": rubric.full_name,
                "content": rubric.content
            }
            for acronym, rubric in self._rubrics_cache.items()
        }

    def get_acronym_to_content_dict(self) -> Dict[str, str]:
        """Get mapping of acronyms to rubric content."""
        return {
            acronym: rubric.content
            for acronym, rubric in self._rubrics_cache.items()
        }

    def get_acronym_to_fullname_dict(self) -> Dict[str, str]:
        """Get mapping of acronyms to full names."""
        return {
            acronym: rubric.full_name
            for acronym, rubric in self._rubrics_cache.items()
        }

    def list_rubrics(self) -> List[Dict]:
        """List all rubrics with their information."""
        return [
            {
                "acronym": rubric.acronym,
                "full_name": rubric.full_name,
                "file_path": rubric.file_path
            }
            for rubric in sorted(self._rubrics_cache.values(), key=lambda r: r.acronym)
        ]

    def reload(self):
        """Reload all rubrics from the folder."""
        logging.info("Reloading rubrics...")
        self._load_rubrics()

    def add_custom_rubric(
            self,
            acronym: str,
            full_name: str,
            content: str,
            save_to_folder: bool = True
        ) -> bool:
        """
        Add a new custom rubric.

        Args:
            acronym: Short identifier for the rubric
            full_name: Full descriptive name
            content: Rubric content with level descriptions
            save_to_folder: Whether to save to the rubrics folder

        Returns:
            True if successful, False otherwise
        """
        # Validate content
        is_valid, validation_msg = validate_rubric_file(content)
        if not is_valid:
            logging.error(f"Invalid rubric content: {validation_msg}")
            return False

        # Create rubric info
        rubric_info = RubricInfo(
            acronym=acronym,
            full_name=full_name,
            content=content,
            file_path=""  # Will be set if saved to folder
        )

        # Save to folder if requested and folder exists
        if save_to_folder:
            try:
                self.rubrics_folder.mkdir(parents=True, exist_ok=True)
                file_path = self.rubrics_folder / f"{acronym}.txt"

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"{full_name}\n{content}")

                rubric_info.file_path = str(file_path)
                logging.info(f"Saved custom rubric '{acronym}' to {file_path}")

            except Exception as e:
                logging.error(f"Failed to save rubric to file: {e}")
                return False

        # Add to cache
        self._rubrics_cache[acronym] = rubric_info
        logging.info(f"Added custom rubric '{acronym}': {full_name}")

        return True

    def export_rubrics(self, export_folder: str):
        """Export all rubrics to a folder."""
        export_path = Path(export_folder)
        export_path.mkdir(parents=True, exist_ok=True)

        exported_count = 0
        for rubric in self._rubrics_cache.values():
            file_path = export_path / f"{rubric.acronym}.txt"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"{rubric.full_name}\n{rubric.content}")

            exported_count += 1

        logging.info(f"Exported {exported_count} rubrics to {export_folder}")


def validate_rubric_file(content: str) -> Tuple[bool, str]:
    """
    Validate rubric content to ensure it follows expected format.

    Args:
        content: The rubric content to validate

    Returns:
        Tuple of (is_valid, validation_message)
    """
    if not content or not content.strip():
        return False, "Empty rubric content"

    # Check for level mentions (0-5)
    level_pattern = r'[Ll]evel\s*[0-5]'
    levels_found = re.findall(level_pattern, content)

    if len(levels_found) < 6:  # All levels 0-5 should be present
        return False, "Rubric should contain, at least, all level descriptions from Level 0 to Level 5."

    # Check minimum length
    if len(content.strip()) < 100:
        return False, "Rubric content seems too short"

    # Check for examples
    example_pattern = r'[Ee]xamples?'
    if not re.search(example_pattern, content, re.IGNORECASE):
        return False, "Rubric should contain examples for each level"
    elif len(re.findall(example_pattern, content, re.IGNORECASE)) < 6:
        return False, "Rubric should contain at least one example for each Level from 0 to 5"

    return True, "Valid rubric"
