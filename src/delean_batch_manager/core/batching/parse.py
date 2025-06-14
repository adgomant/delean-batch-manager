# -*- coding: utf-8 -*-

import re
import json
import logging
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import polars as pl

from ..utils.misc import mask_path, write_jsonl


@dataclass
class ParseArguments:
    """
    Configuration for parsing batch output files.
    """
    format: Literal['long', 'wide'] = 'long'
    only_levels: bool = False
    only_succeed: bool = False
    only_failed: bool = False
    finish_reason: Literal['stop', 'length', 'other'] | None = None
    source_prompts: Optional[dict] = None
    verbose: bool = False

    def __post_init__(self):
        self._check_arguments()

    def _check_arguments(self):
        """
        Check the validity of the arguments passed to the parsing functions.
        """
        if self.format not in ['long', 'wide']:
            raise ValueError("Invalid format. Expected 'long' or 'wide'.")

        if self.only_succeed and self.only_failed:
            logging.warning("Setting both `only_succeed` and `only_failed` "
                            "is redundant. Proceeding but consider setting "
                            "both to false to obtain the same results "
                            "(all annotations included)")

        if self.finish_reason not in ['stop', 'length', 'other', None]:
            raise ValueError("Invalid finish reason. Expected 'stop', "
                             "'length', 'other' or None.")


class BatchOutputParser:
    """
    A class to parse batch output files for demand levels.

    Attributes:
        format (str): The format of the output file. Can be 'long' or 'wide'.
        only_levels (bool): If True, only returns the demand levels without
            additional information.
        only_succeed (bool): If True, only includes successful annotations
            in the output file.
        only_failed (bool): If True, only includes failed annotations in
            the output file.
        finish_reason (str | None): If specified, only includes annotations
            with this finish reason.
        source_prompts (dict | None): A dictionary mapping custom ids to their
            corresponding prompts.
        verbose (bool): If True, logs warnings for any issues encountered when
            extracting demand levels.
    """
    def __init__(self, **kwargs):
        self.args = ParseArguments(**kwargs)
        self._output_files: list[str | Path] = []
        self._results: list[dict] = []

    def parse(self, output_files: str | Path | list[str | Path]):
        """
        Parse the output files based on the provided configuration.

        Args:
            output_files (str | Path | list[str | Path]): The output files to parse.
        """
        self._results = parse_subdomain_output_files(
            output_files=output_files,
            only_levels=self.args.only_levels,
            only_succeed=self.args.only_succeed,
            only_failed=self.args.only_failed,
            finish_reason=self.args.finish_reason,
            source_prompts=self.args.source_prompts,
            verbose=self.args.verbose
        )
        if isinstance(output_files, list):
            self._output_files = [Path(f).resolve() for f in output_files]
        else:
            self._output_files = [Path(output_files).resolve()]

    @classmethod
    def from_output_files(cls, output_files: str | Path | list[str | Path], **kwargs):
        """
        Create a BatchOutputParser instance and immediately parse the given output files.
        """
        instance = cls(**kwargs)
        instance.parse(output_files)
        return instance

    def to_jsonl(self) -> list[dict]:
        """Convert the parsed results to JSONL"""
        if self.args.format == 'wide':
            return _long_to_wide_jsonl(self._results)
        return self._results

    def to_df(self) -> pl.DataFrame:
        """Convert the parsed results to a Polars DataFrame."""
        df = pl.DataFrame(self._results)
        if self.args.format == 'wide':
            df = df.pivot(on='demand', index='custom_id', values='level')
            df = df.fill_null(np.nan)
        return df

    def _validate_path(self, path: str | Path):
        if not isinstance(path, (str, Path)):
            raise ValueError("`path` must be a string or a Path object.")
        path = Path(path).resolve()
        if path.is_dir():
            if not path.exists():
                logging.info("Creating directory: %s", mask_path(path))
                path.mkdir(parents=True, exist_ok=True)
        else:
            if not path.exists():
                raise ValueError(f"The specified file path does not exist: {mask_path(path)}")
        return path

    def _get_default_filename(
            self,
            base_name: str,
            prefix: str,
            extension: str
        ) -> str:
        """
        Generate a default file name for the annotations file based on the
        provided parameters.
        """
        return _get_default_annotations_file_name(
            base_name=base_name,
            prefix=prefix,
            finish_reason=self.args.finish_reason,
            only_succeed=self.args.only_succeed,
            only_failed=self.args.only_failed,
            only_levels=self.args.only_levels,
            format=self.args.format,
            extension=extension
        )

    def _write_base(
            self,
            path: str | Path,
            prefix: str,
            file_type: str,
            split_by_demand: bool = False
        ) -> None:
        if not self._results:
            raise ValueError("No results to write. Please parse the output files first.")
        path = self._validate_path(path)
        if file_type == 'jsonl':
            results = self.to_jsonl()
        else:
            results = self.to_df()
        if split_by_demand:
            if not path.is_dir() or self.args.format != 'long':
                raise ValueError(
                    "`split_by_demands` can only be True when `format` is 'long' "
                    "and `path` is a directory."
                )
            splits = _make_splits_by_subdomain(results)
            for subdomain, sub_results in splits.items():
                filename = self._get_default_filename(
                    base_name=f'annotations_{subdomain}',
                    prefix=prefix,
                    extension=file_type
                )
                save_path = path / filename
                match file_type:
                    case 'jsonl':
                        write_jsonl(sub_results, save_path)
                    case 'csv':
                        sub_results.write_csv(save_path)
                    case 'parquet':
                        sub_results.write_parquet(save_path)
        else:
            if path.is_dir():
                if len(self._output_files) == 1:
                    base_name = 'annotations_' + Path(self._output_files[0]).parent.name
                else:
                    base_name = 'annotations'
                filename = self._get_default_filename(
                    base_name=base_name,
                    prefix=prefix,
                    extension=file_type
                )
                path /= filename
            match file_type:
                case 'jsonl':
                    write_jsonl(results, path)
                case 'csv':
                    results.write_csv(path)
                case 'parquet':
                    results.write_parquet(path)

    def write_json(
            self, path: str | Path,
            prefix: str = '',
            split_by_demand: bool = False
        ) -> None:
        """
        Write the parsed results to a JSONL file.

        Args:
            path (str | Path): The path where the results should be saved.
                Can be a file or a directory. If a file is provided, it will
                save all results in that file, overwriting any existing
                content. If a directory is provided, the results will be
                saved in that directory using the following naming convention:
                '[<prefix>_]annotations[_<subdomain>][_<finish_reason>][_succeed][_failed][_only_levels]_<format>.jsonl'.
            prefix (str): An optional prefix to add to the output file name.
            split_by_demand (bool): If True, saves the results in separate
                files for each subdomain. Note that this is only applicable 
                when `format` was 'long' and `path` is a directory.
        """
        self._write_base(path, prefix, 'jsonl', split_by_demand)

    def write_csv(
            self, path: str | Path,
            prefix: str = '',
            split_by_demand: bool = False
        ) -> None:
        """
        Write the parsed results to a CSV file.

        Args:
            path (str | Path): The path where the results should be saved.
                Can be a file or a directory. If a file is provided, it will
                save all results in that file, overwriting any existing
                content. If a directory is provided, the results will be
                saved in that directory using the following naming convention:
                '[<prefix>_]annotations[_<subdomain>][_<finish_reason>][_succeed][_failed][_only_levels]_<format>.csv'.
            prefix (str): An optional prefix to add to the output file name.
            split_by_demand (bool): If True, saves the results in separate
                files for each subdomain. Note that this is only applicable 
                when `format` was 'long' and `path` is a directory.
        """
        self._write_base(path, prefix, 'csv', split_by_demand)

    def write_parquet(
            self, path: str | Path,
            prefix: str = '',
            split_by_demand: bool = False
        ) -> None:
        """
        Write the parsed results to a Parquet file.

        Args:
            path (str | Path): The path where the results should be saved.
                Can be a file or a directory. If a file is provided, it will
                save all results in that file, overwriting any existing
                content. If a directory is provided, the results will be
                saved in that directory using the following naming convention:
                '[<prefix>_]annotations[_<subdomain>][_<finish_reason>][_succeed][_failed][_only_levels]_<format>.parquet'.
            prefix (str): An optional prefix to add to the output file name.
            split_by_demand (bool): If True, saves the results in separate
                files for each subdomain. Note that this is only applicable 
                when `format` was 'long' and `path` is a directory.
        """
        self._write_base(path, prefix, 'parquet', split_by_demand)

    def summary(self, return_as: Literal['print', 'dict'] = 'print'):
        """
        Show or return a summary of parsed results:
        - Total count
        - Successful vs failed annotations
        - Breakdown of failed cases by finish_reason (if available)

        Args:
            return_as: 'print' to display the summary, 'dict' to return as a
                dictionary

        Returns:
            dict: Summary stats if return_as == 'dict'
        """
        if not self._results:
            raise ValueError("No results to write. Please parse the output files first.")

        total = len(self._results)
        failed_items = [r for r in self._results if np.isnan(r['level'])]
        failed = len(failed_items)
        success = total - failed

        failed_breakdown = {}
        if not self.args.only_levels:
            for r in failed_items:
                reason = r.get('finish_reason', 'N/A')
                failed_breakdown[reason] = failed_breakdown.get(reason, 0) + 1

        failed_with_breakdown = {
            'count': failed,
            'percent': round(100 * failed / total, 1),
        }
        if failed_breakdown:
            failed_with_breakdown['finish_reasons'] = {
                k: {
                    'count': v,
                    'percent': round(100 * v / failed, 1) if failed > 0 else 0.0
                } for k, v in failed_breakdown.items()
            }

        summary = {
            'total': total,
            'successful': {
                'count': success,
                'percent': round(100 * success / total, 1)
            },
            'failed': failed_with_breakdown
        }

        if return_as == 'dict':
            return summary

        # Print nicely
        print(f"\nParsed {total} annotations:")
        print(f"  Successful: {success} ({summary['successful']['percent']}%)")
        print(f"  Failed:     {failed} ({failed_with_breakdown['percent']}%)")
        if 'finish_reasons' in failed_with_breakdown:
            for k, v in failed_with_breakdown['finish_reasons'].items():
                print(f"    - {k}: {v['count']} ({v['percent']}%)")
        print()


def parse_subdomain_output_files(output_files, **kwargs) -> list:
    """
    Parse output files from the batch processing to extract demand levels.
    The output will contain one line per annotation with keys:
        - 'custom_id': The custom id of the prompt.
        - 'prompt': The original prompt for the custom id, if provided in
            `source_prompts`.
        - 'demand': The subdomain for which the demand level was assessed.
        - 'level': The extracted demand level, which is a float between 0
            and 5, or NaN if extraction fails.
        - 'finish_reason': The finish reason of the response. If only_levels
            is True, this key is not included.
        - 'model_response': The model's response content. If only_levels is
            True, this key is not included.

    Args:
        output_files (str, Path, list): Path to the output file containing the results
            of the batch processing for a specific subdomain, or a list of paths
            to multiple output files.
        only_levels (bool, optional): If True, only returns the demand levels
            without additional information.
        only_succeed (bool, optional): If True, only it will only include the
            successful annotations in the output file.
        only_failed (bool, optional): If True, it will only include the failed
            annotations in the output file.
        finish_reason (str, optional): If specified, only includes annotations
            with this finish reason.
        source_prompts (dict, optional): A dictionary mapping custom ids to
            their corresponding prompts.
        verbose (bool, optional): If True, logs warnings for any issues
            encountered when extracting demand levels.

    Raises:
        ValueError: If `output_files` is not a string, Path, or a list of strings/Paths.

    Returns:
        list: A list of dictionaries containing the parsed outputs.
    """
    if isinstance(output_files, (str, Path)):
        output_files = [output_files]
    elif not isinstance(output_files, list):
        raise ValueError("`output_files` must be a string, Path, or a list of strings/Paths.")

    logging.info(f"Parsing results from {len(output_files)} output files...")

    results = []
    for outfile in output_files:
        try:
            results.extend(
                _parse_subdomain_output_file_as_long_jsonl(
                    output_file=outfile,
                    only_levels=kwargs.get('only_levels', False),
                    only_succeed=kwargs.get('only_succeed', False),
                    only_failed=kwargs.get('only_failed', False),
                    finish_reason=kwargs.get('finish_reason', None),
                    source_prompts=kwargs.get('source_prompts', None),
                    verbose=kwargs.get('verbose', False),
                )
            )
        except Exception as e:
            logging.error(f"Error parsing file {mask_path(outfile)}: {e}")
            continue

    logging.info("All results parsed successfully.")

    return results


def _parse_subdomain_output_file_as_long_jsonl(
        output_file: str | Path,
        only_levels: bool = False,
        only_succeed: bool = False,
        only_failed: bool = False,
        finish_reason: Literal['stop', 'length', 'other'] | None = None,
        source_prompts: dict | None = None,
        verbose: bool = False,
    ) -> dict:
    """
    Parse the output file from the batch processing for a specific subdomain
    to extract demand levels in a long format. The output will contain one
    line per annotation.
    """
    results = []
    subdomain = Path(output_file).parent.name.split('_')[0]

    with open(output_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    for item in data:
        custom_id = item['custom_id']

        finish = item['response']['body']['choices'][0]['finish_reason']
        if finish_reason is not None:
            if (
                (finish in ['stop', 'length'] and finish != finish_reason)
                or
                (finish not in ['stop', 'length'] and finish_reason != 'other')
            ):
                continue
        response = item['response']['body']['choices'][0]['message']['content']
        if finish == 'stop':
            demand_level, ok = extract_demand_level_from_response(response)
            #print(demand_level, ok)
            if not ok:
                if verbose:
                    msg = (
                        f"Failed to extract demand level from response for "
                        f"item '{custom_id}' and subdomain '{subdomain}'. "
                        f"Response content: {repr(response)}"
                    )
                    logging.warning(msg)
                demand_level = np.nan
                failed = True
            else:
                failed = False
        else:
            demand_level = np.nan
            failed = True
        if (only_succeed and failed) or (only_failed and not failed):
            continue

        result = {
            'custom_id': custom_id,
            'demand': subdomain,
            'level': demand_level
        }

        if not only_levels:
            result['finish_reason'] = finish
            result['model_response'] = response

        if source_prompts is not None:
            result['prompt'] = source_prompts.get(custom_id, "")
        results.append(result)

    logging.info(f"Parsed {len(results)} results from {mask_path(output_file)}")

    return results


def extract_demand_level_from_response(response: str) -> float:
    """
    Extract the demand level from the response string.

    Args:
        response (str): The ressponse string from which to extract the demand level.

    Returns:
        float: The extracted demand level, or NaN if extraction fails.
    """
    *cot_steps, conclusion = response.split('\n\n')
    try:
        # Extract the last number from the conclusion
        match = re.findall(r'\d+', conclusion)
        demand_level = float(match[-1])
        if demand_level < 0 or demand_level > 5:
            # If the demand level is outside the expected range, 
            # result is considered invalid
            return float('nan'), False
        if len(match) == 1 and conclusion.startswith(str(demand_level)):
            # Avoid cases where the only number present in the final statement 
            # is a leading section number (yes, this could happen)
            # e.g., "4. Conclusion: Thus, the level of Attention and Search 
            # demanded by the given TASK INSTANCE is: **Not Applicable**"
            return float('nan'), False
        return demand_level, True
    except IndexError:
        return float('nan'), False


def _long_to_wide_jsonl(results: list[dict]) -> list[dict]:
    """
    Convert a list of annotations in long format to wide format.
    """
    results_wide = defaultdict(lambda: {'demand': {}})

    for item in results:
        custom_id = item['custom_id']
        subdomain = item['demand']
        level = item['level']

        results_wide[custom_id]['demand'][subdomain] = {
            'level': level
        }

        if 'finish_reason' in item:
            results_wide[custom_id]['demand'][subdomain]['finish_reason'] = \
                item['finish_reason']
        if 'model_response' in item:
            results_wide[custom_id]['demand'][subdomain]['model_response'] = \
                item['model_response']
        if 'prompt' in item:
            results_wide[custom_id]['prompt'] = item['prompt']

    return [
        {'custom_id': custom_id, **entry}
        for custom_id, entry in results_wide.items()
    ]


def _make_splits_by_subdomain(results: list) -> dict:
    """Split the results by subdomain."""
    results_by_subdomain = defaultdict(list)
    for item in results:
        subdomain = item['demand']
        results_by_subdomain[subdomain].append(item)
    return results_by_subdomain


def _get_default_annotations_file_name(
        base_name: str,
        prefix: Optional[str] = None,
        finish_reason: Literal['stop', 'length', 'other'] | None = None,
        only_succeed: bool = False,
        only_failed: bool = False,
        only_levels: bool = False,
        format: Literal['long', 'wide'] = 'long',
        extension: Literal['jsonl', 'csv', 'parquet'] = 'jsonl',
    ) -> str:
    """
    Generate a default file name for the annotations file based on the
    provided parameters. The file name will follow the format:
        '[<prefix>_]<base_name>[_<finish_reason>][_succeed][_failed][_only_levels][_format].<extension>'

    where:
        - <prefix> is an optional prefix for the file name.
        - <base_name> is the base name for the file. Usually 'annotations'
            or 'annotations_<subdomain>', where <subdomain> is the name of
            the demand subdomain.
        - <finish_reason> is the finish reason, if provided.
        - [_succeed] is included if only_succeed is True.
        - [_failed] is included if only_failed is True.
        - [_only_levels] is included if only_levels is True.
        - [_format] is included if format is 'wide'.
        - <extension> is the file extension.

    Args:
        base_name (str): The base name for the file.
        prefix (str, optional): An optional prefix to add to the file name.
        finish_reason (str, optional): The finish reason to include in the file name. 
            Can be 'stop', 'length', or 'other'.
        only_succeed (bool, optional): If True, indicates that the file contains only successful annotations.
        only_failed (bool, optional): If True, indicates that the file contains only failed annotations.
        only_levels (bool, optional): If True, indicates that the file contains only demand levels.
        format (str, optional): The format of the output file. Can be 'long' or 'wide'.
        extension (str, optional): The file extension. Can be 'jsonl', 'csv', or 'parquet'.

    Returns:
        str: The generated file name.
    """
    parts = []
    if prefix:
        parts.append(prefix)
    parts.append(base_name)
    if finish_reason:
        parts.append(f'{finish_reason}')
    if only_succeed:
        parts.append('succeed')
    if only_failed:
        parts.append('failed')
    if only_levels:
        parts.append('only_levels')
    parts.append(f'{format}')
    filename = '_'.join(parts) + f'.{extension}'
    return filename
