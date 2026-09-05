"""
Reproducibility seeds, logging, and checkpoint utilities.

Public Exports:
- `aggregate_sweep_results`: Multi-seed sweep aggregation into sweep_summary.json.
- `collect_runs`: Discovery and parsing of expXX_kY_sZ run directories.
- `parse_run_directory`: Flat metric record extraction for a single run.
- `format_summary_table`: Console rendering of an aggregated summary.
"""

from .logger import (
    aggregate_sweep_results,
    collect_runs,
    format_summary_table,
    parse_run_directory,
)

__all__ = [
    "aggregate_sweep_results",
    "collect_runs",
    "format_summary_table",
    "parse_run_directory",
]
