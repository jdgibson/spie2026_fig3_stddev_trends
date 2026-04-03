#!/usr/bin/env python3
"""
Simplified Telescope Elevation Error Analysis Notebook Template

This notebook demonstrates the simplified workflow where most logic is 
encapsulated in the telescope_notebook_runner module.
"""

import json
from pathlib import Path

# Template notebook structure
template_notebook = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Setup and imports\n",
                "import sys\n",
                "import os\n",
                "from datetime import datetime\n",
                "\n",
                "# Add current directory to path\n",
                "notebook_dir = os.getcwd()\n",
                "if notebook_dir not in sys.path:\n",
                "    sys.path.insert(0, notebook_dir)\n",
                "\n",
                "print(f\"✓ Notebook directory: {notebook_dir}\")\n",
                "\n",
                "# Import the analysis module\n",
                "from telescope_notebook_runner import run_analysis\n",
                "print(f\"✓ Imported telescope_notebook_runner module\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Telescope Elevation Error Analysis\n",
                "\n",
                "This notebook performs a comprehensive analysis of telescope elevation tracking errors. Modify the dates below to analyze a specific time period."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Configuration: Specify analysis period and optional settings\n",
                "\n",
                "# DATE RANGE - Modify these dates for your analysis\n",
                "start_datetime = '2019-01-01 00:00:00'\n",
                "end_datetime = '2019-01-31 23:59:59'\n",
                "\n",
                "# Optional configuration parameters (use defaults if not specified)\n",
                "# Uncomment and modify any of these as needed:\n",
                "\n",
                "# config = {\n",
                "#     'threshold_arcsec': 4.0,\n",
                "#     'enable_query_cache': False,\n",
                "#     'skip_individual_plots': False,\n",
                "#     'skip_summary_plots': False,\n",
                "#     'enable_variance_filter': True,\n",
                "#     'variance_filter_threshold_arcsec': 2.0,\n",
                "# }\n",
                "\n",
                "# Use default configuration\n",
                "config = {}\n",
                "\n",
                "print(f\"Analysis configuration:\")\n",
                "print(f\"  Period: {start_datetime} to {end_datetime}\")\n",
                "print(f\"  Settings: {config if config else 'Using defaults'}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run the complete analysis\n",
                "print(\"\\nStarting analysis...\\n\")\n",
                "results = run_analysis(start_datetime, end_datetime, **config)\n",
                "print(\"\\n✓ Analysis complete! Results stored in 'results' variable.\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.9.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Save the template
template_path = Path('SIMPLIFIED_NOTEBOOK_TEMPLATE.ipynb')
with open(template_path, 'w') as f:
    json.dump(template_notebook, f, indent=1)

print(f"✓ Created simplified notebook template: {template_path}")
print(f"\nUsage:")
print(f"  1. Copy this template to create new analysis notebooks")
print(f"  2. Modify start_datetime and end_datetime in cell 3")
print(f"  3. Optionally customize config settings in cell 3")
print(f"  4. Run all cells to execute the analysis")
