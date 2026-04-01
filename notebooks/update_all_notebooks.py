#!/usr/bin/env python3
"""Add path setup and imports cells to all TelescopeElevationError notebooks."""

import json
from pathlib import Path
from datetime import datetime

# Define the cells to add
PATH_SETUP_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import sys\n",
        "import os\n",
        "from datetime import datetime\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import pandas as pd\n",
        "import warnings\n",
        "import gc\n",
        "\n",
        "# Suppress matplotlib warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "# Get the current working directory (notebook directory)\n",
        "notebook_dir = os.getcwd()\n",
        "if notebook_dir not in sys.path:\n",
        "    sys.path.insert(0, notebook_dir)\n",
        "\n",
        "print(f\"✓ Notebook directory: {notebook_dir}\")\n",
        "print(f\"✓ Python path: {sys.path[0]}\")"
    ]
}

IMPORTS_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Import core analysis functions from analyze_telescope_elevation_error module\n",
        "try:\n",
        "    from analyze_telescope_elevation_error import (\n",
        "        # Query functions\n",
        "        query_all_instruments,\n",
        "        query_hexapod_instrument_data,\n",
        "        query_altitude_velocity_data,\n",
        "        query_telescope_alterr_data_optimized,\n",
        "        \n",
        "        # Data processing functions\n",
        "        identify_observing_runs,\n",
        "        calculate_statistics,\n",
        "        calculate_rolling_stddev,\n",
        "        calculate_altitude_velocity,\n",
        "        downsample_for_plotting,\n",
        "        classify_altitude_direction,\n",
        "        split_data_by_direction,\n",
        "        \n",
        "        # Filter functions\n",
        "        apply_settling_filter,\n",
        "        apply_stddev_filter,\n",
        "        apply_stddev_spike_filter,\n",
        "        apply_stddev_absolute_filter,\n",
        "        apply_variance_filter,\n",
        "        \n",
        "        # Plotting functions\n",
        "        plot_rms_analysis,\n",
        "        plot_run_direction_comparison,\n",
        "        plot_instrument_summary,\n",
        "        plot_combined_instruments_summary,\n",
        "        plot_instrument_timeseries_violin,\n",
        "        plot_combined_timeseries_violin,\n",
        "        plot_individual_instrument_timeseries,\n",
        "        plot_combined_instruments_direction,\n",
        "        plot_summary_statistics_overview,\n",
        "        plot_direction_comparison,\n",
        "        \n",
        "        # Utility functions\n",
        "        print_instruments_on_telescope,\n",
        "        print_summary,\n",
        "        create_instrument_summary_table,\n",
        "        create_directional_summary_table,\n",
        "        format_duration,\n",
        "        \n",
        "        # Analysis functions\n",
        "        analyze_all_instruments_with_filters,\n",
        "        query_and_process_run_data,\n",
        "        export_instrument_results_to_json\n",
        "    )\n",
        "    print(\"✓ Successfully imported all analysis functions from analyze_telescope_elevation_error module\")\n",
        "except ImportError as e:\n",
        "    print(f\"✗ Error importing from analyze_telescope_elevation_error: {e}\")\n",
        "    print(\"  Make sure analyze_telescope_elevation_error.py is in the same directory as this notebook\")\n",
        "    raise"
    ]
}

notebooks = sorted(Path('.').glob('TelescopeElevationError_*.ipynb'))
updated_count = 0
skipped_count = 0
error_count = 0

for nb_path in notebooks:
    if nb_path.name == 'TelescopeElevationError_201901.ipynb':
        print(f"⊘ {nb_path.name}: Skipping (already updated)")
        skipped_count += 1
        continue
    
    try:
        with open(nb_path, 'r') as f:
            data = json.load(f)
        
        # Check if path setup already exists
        has_path_setup = False
        if data['cells'] and 'os.getcwd()' in ''.join(data['cells'][0].get('source', [])):
            has_path_setup = True
        
        if not has_path_setup:
            # Prepend path setup and imports cells
            data['cells'] = [PATH_SETUP_CELL, IMPORTS_CELL] + data['cells']
            
            with open(nb_path, 'w') as f:
                json.dump(data, f, indent=1)
            
            print(f"✓ {nb_path.name}: Added path setup and imports")
            updated_count += 1
        else:
            print(f"⊘ {nb_path.name}: Already has path setup")
            skipped_count += 1
            
    except Exception as e:
        print(f"✗ {nb_path.name}: Error - {e}")
        error_count += 1

print(f"\nSummary: {updated_count} updated, {skipped_count} skipped, {error_count} errors")
