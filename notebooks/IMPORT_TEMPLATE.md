# Notebook Import Template

Use the following cells in your notebooks to properly import from the `analyze_telescope_elevation_error` module.

## Cell 1: Set Up Paths and Imports

```python
import sys
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore')

# Add the current notebook directory to the path
notebook_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, notebook_dir)

print(f"Notebook directory: {notebook_dir}")
print(f"Python path: {sys.path[0]}")
```

## Cell 2: Import Analysis Functions

```python
# Import core analysis functions from analyze_telescope_elevation_error module
from analyze_telescope_elevation_error import (
    # Query functions
    query_all_instruments,
    query_hexapod_instrument_data,
    query_altitude_velocity_data,
    
    # Data processing functions
    identify_observing_runs,
    calculate_statistics,
    calculate_rolling_stddev,
    calculate_altitude_velocity,
    downsample_for_plotting,
    classify_altitude_direction,
    split_data_by_direction,
    
    # Filter functions
    apply_settling_filter,
    apply_stddev_filter,
    apply_stddev_spike_filter,
    apply_stddev_absolute_filter,
    apply_variance_filter,
    
    # Plotting functions
    plot_rms_analysis,
    plot_run_direction_comparison,
    plot_instrument_summary,
    plot_combined_instruments_summary,
    plot_instrument_timeseries_violin,
    plot_combined_timeseries_violin,
    plot_individual_instrument_timeseries,
    plot_combined_instruments_direction,
    plot_summary_statistics_overview,
    plot_direction_comparison,
    generate_all_plots,
    
    # Utility and formatting functions
    print_instruments_on_telescope,
    print_summary,
    create_instrument_summary_table,
    create_directional_summary_table,
    print_instrument_summary_table,
    print_directional_summary_table,
    format_duration,
    
    # Cache management
    clear_altitude_velocity_cache,
    get_altitude_velocity_cache_info,
    clear_plot_data_caches,
    
    # Analysis functions
    analyze_all_instruments_with_filters,
    query_and_process_run_data,
    export_instrument_results_to_json
)

print("✓ Successfully imported all analysis functions")
```

## Cell 3: Define Analysis Parameters (Example)

```python
# Define the analysis period - modify these dates as needed
start_datetime = '2019-01-01 00:00:00'
end_datetime = '2019-12-31 23:59:59'

# RMS error threshold in arc-seconds for filtering outliers
threshold_arcsec = 4.0
RMS_ERROR_THRESHOLD_DEGREES = threshold_arcsec / 3600.0  # Convert to degrees

# Minimum run duration (skip short runs)
MIN_RUN_TIME_HOURS = 1
MIN_RUN_TIME_SECONDS = MIN_RUN_TIME_HOURS * 3600

print(f"Analysis period: {start_datetime} to {end_datetime}")
print(f"RMS threshold: {threshold_arcsec} arc-seconds ({RMS_ERROR_THRESHOLD_DEGREES:.6f} degrees)")
```

## Usage Notes

1. **Path Setup**: The `notebook_dir` approach works for notebooks in the same directory as `analyze_telescope_elevation_error.py`
2. **Database Credentials**: The analysis functions use default database parameters (host, user, password). For custom connections, pass parameters to query functions.
3. **Caching**: Use `clear_plot_data_caches()` and `clear_altitude_velocity_cache()` to manage memory for large datasets.
4. **Filtering Options**: Each notebook can customize filtering options (settling time, stddev thresholds, etc.)
