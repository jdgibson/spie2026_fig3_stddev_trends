# Telescope Elevation Error Analysis - Architecture Refactoring

## Overview

The analysis code has been refactored to move most logic from individual notebooks into a reusable module (`telescope_notebook_runner.py`). This dramatically simplifies the notebook structure and makes it easier to maintain and extend the analysis.

## New Module Structure

### `telescope_notebook_runner.py`

A new module that encapsulates the complete analysis workflow:

**Main Class: `TelescopeElevationAnalysis`**
- Handles all configuration management
- Manages the complete analysis pipeline
- Provides clean separation of concerns
- All complex logic isolated from notebooks

**Key Methods:**
- `__init__()` - Initialize with dates and optional settings
- `run()` - Execute complete analysis pipeline
- `_query_instruments()` - Query telescope instruments
- `_analyze_data()` - Run analysis on collected data
- `_generate_plots()` - Create visualizations
- `_print_completion()` - Show completion summary

**Convenience Function:**
- `run_analysis(start_datetime, end_datetime, **kwargs)` - Simple function to run analysis

## Simplified Notebook Workflow

### Before (Complex)
Each notebook contained 12+ cells with:
1. Setup and imports
2. Parameter definitions
3. Configuration variables
4. Module imports
5. Data queries
6. Data processing
7. Analysis execution
8. Plot generation
9. Completion summary

### After (Simplified)
Each notebook now has only 4 cells:
1. **Setup** - Add path and import module
2. **Description** - Markdown title and description
3. **Configuration** - Specify start_datetime, end_datetime, and optional settings
4. **Run** - Call `run_analysis()` with dates and config

### Example Simplified Notebook

```python
# Cell 1: Setup
import sys, os
sys.path.insert(0, os.getcwd())
from telescope_notebook_runner import run_analysis

# Cell 2: Markdown
# Telescope Elevation Error Analysis

# Cell 3: Configuration
start_datetime = '2019-01-01 00:00:00'
end_datetime = '2019-01-31 23:59:59'
config = {}  # Use defaults or customize

# Cell 4: Run
results = run_analysis(start_datetime, end_datetime, **config)
```

## Configuration Parameters

All configuration is handled through kwargs to `run_analysis()`:

### Date Range (Required)
- `start_datetime` - Start date (format: 'YYYY-MM-DD HH:MM:SS')
- `end_datetime` - End date (format: 'YYYY-MM-DD HH:MM:SS')

### Database (Optional)
- `db_host` - Database host (default: 'mariadb.mmto.arizona.edu')
- `db_user` - Database user (default: 'mmtstaff')
- `db_password` - Database password (default: 'multiple')
- `db_measurements` - Database name (default: 'measurements')

### Analysis Settings (Optional)
- `threshold_arcsec` - RMS error threshold (default: 4.0)
- `min_run_time_hours` - Minimum run duration (default: 1)

### Filters (Optional)
- `enable_settling_filter` - Skip initial settling period (default: False)
- `settling_time_minutes` - Duration to skip (default: 5)
- `enable_stddev_filter` - Filter high stddev spikes (default: False)
- `stddev_spike_threshold` - Spike threshold multiplier (default: 1.5)
- `enable_stddev_absolute_filter` - Filter by absolute stddev (default: False)
- `stddev_absolute_threshold_arcsec` - Absolute threshold (default: 1e-6)
- `enable_variance_filter` - Filter high variance intervals (default: True)
- `variance_filter_threshold_arcsec` - Variance threshold (default: 2.0)
- `variance_filter_duration_seconds` - Window duration (default: 10)

### Performance (Optional)
- `enable_query_cache` - Cache database results (default: False)
- `skip_individual_plots` - Skip individual timeseries plots (default: False)
- `skip_summary_plots` - Skip summary visualizations (default: False)
- `enable_plot_downsampling` - Downsample plot data (default: True)
- `max_plot_points` - Maximum points in plots (default: 10000)

## Benefits

1. **Reduced Notebook Complexity**
   - Each notebook goes from 12+ cells to 4 cells
   - Easier to understand and modify

2. **Code Reusability**
   - All 24 notebooks share the same analysis logic
   - Bug fixes and improvements apply to all notebooks automatically

3. **Easier Maintenance**
   - Single source of truth for analysis logic
   - Easier to add new features or modify behavior

4. **Flexibility**
   - Users can customize analysis by passing config parameters
   - Default behavior works out-of-box for most cases

5. **Cleaner Notebooks**
   - Focus on data and results, not implementation details
   - Professional appearance

## Usage Example

### For Simple Analysis (Default Settings)
```python
from telescope_notebook_runner import run_analysis

results = run_analysis(
    '2019-01-01 00:00:00',
    '2019-01-31 23:59:59'
)
```

### For Customized Analysis
```python
from telescope_notebook_runner import run_analysis

results = run_analysis(
    '2019-06-01 00:00:00',
    '2019-06-30 23:59:59',
    threshold_arcsec=3.5,
    enable_variance_filter=True,
    variance_filter_threshold_arcsec=1.5,
    skip_individual_plots=False,
    skip_summary_plots=False,
    enable_query_cache=True
)
```

### For Advanced Analysis (Using Class)
```python
from telescope_notebook_runner import TelescopeElevationAnalysis

analysis = TelescopeElevationAnalysis(
    '2019-03-01 00:00:00',
    '2019-03-31 23:59:59',
    enable_settling_filter=True,
    settling_time_minutes=10,
    enable_stddev_filter=True,
    stddev_spike_threshold=2.0
)

results = analysis.run()
```

## Converting Existing Notebooks

To convert an existing complex notebook to use the new module:

1. Create new notebook with 4 cells using the template
2. Set `start_datetime` and `end_datetime` from filename
3. Set desired config options in cell 3
4. Delete old, complex notebook

## Files Created

- `telescope_notebook_runner.py` - Main module with analysis pipeline
- `SIMPLIFIED_NOTEBOOK_TEMPLATE.ipynb` - Template for new simplified notebooks
- `NOTEBOOK_REFACTORING_SUMMARY.md` - This file

## Migration Status

**Current Notebooks:**
- 24 existing notebooks (TelescopeElevationError_YYYYMM.ipynb)
- Currently using old structure with 12+ cells each
- Can be updated to use new module for simplified implementation

**Options:**
1. Keep existing notebooks - They still work fine
2. Update to new format - Use simplified structure for cleaner notebooks
3. Mix and match - Some old style, some new style (not recommended)

## Next Steps

1. Optionally update existing notebooks to use simplified format
2. All future notebooks should use the simplified template
3. Monitor for any issues with the new module
4. Gather feedback from users

## Support and Troubleshooting

If the module fails to import:
1. Ensure `telescope_notebook_runner.py` is in same directory as notebooks
2. Ensure `analyze_telescope_elevation_error.py` is available
3. Check Python path includes the notebook directory
4. Verify all required packages are installed (numpy, pandas, matplotlib, pymysql)

If analysis fails:
1. Check database credentials in config
2. Verify date range has data
3. Check error messages for specific issues
4. Review logs in CACHE_DIR
