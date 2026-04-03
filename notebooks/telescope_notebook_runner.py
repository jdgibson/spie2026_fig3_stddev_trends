#!/usr/bin/env python3
"""
Telescope Elevation Error Analysis Module

Encapsulates the complete analysis workflow for telescope elevation error notebooks.
Allows notebooks to run analysis by simply specifying start_datetime and end_datetime.
"""

import os
import sys
import gc
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for interactive environments
try:
    get_ipython()
    plt.ion()
    plt.rcParams['figure.figsize'] = (14, 10)
except NameError:
    pass


class TelescopeElevationAnalysis:
    """Complete pipeline for telescope elevation error analysis."""
    
    def __init__(self, start_datetime, end_datetime, **kwargs):
        """
        Initialize analysis parameters.
        
        Parameters:
        -----------
        start_datetime : str
            Start date in format 'YYYY-MM-DD HH:MM:SS'
        end_datetime : str
            End date in format 'YYYY-MM-DD HH:MM:SS'
        **kwargs : dict
            Optional configuration parameters:
            - instruments_filter: List of instrument names to process (default: None - process all)
            - threshold_arcsec: RMS error threshold (default: 4.0)
            - min_run_time_hours: Minimum run duration (default: 1)
            - db_host: Database host (default: 'mariadb.mmto.arizona.edu')
            - db_user: Database user (default: 'mmtstaff')
            - db_password: Database password (default: 'multiple')
            - db_measurements: Database name (default: 'measurements')
            - cache_dir: Cache directory (default: '.notebook_cache')
            - enable_query_cache: Enable caching (default: False)
            - skip_individual_plots: Skip individual plots (default: False)
            - skip_summary_plots: Skip summary plots (default: False)
            - enable_plot_downsampling: Enable plot downsampling (default: True)
            - max_plot_points: Max points in plots (default: 10000)
            - enable_settling_filter: Enable settling filter (default: False)
            - settling_time_minutes: Settling period (default: 5)
            - enable_stddev_filter: Enable stddev filter (default: False)
            - stddev_spike_threshold: Stddev threshold (default: 1.5)
            - stddev_filter_window_size: Window size (default: 100)
            - enable_stddev_absolute_filter: Enable absolute filter (default: False)
            - stddev_absolute_threshold_arcsec: Absolute threshold (default: 1e-6)
            - enable_variance_filter: Enable variance filter (default: True)
            - variance_filter_threshold_arcsec: Variance threshold (default: 2.0)
            - variance_filter_duration_seconds: Duration (default: 10)
            - json_output_dir: Directory for JSON output (default: 'run_statistics_json')
            - save_run_statistics_json: Save individual run stats to JSON (default: True)
        """
        
        # Date/time settings
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.start_dt = start_datetime  # Alias for compatibility
        self.end_dt = end_datetime
        self.notebook_start_time = datetime.now()
        
        # Instrument filtering
        self.INSTRUMENTS_FILTER = kwargs.get('instruments_filter', None)
        if self.INSTRUMENTS_FILTER:
            self.INSTRUMENTS_FILTER = [inst.lower() for inst in self.INSTRUMENTS_FILTER]
        
        # Extract and set parameters with defaults
        self.threshold_arcsec = kwargs.get('threshold_arcsec', 4.0)
        self.RMS_ERROR_THRESHOLD_DEGREES = self.threshold_arcsec / 3600.0
        
        # Run duration
        self.min_run_time_hours = kwargs.get('min_run_time_hours', 1)
        self.MIN_RUN_TIME_SECONDS = self.min_run_time_hours * 3600
        
        # Database
        self.DB_HOST = kwargs.get('db_host', 'mariadb.mmto.arizona.edu')
        self.DB_USER = kwargs.get('db_user', 'mmtstaff')
        self.DB_PASSWORD = kwargs.get('db_password', 'multiple')
        self.DB_MEASUREMENTS = kwargs.get('db_measurements', 'measurements')
        
        # Cache
        self.CACHE_DIR = kwargs.get('cache_dir', os.path.join(os.getcwd(), '.notebook_cache'))
        self.ENABLE_QUERY_CACHE = kwargs.get('enable_query_cache', False)
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        
        # Performance
        self.SKIP_INDIVIDUAL_PLOTS = kwargs.get('skip_individual_plots', False)
        self.SKIP_SUMMARY_PLOTS = kwargs.get('skip_summary_plots', False)
        self.ENABLE_PLOT_DOWNSAMPLING = kwargs.get('enable_plot_downsampling', True)
        self.MAX_PLOT_POINTS = kwargs.get('max_plot_points', 10000)
        
        # Filters
        self.ENABLE_SETTLING_FILTER = kwargs.get('enable_settling_filter', False)
        self.SETTLING_TIME_MINUTES = kwargs.get('settling_time_minutes', 5)
        
        self.ENABLE_STDDEV_FILTER = kwargs.get('enable_stddev_filter', False)
        self.STDDEV_SPIKE_THRESHOLD = kwargs.get('stddev_spike_threshold', 1.5)
        self.STDDEV_FILTER_WINDOW_SIZE = kwargs.get('stddev_filter_window_size', 100)
        
        self.ENABLE_STDDEV_ABSOLUTE_FILTER = kwargs.get('enable_stddev_absolute_filter', False)
        self.STDDEV_ABSOLUTE_THRESHOLD_ARCSEC = kwargs.get('stddev_absolute_threshold_arcsec', 1e-6)
        
        self.ENABLE_VARIANCE_FILTER = kwargs.get('enable_variance_filter', True)
        self.VARIANCE_FILTER_THRESHOLD_ARCSEC = kwargs.get('variance_filter_threshold_arcsec', 2.0)
        self.VARIANCE_FILTER_DURATION_SECONDS = kwargs.get('variance_filter_duration_seconds', 10)
        
        # JSON output directory for individual run statistics
        self.JSON_OUTPUT_DIR = kwargs.get('json_output_dir', os.path.join(os.getcwd(), 'json'))
        self.SAVE_RUN_STATISTICS_JSON = kwargs.get('save_run_statistics_json', True)
        if self.SAVE_RUN_STATISTICS_JSON:
            os.makedirs(self.JSON_OUTPUT_DIR, exist_ok=True)
        
        # Import analysis functions
        self._import_analysis_functions()
        
        # Results storage
        self.instruments_dict = {}
        self.observing_runs_dict = {}
        self.instrument_stats_dict = {}
        self.cached_data = None
    
    def _import_analysis_functions(self):
        """Import all required analysis functions."""
        try:
            from analyze_telescope_elevation_error import (
                query_all_instruments,
                query_hexapod_instrument_data,
                identify_observing_runs,
                analyze_all_instruments_with_filters,
                plot_individual_instrument_timeseries,
                plot_summary_statistics_overview,
                print_instruments_on_telescope,
                get_cached_results,
                save_to_cache,
            )
            
            self.query_all_instruments = query_all_instruments
            self.query_hexapod_instrument_data = query_hexapod_instrument_data
            self.identify_observing_runs = identify_observing_runs
            self.analyze_all_instruments_with_filters = analyze_all_instruments_with_filters
            self.plot_individual_instrument_timeseries = plot_individual_instrument_timeseries
            self.plot_summary_statistics_overview = plot_summary_statistics_overview
            self.print_instruments_on_telescope = print_instruments_on_telescope
            self.get_cached_results = get_cached_results
            self.save_to_cache = save_to_cache
            
        except ImportError as e:
            raise ImportError(f"Failed to import analysis functions: {e}")
    
    def print_header(self, text):
        """Print a formatted header."""
        print("\n" + "="*80)
        print(f"  {text}")
        print("="*80)
    
    def run(self):
        """Execute the complete analysis pipeline."""
        
        self.print_header("TELESCOPE ELEVATION ERROR ANALYSIS")
        print(f"Start DateTime: {self.notebook_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Analysis period: {self.start_datetime} to {self.end_datetime}\n")
        
        # Step 1: Query instruments
        self._query_instruments()
        
        # Step 2: Build observing runs and analyze
        self._analyze_data()
        
        # Step 3: Save individual run statistics to JSON
        self._save_individual_run_statistics()
        
        # Step 4: Generate plots
        self._generate_plots()
        
        # Step 5: Print summary
        self._print_completion()
        
        return self.instrument_stats_dict
    
    def _query_instruments(self):
        """Query all instruments from database."""
        
        print("Querying all instruments on telescope during analysis period...\n")
        
        # Try to load from cache
        self.cached_data = self.get_cached_results(
            self.CACHE_DIR, 
            self.start_dt, 
            self.end_dt, 
            enable_cache=self.ENABLE_QUERY_CACHE
        )
        
        if self.cached_data is not None:
            self.instruments_dict = self.cached_data.get('instruments_dict', {})
            self.observing_runs_dict = self.cached_data.get('observing_runs_dict', {})
            self.instrument_stats_dict = self.cached_data.get('instrument_stats_dict', {})
            print("✓ Analysis results loaded from cache - skipping database queries\n")
            return
        
        try:
            self.instruments_dict = self.query_all_instruments(
                host=self.DB_HOST,
                user=self.DB_USER,
                password=self.DB_PASSWORD,
                database=self.DB_MEASUREMENTS,
                start_datetime=self.start_datetime,
                end_datetime=self.end_datetime
            )
            
            # Filter instruments if specified
            if self.INSTRUMENTS_FILTER:
                filtered_dict = {}
                for inst_name, inst_data in self.instruments_dict.items():
                    if inst_name.lower() in self.INSTRUMENTS_FILTER:
                        filtered_dict[inst_name] = inst_data
                self.instruments_dict = filtered_dict
                
                if self.INSTRUMENTS_FILTER:
                    print(f"Instruments filter applied: {', '.join(self.INSTRUMENTS_FILTER)}\n")
            
            self.print_instruments_on_telescope(self.instruments_dict)
            
            if not self.instruments_dict or len(self.instruments_dict) == 0:
                print("\n✗ No instruments found for the analysis period.")
                return
                
        except Exception as e:
            print(f"✗ Error querying instruments: {e}")
            self.instruments_dict = {}
    
    def _analyze_data(self):
        """Build observing runs and run analysis pipeline."""
        
        if self.cached_data is not None:
            # Already loaded from cache
            return
        
        if not self.instruments_dict:
            print("✗ Cannot proceed - no instruments found")
            return
        
        # Build observing runs
        self.observing_runs_dict = {}
        print("\nPreparing observing runs data...")
        
        for instrument_name in sorted(self.instruments_dict.keys()):
            try:
                instrument_dates = self.query_hexapod_instrument_data(
                    host=self.DB_HOST,
                    user=self.DB_USER,
                    password=self.DB_PASSWORD,
                    database=self.DB_MEASUREMENTS,
                    start_datetime=self.start_dt,
                    end_datetime=self.end_dt,
                    instrument_name=instrument_name
                )
                
                if len(instrument_dates) > 0:
                    observing_runs = self.identify_observing_runs(instrument_dates, min_gap_hours=1)
                    self.observing_runs_dict[instrument_name] = observing_runs
                    print(f"  ✓ {instrument_name.upper()}: {len(observing_runs)} observing runs")
                else:
                    self.observing_runs_dict[instrument_name] = []
                    print(f"  - {instrument_name.upper()}: No observing runs")
                    
            except Exception as e:
                print(f"  ✗ {instrument_name.upper()}: {str(e)}")
                self.observing_runs_dict[instrument_name] = []
            finally:
                gc.collect()
        
        # Run analysis pipeline
        print("\nRunning comprehensive analysis pipeline...")
        self.instrument_stats_dict = self.analyze_all_instruments_with_filters(
            self.instruments_dict,
            self.observing_runs_dict,
            self.start_dt,
            self.end_dt,
            self.RMS_ERROR_THRESHOLD_DEGREES,
            db_host=self.DB_HOST,
            db_user=self.DB_USER,
            db_password=self.DB_PASSWORD,
            min_run_duration_seconds=self.MIN_RUN_TIME_SECONDS,
            settling_filter_enabled=self.ENABLE_SETTLING_FILTER,
            settling_minutes=self.SETTLING_TIME_MINUTES,
            stddev_filter_enabled=self.ENABLE_STDDEV_FILTER,
            stddev_threshold=self.STDDEV_SPIKE_THRESHOLD,
            stddev_window=self.STDDEV_FILTER_WINDOW_SIZE,
            stddev_absolute_filter_enabled=self.ENABLE_STDDEV_ABSOLUTE_FILTER,
            stddev_absolute_threshold_arcsec=self.STDDEV_ABSOLUTE_THRESHOLD_ARCSEC,
            variance_filter_enabled=self.ENABLE_VARIANCE_FILTER,
            variance_filter_threshold_arcsec=self.VARIANCE_FILTER_THRESHOLD_ARCSEC,
            variance_filter_duration_seconds=self.VARIANCE_FILTER_DURATION_SECONDS
        )
        
        # Save to cache
        self.save_to_cache(
            {
                'instruments_dict': self.instruments_dict,
                'observing_runs_dict': self.observing_runs_dict,
                'instrument_stats_dict': self.instrument_stats_dict
            },
            self.CACHE_DIR,
            self.start_dt,
            self.end_dt,
            enable_cache=self.ENABLE_QUERY_CACHE
        )
    
    def _generate_plots(self):
        """Generate analysis plots."""
        
        if not self.instrument_stats_dict:
            print("\n✗ No data to plot")
            return
        
        # Individual plots
        if not self.SKIP_INDIVIDUAL_PLOTS:
            self.print_header("GENERATING INDIVIDUAL INSTRUMENT TIME-SERIES PLOTS")
            
            try:
                self.plot_individual_instrument_timeseries(
                    self.instrument_stats_dict,
                    self.start_dt,
                    self.end_dt,
                    show_altitude=True,
                    db_host=self.DB_HOST,
                    db_user=self.DB_USER,
                    db_password=self.DB_PASSWORD,
                    db_measurements=self.DB_MEASUREMENTS,
                    enable_downsampling=self.ENABLE_PLOT_DOWNSAMPLING,
                    downsample_max_points=self.MAX_PLOT_POINTS
                )
                print("✓ Individual instrument plots complete")
            except Exception as e:
                print(f"✗ Error generating individual plots: {e}")
        
        # Summary plots
        if not self.SKIP_SUMMARY_PLOTS:
            self.print_header("GENERATING SUMMARY RESULTS VISUALIZATIONS")
            
            try:
                self.plot_summary_statistics_overview(
                    self.instrument_stats_dict,
                    self.start_dt,
                    self.end_dt
                )
                print("✓ Summary plots complete")
            except Exception as e:
                print(f"✗ Error generating summary plots: {e}")
    
    def _save_individual_run_statistics(self):
        """Save statistics for each individual run to separate JSON files.
        
        PRIMARY METRIC: Rolling standard deviations of elevation errors.
        SECONDARY METRIC: Raw elevation error statistics for reference.
        """
        
        if not self.SAVE_RUN_STATISTICS_JSON or not self.instrument_stats_dict:
            return
        
        self.print_header("SAVING INDIVIDUAL RUN STATISTICS TO JSON")
        
        total_runs_saved = 0
        
        for instrument_name in sorted(self.instrument_stats_dict.keys()):
            instrument_runs = self.instrument_stats_dict[instrument_name]
            
            for run_data in instrument_runs:
                try:
                    run_number = run_data.get('run_number', 0)
                    run_start_dt = run_data.get('run_start_dt')
                    run_end_dt = run_data.get('run_end_dt')
                    
                    if not run_start_dt or not run_end_dt:
                        print(f"✗ {instrument_name} Run {run_number}: Missing start/end datetime")
                        continue
                    
                    # Format datetimes for filename: YYYYMMDD_HHMMSS
                    start_str = run_start_dt.strftime('%Y%m%d_%H%M%S')
                    end_str = run_end_dt.strftime('%Y%m%d_%H%M%S')
                    
                    # Create filename: instrument_run_date_start_end.json
                    filename = f"{instrument_name}_{start_str}_to_{end_str}.json"
                    filepath = os.path.join(self.JSON_OUTPUT_DIR, filename)
                    
                    # Prepare data for JSON serialization with organized structure by direction
                    stats_both = run_data.get('stats', {})
                    stats_up = run_data.get('stats_up', {})
                    stats_down = run_data.get('stats_down', {})
                    
                    json_data = {
                        'metadata': {
                            'instrument': instrument_name,
                            'run_number': run_number,
                            'run_start_datetime': run_start_dt.isoformat(),
                            'run_end_datetime': run_end_dt.isoformat(),
                            'run_duration_seconds': (run_end_dt - run_start_dt).total_seconds(),
                            'primary_metric': 'Standard Deviation of Elevation Error',
                            'secondary_metric': 'RMS (Root Mean Square) of Elevation Error',
                            'analysis_note': 'This report emphasizes the PRIMARY METRIC (standard deviation of elevation error) to characterize pointing stability and variability. The SECONDARY METRIC (RMS elevation error) is provided for reference and historical comparison.',
                        },
                        'data_quality': {
                            'original_rms_record_count': run_data.get('original_count', 0),
                            'filtered_rms_record_count': run_data.get('filtered_count', 0),
                            'data_retention_percent': run_data.get('retention_percent', 0),
                        },
                        'statistics_by_direction': {
                            'BOTH': {
                                'description': 'Combined statistics for all telescope directions',
                                'standard_deviation': {
                                    'description': 'PRIMARY METRIC: Standard Deviation of Elevation Error - Quantifies pointing stability and variability',
                                    'metric_definition': 'Rolling standard deviation calculated from elevation error time series',
                                    'mean_stddev_arcsec': stats_both.get('rolling_stddev_statistics', {}).get('mean', None),
                                    'std_stddev_arcsec': stats_both.get('rolling_stddev_statistics', {}).get('std', None),
                                    'min_stddev_arcsec': stats_both.get('rolling_stddev_statistics', {}).get('min', None),
                                    'max_stddev_arcsec': stats_both.get('rolling_stddev_statistics', {}).get('max', None),
                                    'data_points': stats_both.get('rolling_stddev_statistics', {}).get('data_points', None),
                                    'total_change_arcsec': stats_both.get('rolling_stddev_statistics', {}).get('total_change', None),
                                    'percent_change': stats_both.get('rolling_stddev_statistics', {}).get('percent_change', None),
                                    'boxplot_stats': stats_both.get('rolling_stddev_statistics', {}).get('boxplot_stats', None),
                                },
                                'elevation_error': {
                                    'description': 'SECONDARY METRIC: Elevation RMS Error - Raw pointing error magnitude for reference',
                                    'metric_definition': 'Root mean square of elevation error measurements (traditional metric)',
                                    'mean_rms_arcsec': stats_both.get('elevation_error_statistics', {}).get('mean', None),
                                    'std_rms_arcsec': stats_both.get('elevation_error_statistics', {}).get('std', None),
                                    'min_rms_arcsec': stats_both.get('elevation_error_statistics', {}).get('min', None),
                                    'max_rms_arcsec': stats_both.get('elevation_error_statistics', {}).get('max', None),
                                    'data_points': stats_both.get('elevation_error_statistics', {}).get('data_points', None),
                                    'total_change_arcsec': stats_both.get('elevation_error_statistics', {}).get('total_change', None),
                                    'percent_change': stats_both.get('elevation_error_statistics', {}).get('percent_change', None),
                                    'boxplot_stats': stats_both.get('elevation_error_statistics', {}).get('boxplot_stats', None),
                                },
                            },
                            'UP': {
                                'description': 'Statistics for upward telescope motion',
                                'standard_deviation': {
                                    'description': 'PRIMARY METRIC: Standard Deviation of Elevation Error (Upward) - Pointing stability during elevation increase',
                                    'metric_definition': 'Rolling standard deviation calculated from elevation error time series',
                                    'mean_stddev_arcsec': stats_up.get('rolling_stddev_statistics', {}).get('mean', None),
                                    'std_stddev_arcsec': stats_up.get('rolling_stddev_statistics', {}).get('std', None),
                                    'min_stddev_arcsec': stats_up.get('rolling_stddev_statistics', {}).get('min', None),
                                    'max_stddev_arcsec': stats_up.get('rolling_stddev_statistics', {}).get('max', None),
                                    'data_points': stats_up.get('rolling_stddev_statistics', {}).get('data_points', None),
                                    'total_change_arcsec': stats_up.get('rolling_stddev_statistics', {}).get('total_change', None),
                                    'percent_change': stats_up.get('rolling_stddev_statistics', {}).get('percent_change', None),
                                    'boxplot_stats': stats_up.get('rolling_stddev_statistics', {}).get('boxplot_stats', None),
                                },
                                'elevation_error': {
                                    'description': 'SECONDARY METRIC: Elevation RMS Error (Upward) - Raw error measurement during elevation increase',
                                    'metric_definition': 'Root mean square of elevation error measurements (traditional metric)',
                                    'mean_rms_arcsec': stats_up.get('elevation_error_statistics', {}).get('mean', None),
                                    'std_rms_arcsec': stats_up.get('elevation_error_statistics', {}).get('std', None),
                                    'min_rms_arcsec': stats_up.get('elevation_error_statistics', {}).get('min', None),
                                    'max_rms_arcsec': stats_up.get('elevation_error_statistics', {}).get('max', None),
                                    'data_points': stats_up.get('elevation_error_statistics', {}).get('data_points', None),
                                    'total_change_arcsec': stats_up.get('elevation_error_statistics', {}).get('total_change', None),
                                    'percent_change': stats_up.get('elevation_error_statistics', {}).get('percent_change', None),
                                    'boxplot_stats': stats_up.get('elevation_error_statistics', {}).get('boxplot_stats', None),
                                },
                            },
                            'DOWN': {
                                'description': 'Statistics for downward telescope motion',
                                'standard_deviation': {
                                    'description': 'PRIMARY METRIC: Standard Deviation of Elevation Error (Downward) - Pointing stability during elevation decrease',
                                    'metric_definition': 'Rolling standard deviation calculated from elevation error time series',
                                    'mean_stddev_arcsec': stats_down.get('rolling_stddev_statistics', {}).get('mean', None),
                                    'std_stddev_arcsec': stats_down.get('rolling_stddev_statistics', {}).get('std', None),
                                    'min_stddev_arcsec': stats_down.get('rolling_stddev_statistics', {}).get('min', None),
                                    'max_stddev_arcsec': stats_down.get('rolling_stddev_statistics', {}).get('max', None),
                                    'data_points': stats_down.get('rolling_stddev_statistics', {}).get('data_points', None),
                                    'total_change_arcsec': stats_down.get('rolling_stddev_statistics', {}).get('total_change', None),
                                    'percent_change': stats_down.get('rolling_stddev_statistics', {}).get('percent_change', None),
                                    'boxplot_stats': stats_down.get('rolling_stddev_statistics', {}).get('boxplot_stats', None),
                                },
                                'elevation_error': {
                                    'description': 'SECONDARY METRIC: Elevation RMS Error (Downward) - Raw error measurement during elevation decrease',
                                    'metric_definition': 'Root mean square of elevation error measurements (traditional metric)',
                                    'mean_rms_arcsec': stats_down.get('elevation_error_statistics', {}).get('mean', None),
                                    'std_rms_arcsec': stats_down.get('elevation_error_statistics', {}).get('std', None),
                                    'min_rms_arcsec': stats_down.get('elevation_error_statistics', {}).get('min', None),
                                    'max_rms_arcsec': stats_down.get('elevation_error_statistics', {}).get('max', None),
                                    'data_points': stats_down.get('elevation_error_statistics', {}).get('data_points', None),
                                    'total_change_arcsec': stats_down.get('elevation_error_statistics', {}).get('total_change', None),
                                    'percent_change': stats_down.get('elevation_error_statistics', {}).get('percent_change', None),
                                    'boxplot_stats': stats_down.get('elevation_error_statistics', {}).get('boxplot_stats', None),
                                },
                            },
                        },
                    }
                    
                    # Convert numpy data types to native Python types for JSON serialization
                    json_data = self._convert_numpy_types(json_data)
                    
                    # Write JSON file
                    with open(filepath, 'w') as f:
                        json.dump(json_data, f, indent=2, default=str)
                    
                    print(f"✓ {instrument_name} Run {run_number}: {filename}")
                    total_runs_saved += 1
                    
                except Exception as e:
                    print(f"✗ {instrument_name} Run {run_number}: Error saving JSON - {str(e)}")
        
        print(f"\n✓ Saved statistics for {total_runs_saved} run(s) to {self.JSON_OUTPUT_DIR}\n")
    
    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (pd.DataFrame, pd.Index)):
            return obj.to_dict()
        else:
            return obj
    
    def _print_completion(self):
        """Print completion summary."""
        
        notebook_end_time = datetime.now()
        elapsed_time = notebook_end_time - self.notebook_start_time
        
        self.print_header("ANALYSIS COMPLETED")
        print(f"Start DateTime: {self.notebook_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End DateTime:   {notebook_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        hours = int(elapsed_time.total_seconds() // 3600)
        minutes = int((elapsed_time.total_seconds() % 3600) // 60)
        seconds = int(elapsed_time.total_seconds() % 60)
        print(f"Elapsed Time:   {hours}h {minutes}m {seconds}s")
        print("="*80 + "\n")


def run_analysis(start_datetime, end_datetime, **kwargs):
    """
    Convenience function to run analysis.
    
    Parameters:
    -----------
    start_datetime : str
        Start date in format 'YYYY-MM-DD HH:MM:SS'
    end_datetime : str
        End date in format 'YYYY-MM-DD HH:MM:SS'
    **kwargs : dict
        Optional configuration parameters (see TelescopeElevationAnalysis.__init__)
    
    Returns:
    --------
    dict : Results dictionary with instrument statistics
    """
    
    analysis = TelescopeElevationAnalysis(start_datetime, end_datetime, **kwargs)
    return analysis.run()
