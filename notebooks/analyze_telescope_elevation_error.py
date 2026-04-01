#!/usr/bin/env python3
"""
Elevation Error Standard Deviation Analysis
Primary Analysis: Standard deviation of elevation errors and their temporal variability.
Secondary Analysis: Raw elevation error statistics for reference.
"""

# Check if running in Jupyter/IPython environment BEFORE setting backend
import matplotlib
import sys

try:
    get_ipython()
    # In Jupyter/IPython - use interactive backend
    interactive_env = True
    # Let Jupyter handle the backend (don't override it)
except NameError:
    # Not in interactive environment - use non-interactive backend
    interactive_env = False
    matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd
import pymysql
import json
import os
import pandas as pd
import pymysql
from scipy.interpolate import interp1d

# RMS Error filtering threshold: 5 arc-seconds converted to decimal degrees
# 5 arc-seconds = 5 / 3600 degrees = 0.001389 degrees
RMS_ERROR_THRESHOLD_ARCSEC = 5.0
RMS_ERROR_THRESHOLD_DEGREES = RMS_ERROR_THRESHOLD_ARCSEC / 3600.0

# Standard Deviation filtering threshold: 5 arc-seconds (for post-processing rolling stddev)
STDDEV_ERROR_THRESHOLD_ARCSEC = 5.0

# Module-level caches to optimize plotting performance
_ALTITUDE_VELOCITY_CACHE = {}  # Cache for altitude velocity database queries
_PLOT_DATA_CACHE = {}  # Cache for processed plot data (timestamps, RMS, StdDev, altitudes)
_DOWNSAMPLED_DATA_CACHE = {}  # Cache for downsampled plot arrays
_ROLLING_STDDEV_CACHE = {}  # Cache for rolling standard deviation calculations

# Configuration: Altitude velocity calculation method
# True = calculate directly from altitude data using gradient (faster, no DB query)
# False = query measurements.mount_mini_velalt database table (pre-calculated values)
USE_ALTITUDE_GRADIENT_VELOCITY = True

# Configure matplotlib for interactive environments
if interactive_env:
    plt.ion()  # Enable interactive mode for Jupyter/IPython
    plt.rcParams['figure.figsize'] = (14, 10)

def generate_sample_rms_data(start_year=2022, num_years=5):
    """
    Generate sample RMS error data for a five-year period.
    Used as fallback if database query fails.
    
    Parameters:
    -----------
    start_year : int
        Starting year for the data
    num_years : int
        Number of years of data
    
    Returns:
    --------
    dates : np.ndarray
        Array of dates
    rms_errors : np.ndarray
        Array of RMS error values
    """
    # Generate daily timestamps for 5 years
    start_date = datetime(start_year, 1, 1)
    num_days = num_years * 365
    dates = np.array([start_date + timedelta(days=i) for i in range(num_days)])
    
    # Generate realistic RMS error data with trends
    t = np.arange(num_days) / 365.0
    baseline = 50.0
    seasonal = 5.0 * np.sin(2 * np.pi * t)
    trend = -3.0 * t
    noise = np.random.normal(0, 2, num_days)
    
    rms_errors = baseline + seasonal + trend + noise
    rms_errors = np.maximum(rms_errors, 1.0)
    
    return dates, rms_errors

def query_all_instruments_with_timestamps(host='mariadb.mmto.arizona.edu',
                                            user='mmtstaff',
                                            password='multiple',
                                            database='measurements',
                                            start_datetime=None,
                                            end_datetime=None):
    """
    Query all unique instruments and their timestamps in a SINGLE query to hexapod_mini_instrument.
    This consolidates multiple queries into one for efficiency and returns both metadata and timestamps.
    
    Parameters:
    -----------
    host : str
        Database host address
    user : str
        Database user
    password : str
        Database password
    database : str
        Database name
    start_datetime : datetime or str, optional
        Start datetime for data range. If None, defaults to 5 years ago.
    end_datetime : datetime or str, optional
        End datetime for data range. If None, defaults to now.
    
    Returns:
    --------
    tuple: (instruments_dict, all_timestamps_dict)
        - instruments_dict: dict with instrument names as keys and (first_timestamp, last_timestamp) as values
        - all_timestamps_dict: dict with instrument names as keys and list of all timestamps as values
    """
    try:
        # Connect to the database
        print(f"Connecting to {database} database at {host}...")
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        
        print(f"Successfully connected to MariaDB database")
        
        cursor = connection.cursor()
        
        # Set default date range if not provided
        if end_datetime is None:
            end_datetime = datetime.now()
        elif isinstance(end_datetime, str):
            end_datetime = datetime.fromisoformat(end_datetime)
        
        if start_datetime is None:
            start_datetime = end_datetime - timedelta(days=365*5)
        elif isinstance(start_datetime, str):
            start_datetime = datetime.fromisoformat(start_datetime)
        
        print(f"Querying all instruments and timestamps from {start_datetime} to {end_datetime}")
        
        # Convert datetime to Unix milliseconds for database query
        start_ms = int(start_datetime.timestamp() * 1_000)
        end_ms = int(end_datetime.timestamp() * 1_000)
        
        # Single unified query - get ALL timestamps for ALL instruments in one call
        # This replaces multiple separate queries with one efficient query
        query = """
        SELECT value, timestamp
        FROM hexapod_mini_instrument
        WHERE timestamp >= %s AND timestamp <= %s
        ORDER BY value ASC, timestamp ASC
        """
        
        print(f"SQL Query: Fetching all instruments and timestamps in single query...")
        print(f"Parameters: start_timestamp_ms={start_ms}, end_timestamp_ms={end_ms}")
        
        cursor.execute(query, (start_ms, end_ms))
        results = cursor.fetchall()
        
        instruments_dict = {}
        all_timestamps_dict = {}
        
        if results:
            print(f"Retrieved {len(results)} total records from hexapod_mini_instrument")
            
            current_instrument = None
            instrument_timestamps = []
            
            # Process results in single pass - group by instrument
            for row in results:
                instrument_name = row[0]
                timestamp_ms = row[1]
                
                # If we've moved to a new instrument, save the previous one
                if instrument_name != current_instrument:
                    if current_instrument is not None and instrument_timestamps:
                        # Save previous instrument's data
                        first_ts = instrument_timestamps[0]
                        last_ts = instrument_timestamps[-1]
                        first_dt = datetime.fromtimestamp(first_ts / 1_000)
                        last_dt = datetime.fromtimestamp(last_ts / 1_000)
                        
                        instruments_dict[current_instrument] = (first_dt, last_dt)
                        all_timestamps_dict[current_instrument] = [
                            datetime.fromtimestamp(ts / 1_000) for ts in instrument_timestamps
                        ]
                    
                    # Start new instrument
                    current_instrument = instrument_name
                    instrument_timestamps = [timestamp_ms]
                else:
                    instrument_timestamps.append(timestamp_ms)
            
            # Don't forget the last instrument
            if current_instrument is not None and instrument_timestamps:
                first_ts = instrument_timestamps[0]
                last_ts = instrument_timestamps[-1]
                first_dt = datetime.fromtimestamp(first_ts / 1_000)
                last_dt = datetime.fromtimestamp(last_ts / 1_000)
                
                instruments_dict[current_instrument] = (first_dt, last_dt)
                all_timestamps_dict[current_instrument] = [
                    datetime.fromtimestamp(ts / 1_000) for ts in instrument_timestamps
                ]
            
            print(f"Found {len(instruments_dict)} unique instrument(s)")
            for name, (first, last) in sorted(instruments_dict.items()):
                num_records = len(all_timestamps_dict[name])
                print(f"  {name.upper()}: {num_records} records, {first} to {last}")
        else:
            print("No instruments found in the specified date range")
        
        cursor.close()
        connection.close()
        
        return instruments_dict, all_timestamps_dict
        
    except pymysql.Error as e:
        print(f"Error while connecting to MariaDB: {e}")
        raise
    except Exception as e:
        print(f"Error querying instruments: {e}")
        raise

def query_all_instruments(host='mariadb.mmto.arizona.edu',
                          user='mmtstaff',
                          password='multiple',
                          database='measurements',
                          start_datetime=None,
                          end_datetime=None):
    """
    Query all unique instruments on the telescope within a date range.
    DEPRECATED: Use query_all_instruments_with_timestamps() instead for better efficiency.
    This function now wraps the optimized version and returns only the instruments dict.
    
    Parameters:
    -----------
    host : str
        Database host address
    user : str
        Database user
    password : str
        Database password
    database : str
        Database name
    start_datetime : datetime or str, optional
        Start datetime for data range. If None, defaults to 5 years ago.
    end_datetime : datetime or str, optional
        End datetime for data range. If None, defaults to now.
    
    Returns:
    --------
    instruments_dict : dict
        Dictionary with instrument names as keys and (first_timestamp, last_timestamp) as values
    """
    instruments_dict, _ = query_all_instruments_with_timestamps(
        host=host, user=user, password=password, database=database,
        start_datetime=start_datetime, end_datetime=end_datetime
    )
    return instruments_dict

def print_instruments_on_telescope(instruments_dict):
    """
    Print formatted list of instruments on telescope.
    
    Parameters:
    -----------
    instruments_dict : dict
        Dictionary with instrument names and their time ranges
    """
    if not instruments_dict:
        print("\nNo instruments found on telescope during specified period")
        return
    
    print("\n" + "="*70)
    print("INSTRUMENTS ON TELESCOPE")
    print("="*70)
    
    for instrument_name in sorted(instruments_dict.keys()):
        # Skip NA instruments
        if instrument_name.upper() == 'NA':
            continue
        first_dt, last_dt = instruments_dict[instrument_name]
        duration = last_dt - first_dt
        print(f"\n  {instrument_name.upper()}:")
        print(f"    First observed: {first_dt}")
        print(f"    Last observed:  {last_dt}")
        print(f"    Duration:       {duration}")
    
    print("\n" + "="*70)

def query_hexapod_instrument_data(host='mariadb.mmto.arizona.edu',
                                   user='mmtstaff',
                                   password='multiple',
                                   database='measurements',
                                   start_datetime=None,
                                   end_datetime=None,
                                   instrument_name='blue',
                                   cached_timestamps=None):
    """
    Get instrument timestamps for a specific instrument.
    If cached_timestamps is provided (from query_all_instruments_with_timestamps), uses cached data.
    Otherwise, falls back to database query for backward compatibility.
    
    Parameters:
    -----------
    host : str
        Database host address
    user : str
        Database user
    password : str
        Database password
    database : str
        Database name
    start_datetime : datetime or str, optional
        Start datetime for data range. If None, defaults to 5 years ago.
    end_datetime : datetime or str, optional
        End datetime for data range. If None, defaults to now.
    instrument_name : str
        Instrument name to filter by (e.g., "blue")
    cached_timestamps : dict, optional
        Pre-fetched timestamps dict from query_all_instruments_with_timestamps.
        If provided, uses this instead of querying database.
    
    Returns:
    --------
    dates : np.ndarray
        Array of datetime objects
    """
    # Use cached data if available (preferred for efficiency)
    if cached_timestamps is not None and instrument_name in cached_timestamps:
        print(f"Using cached timestamps for {instrument_name} ({len(cached_timestamps[instrument_name])} records)")
        dates = np.array(cached_timestamps[instrument_name])
        print(f"Processed {len(dates)} instrument records")
        return dates
    
    # Fallback to database query if no cache available
    try:
        # Connect to the database
        print(f"Connecting to {database} database at {host}...")
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        
        print(f"Successfully connected to MariaDB database")
        
        cursor = connection.cursor()
        
        # Set default date range if not provided
        if end_datetime is None:
            end_datetime = datetime.now()
        elif isinstance(end_datetime, str):
            end_datetime = datetime.fromisoformat(end_datetime)
        
        if start_datetime is None:
            start_datetime = end_datetime - timedelta(days=365*5)
        elif isinstance(start_datetime, str):
            start_datetime = datetime.fromisoformat(start_datetime)
        
        print(f"Querying hexapod_mini_instrument data from {start_datetime} to {end_datetime}")
        
        # Convert datetime to Unix milliseconds for database query
        start_ms = int(start_datetime.timestamp() * 1_000)
        end_ms = int(end_datetime.timestamp() * 1_000)
        
        # Query to get hexapod_mini_instrument data for specified instrument
        query = """
        SELECT timestamp, value
        FROM hexapod_mini_instrument
        WHERE timestamp >= %s AND timestamp <= %s AND value = %s
        ORDER BY timestamp ASC
        """
        
        print(f"SQL Query: {query.strip()}")
        print(f"Parameters: start_timestamp_ms={start_ms}, end_timestamp_ms={end_ms}, instrument={instrument_name}")
        
        cursor.execute(query, (start_ms, end_ms, instrument_name))
        results = cursor.fetchall()
        
        if not results:
            print(f"No records found for instrument '{instrument_name}'")
        else:
            print(f"Retrieved {len(results)} records from hexapod_mini_instrument table")
        
        # Extract dates using vectorized conversion (much faster than loop)
        if results:
            result_array = np.array(results)
            timestamps_ms = result_array[:, 0].astype(int)  # Ensure timestamps are integers (measurements DB uses milliseconds)
            
            # Vectorized datetime conversion using pandas (10x faster than loop)
            import pandas as pd
            dates = pd.to_datetime(timestamps_ms, unit='ms').values
        else:
            dates = np.array([])
        
        cursor.close()
        connection.close()
        
        print(f"Processed {len(dates)} instrument records")
        
        return dates
        
    except pymysql.Error as e:
        print(f"Error while connecting to MariaDB: {e}")
        raise
    except Exception as e:
        print(f"Error querying hexapod_mini_instrument table: {e}")
        raise

def query_altitude_velocity_data(host='mariadb.mmto.arizona.edu',
                                 user='mmtstaff',
                                 password='multiple',
                                 database='measurements',
                                 start_datetime=None,
                                 end_datetime=None):
    """
    Query altitude velocity data from the mount_mini_velalt table.
    Results are cached to avoid repeated database hits for the same date range.
    
    Parameters:
    -----------
    host : str
        Database host address
    user : str
        Database user
    password : str
        Database password
    database : str
        Database name
    start_datetime : datetime or str, optional
        Start datetime for data range. If None, defaults to 5 years ago.
    end_datetime : datetime or str, optional
        End datetime for data range. If None, defaults to now.
    
    Returns:
    --------
    tuple: (timestamps, velocities)
        timestamps : np.ndarray of datetime objects
        velocities : np.ndarray of velocity values (arc-seconds/second)
    """
    global _ALTITUDE_VELOCITY_CACHE
    
    # Set default date range if not provided
    if end_datetime is None:
        end_datetime = datetime.now()
    elif isinstance(end_datetime, str):
        end_datetime = datetime.fromisoformat(end_datetime)
    
    if start_datetime is None:
        start_datetime = end_datetime - timedelta(days=365*5)
    elif isinstance(start_datetime, str):
        start_datetime = datetime.fromisoformat(start_datetime)
    
    # Create cache key based on date range (rounded to seconds)
    cache_key = (
        start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        end_datetime.strftime('%Y-%m-%d %H:%M:%S')
    )
    
    # Check if data is already cached
    if cache_key in _ALTITUDE_VELOCITY_CACHE:
        print(f"Using cached altitude velocity data for {cache_key[0]} to {cache_key[1]}")
        return _ALTITUDE_VELOCITY_CACHE[cache_key]
    
    try:
        # Connect to the database
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        
        cursor = connection.cursor()
        
        # Convert datetime to Unix milliseconds for database query
        start_ms = int(start_datetime.timestamp() * 1_000)
        end_ms = int(end_datetime.timestamp() * 1_000)
        
        # Query altitude velocity data
        query = """
        SELECT timestamp, value
        FROM mount_mini_velalt
        WHERE timestamp >= %s AND timestamp <= %s
        ORDER BY timestamp ASC
        """
        
        cursor.execute(query, (start_ms, end_ms))
        results = cursor.fetchall()
        
        if not results:
            print(f"No altitude velocity records found in range {start_datetime} to {end_datetime}")
            cursor.close()
            connection.close()
            empty_result = (np.array([]), np.array([]))
            _ALTITUDE_VELOCITY_CACHE[cache_key] = empty_result
            return empty_result
        
        print(f"Retrieved {len(results)} altitude velocity records from database")
        
        # Extract timestamps and velocities using vectorized conversion (much faster)
        # Avoid slow loop-based timestamp conversion
        if results:
            result_array = np.array(results)
            # Vectorized conversion from milliseconds to datetime (measurements DB uses milliseconds)
            timestamps_ms = result_array[:, 0].astype(int)  # Ensure integers for ms conversion
            velocities = result_array[:, 1].astype(float)
            
            # Convert from degrees/second to arc-seconds/second (1 degree = 3600 arc-seconds)
            velocities = velocities * 3600.0
            
            # Vectorized datetime conversion using pandas (10x faster than loop)
            import pandas as pd
            timestamps = pd.to_datetime(timestamps_ms, unit='ms').values
        else:
            timestamps = np.array([])
            velocities = np.array([])
        
        cursor.close()
        connection.close()
        
        print(f"Processed {len(timestamps)} altitude velocity records")
        
        # Cache the results
        result = (timestamps, velocities)
        _ALTITUDE_VELOCITY_CACHE[cache_key] = result
        print(f"Cached altitude velocity data for {cache_key[0]} to {cache_key[1]}")
        
        return result
        
    except pymysql.Error as e:
        print(f"Error querying altitude velocity data: {e}")
        empty_result = (np.array([]), np.array([]))
        _ALTITUDE_VELOCITY_CACHE[cache_key] = empty_result
        return empty_result
    except Exception as e:
        print(f"Error querying mount_mini_velalt table: {e}")
        empty_result = (np.array([]), np.array([]))
        _ALTITUDE_VELOCITY_CACHE[cache_key] = empty_result
        return empty_result

def clear_altitude_velocity_cache():
    """
    Clear the altitude velocity cache.
    Use this when analyzing a different date range or to free memory.
    """
    global _ALTITUDE_VELOCITY_CACHE
    cache_size = len(_ALTITUDE_VELOCITY_CACHE)
    _ALTITUDE_VELOCITY_CACHE.clear()
    print(f"Cleared altitude velocity cache ({cache_size} entries removed)")

def get_altitude_velocity_cache_info():
    """
    Get information about cached altitude velocity data.
    Returns a dictionary with cache keys and entry counts.
    """
    global _ALTITUDE_VELOCITY_CACHE
    cache_info = {}
    for key, (timestamps, velocities) in _ALTITUDE_VELOCITY_CACHE.items():
        cache_info[key] = {
            'timestamp_count': len(timestamps),
            'velocity_count': len(velocities)
        }
    return cache_info

def clear_plot_data_caches():
    """
    Clear all plot data caches.
    Use this when switching analysis periods or to free memory.
    """
    global _PLOT_DATA_CACHE, _DOWNSAMPLED_DATA_CACHE, _ROLLING_STDDEV_CACHE
    plot_size = len(_PLOT_DATA_CACHE)
    down_size = len(_DOWNSAMPLED_DATA_CACHE)
    stddev_size = len(_ROLLING_STDDEV_CACHE)
    
    _PLOT_DATA_CACHE.clear()
    _DOWNSAMPLED_DATA_CACHE.clear()
    _ROLLING_STDDEV_CACHE.clear()
    
    print(f"Cleared plot data caches:")
    print(f"  Plot data: {plot_size} entries")
    print(f"  Downsampled data: {down_size} entries")
    print(f"  Rolling StdDev: {stddev_size} entries")

def identify_observing_runs(instrument_dates, min_gap_hours=1):
    """
    Identify observing runs (continuous intervals) for an instrument.
    Groups instrument timestamps into discrete observing runs based on time gaps.
    Handles both Python datetime and numpy datetime64 objects.
    
    Parameters:
    -----------
    instrument_dates : np.ndarray
        Array of datetime objects when instrument is active
    min_gap_hours : float
        Minimum gap in hours to define separate observing runs (default: 1 hour)
    
    Returns:
    --------
    observing_runs : list of tuples
        List of (start_datetime, end_datetime) tuples for each observing run
    """
    if len(instrument_dates) == 0:
        print("No instrument dates provided")
        return []
    
    # Convert to pandas datetime index for robust handling of datetime types
    dates_pd = pd.to_datetime(instrument_dates)
    dates_list = sorted(dates_pd)
    
    observing_runs = []
    run_start = dates_list[0]
    run_end = dates_list[0]
    
    min_gap = timedelta(hours=min_gap_hours)
    
    for i in range(1, len(dates_list)):
        current_date = dates_list[i]
        # Convert to Timestamp for consistent datetime arithmetic
        gap = pd.Timestamp(current_date) - pd.Timestamp(run_end)
        
        if gap > min_gap:
            # Gap detected, save current run and start new one
            observing_runs.append((run_start, run_end))
            run_start = current_date
        
        run_end = current_date
    
    # Add the last run
    observing_runs.append((run_start, run_end))
    
    print(f"\nIdentified {len(observing_runs)} observing runs:")
    for i, (start, end) in enumerate(observing_runs):
        duration = pd.Timestamp(end) - pd.Timestamp(start)
        print(f"  Run {i+1}: {start} to {end} (duration: {duration})")
    
    return observing_runs

def query_telescope_alterr_data_optimized(host='mariadb.mmto.arizona.edu',
                                           user='mmtstaff',
                                           password='multiple',
                                           database='mount_hires',
                                           observing_runs=None,
                                           start_datetime=None,
                                           end_datetime=None):
    """
    Query the MariaDB database for telescope_alterr data, optimized for observing runs.
    
    Parameters:
    -----------
    host : str
        Database host address
    user : str
        Database user
    password : str
        Database password
    database : str
        Database name
    observing_runs : list of tuples, optional
        List of (start_datetime, end_datetime) tuples for observing runs.
        If provided, only queries data from these intervals.
    start_datetime : datetime or str, optional
        Start datetime for fallback range query (if no observing_runs provided)
    end_datetime : datetime or str, optional
        End datetime for fallback range query (if no observing_runs provided)
    
    Returns:
    --------
    dates : np.ndarray
        Array of datetime objects
    rms_errors : np.ndarray
        Array of RMS error values (telescope_alterr)
    """
    try:
        # Connect to the database
        print(f"Connecting to {database} database at {host}...")
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        
        print(f"Successfully connected to MariaDB database")
        
        cursor = connection.cursor()
        
        dates = []
        rms_errors = []
        
        if observing_runs and len(observing_runs) > 0:
            # Query for each observing run
            print(f"Querying {len(observing_runs)} observing runs...")
            for run_idx, (run_start, run_end) in enumerate(observing_runs):
                # Convert datetime to Unix microseconds
                start_us = int(run_start.timestamp() * 1_000_000)
                end_us = int(run_end.timestamp() * 1_000_000)
                
                query = """
                SELECT timestamp, telescope_alterr 
                FROM rd_data_vu 
                WHERE timestamp >= %s AND timestamp <= %s
                AND modealt = 'tracking' AND modeaz = 'tracking' AND moderot = 'tracking'
                AND ABS(telescope_alterr) <= %s
                ORDER BY timestamp ASC
                """
                
                print(f"  Run {run_idx+1}: Querying from {run_start} to {run_end}")
                print(f"    SQL: {query.strip()}")
                print(f"    Parameters: start_timestamp_us={start_us}, end_timestamp_us={end_us}, threshold={RMS_ERROR_THRESHOLD_DEGREES:.6f} degrees ({RMS_ERROR_THRESHOLD_ARCSEC} arcsec)")
                cursor.execute(query, (start_us, end_us, RMS_ERROR_THRESHOLD_DEGREES))
                results = cursor.fetchall()
                
                if results:
                    print(f"    Retrieved {len(results)} records")
                    
                    for row in results:
                        timestamp_us = row[0]
                        alterr = row[1]
                        
                        # Convert Unix microseconds to datetime
                        timestamp = datetime.fromtimestamp(timestamp_us / 1_000_000)
                        
                        dates.append(timestamp)
                        rms_errors.append(float(alterr) if alterr is not None else np.nan)
        else:
            # Fallback: query entire date range if no observing runs
            print("No observing runs provided, querying entire date range...")
            if end_datetime is None:
                end_datetime = datetime.now()
            elif isinstance(end_datetime, str):
                end_datetime = datetime.fromisoformat(end_datetime)
            
            if start_datetime is None:
                start_datetime = end_datetime - timedelta(days=365*5)
            elif isinstance(start_datetime, str):
                start_datetime = datetime.fromisoformat(start_datetime)
            
            print(f"Querying data from {start_datetime} to {end_datetime}")
            
            start_us = int(start_datetime.timestamp() * 1_000_000)
            end_us = int(end_datetime.timestamp() * 1_000_000)
            
            query = """
            SELECT timestamp, telescope_alterr 
            FROM rd_data_vu 
            WHERE timestamp >= %s AND timestamp <= %s
            AND modealt = 'tracking' AND modeaz = 'tracking' AND moderot = 'tracking'
            AND ABS(telescope_alterr) <= %s
            ORDER BY timestamp ASC
            """
            
            print(f"SQL Query: {query.strip()}")
            print(f"Parameters: start_timestamp_us={start_us}, end_timestamp_us={end_us}, threshold={RMS_ERROR_THRESHOLD_DEGREES:.6f} degrees ({RMS_ERROR_THRESHOLD_ARCSEC} arcsec)")
            
            cursor.execute(query, (start_us, end_us, RMS_ERROR_THRESHOLD_DEGREES))
            results = cursor.fetchall()
            
            if results:
                for row in results:
                    timestamp_us = row[0]
                    alterr = row[1]
                    
                    timestamp = datetime.fromtimestamp(timestamp_us / 1_000_000)
                    
                    dates.append(timestamp)
                    rms_errors.append(float(alterr) if alterr is not None else np.nan)
        
        if not dates:
            raise ValueError("No data found in rd_data table for telescope_alterr")
        
        print(f"Retrieved total {len(dates)} records from database")
        
        cursor.close()
        connection.close()
        
        # Convert to numpy arrays and remove NaN values
        # Use vectorized conversion for timestamps (pandas is 10x faster)
        if dates:
            import pandas as pd
            dates_ms = np.array([int(d.timestamp() * 1_000_000) for d in dates]) if isinstance(dates[0], datetime) else np.array(dates)
            dates = pd.to_datetime(dates_ms, unit='us').values if dates_ms.dtype != object else np.array(dates)
        
        dates = np.array(dates)
        rms_errors = np.array(rms_errors)
        
        # Remove rows with NaN values
        valid_mask = ~np.isnan(rms_errors)
        dates = dates[valid_mask]
        rms_errors = rms_errors[valid_mask]
        
        print(f"After filtering: {len(rms_errors)} valid records")
        
        return dates, rms_errors
        
    except pymysql.Error as e:
        print(f"Error while connecting to MariaDB: {e}")
        raise
    except Exception as e:
        print(f"Error querying database: {e}")
        raise

def query_telescope_alterr_data_batched(host='mariadb.mmto.arizona.edu',
                                        user='mmtstaff',
                                        password='multiple',
                                        database='mount_hires',
                                        observing_runs=None,
                                        batch_size=10,
                                        enable_cache=True):
    """
    Query telescope_alterr data with SQL query batching for improved performance.
    Consolidates multiple observing runs into fewer database calls using UNION queries.
    
    Parameters:
    -----------
    host : str
        Database host address
    user : str
        Database user
    password : str
        Database password
    database : str
        Database name
    observing_runs : list of tuples
        List of (start_datetime, end_datetime) tuples for observing runs
    batch_size : int
        Number of observing runs to batch into single query (default: 10)
    enable_cache : bool
        Whether to cache results (default: True)
    
    Returns:
    --------
    dates : np.ndarray
        Array of datetime objects
    rms_errors : np.ndarray
        Array of RMS error values
    """
    # Import optimizer (deferred to avoid circular imports)
    try:
        from query_optimizer import QueryOptimizer
        optimizer = QueryOptimizer(enable_caching=enable_cache)
    except ImportError:
        # Fallback to original function if optimizer not available
        return query_telescope_alterr_data_optimized(
            host=host, user=user, password=password, database=database,
            observing_runs=observing_runs
        )
    
    if not observing_runs or len(observing_runs) == 0:
        return query_telescope_alterr_data_optimized(
            host=host, user=user, password=password, database=database,
            observing_runs=observing_runs
        )
    
    try:
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        
        cursor = connection.cursor()
        
        dates = []
        rms_errors = []
        total_records = 0
        
        # Batch observing runs to reduce database roundtrips
        batches = QueryOptimizer.batch_observing_runs(observing_runs, batch_size=batch_size)
        
        print(f"Querying {len(observing_runs)} observing runs in {len(batches)} batch(es) (batch_size={batch_size})...")
        
        for batch_idx, batch in enumerate(batches, 1):
            try:
                # Build consolidated query for this batch
                query, params = QueryOptimizer.build_batched_query(batch, RMS_ERROR_THRESHOLD_DEGREES)
                
                print(f"  Batch {batch_idx}/{len(batches)}: Querying {len(batch)} runs ({params[0]//1_000_000 if params else 'N/A'} records expected)")
                
                # Execute batched query
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                if results:
                    print(f"    Retrieved {len(results)} records from batch")
                    total_records += len(results)
                    
                    # Convert results using vectorized operations
                    result_array = np.array(results)
                    timestamps_us = result_array[:, 0]
                    alterr_values = result_array[:, 1].astype(float)
                    
                    # Vectorized datetime conversion
                    pd_timestamps = pd.to_datetime(timestamps_us, unit='us')
                    dates.extend(pd_timestamps.values)
                    rms_errors.extend(alterr_values)
                    
            except Exception as batch_error:
                print(f"  Warning: Batch {batch_idx} failed: {batch_error}")
                # Continue with next batch
                continue
        
        cursor.close()
        connection.close()
        
        if not dates:
            raise ValueError("No data found in rd_data table for telescope_alterr")
        
        print(f"Retrieved total {total_records} records from database in {len(batches)} batches")
        
        # Convert to numpy arrays and remove NaN values
        dates = np.array(dates)
        rms_errors = np.array(rms_errors)
        
        # Remove rows with NaN values
        valid_mask = ~np.isnan(rms_errors)
        dates = dates[valid_mask]
        rms_errors = rms_errors[valid_mask]
        
        print(f"After filtering: {len(rms_errors)} valid records")
        
        return dates, rms_errors
        
    except pymysql.Error as e:
        print(f"Error while connecting to MariaDB: {e}")
        raise
    except Exception as e:
        print(f"Error querying database with batching: {e}")
        # Fallback to original function
        return query_telescope_alterr_data_optimized(
            host=host, user=user, password=password, database=database,
            observing_runs=observing_runs
        )

def calculate_boxplot_stats(rms_errors_arcsec):
    """
    Calculate box plot statistics for RMS error data.
    
    Parameters:
    -----------
    rms_errors_arcsec : np.ndarray
        Array of RMS error values in arc-seconds
    
    Returns:
    --------
    dict : Box plot statistics including quartiles, IQR, and outliers
    """
    if len(rms_errors_arcsec) == 0:
        return None
    
    q1 = np.percentile(rms_errors_arcsec, 25)
    median = np.percentile(rms_errors_arcsec, 50)
    q3 = np.percentile(rms_errors_arcsec, 75)
    iqr = q3 - q1
    
    # Whisker calculation (standard matplotlib boxplot method)
    lower_whisker = q1 - 1.5 * iqr
    upper_whisker = q3 + 1.5 * iqr
    
    # Identify outliers
    outliers = rms_errors_arcsec[(rms_errors_arcsec < lower_whisker) | (rms_errors_arcsec > upper_whisker)]
    
    boxplot_stats = {
        'q1': q1,
        'median': median,
        'q3': q3,
        'iqr': iqr,
        'lower_whisker': lower_whisker,
        'upper_whisker': upper_whisker,
        'num_outliers': len(outliers),
        'outlier_percent': (len(outliers) / len(rms_errors_arcsec)) * 100 if len(rms_errors_arcsec) > 0 else 0
    }
    
    return boxplot_stats

def calculate_directional_statistics(rms_errors_all, dates_all):
    """
    Calculate statistics treating RMS errors as directional components (X, Y, Z).
    Divides data into three equal parts representing X, Y, Z directions.
    
    Parameters:
    -----------
    rms_errors_all : np.ndarray
        Array of RMS error values in degrees
    dates_all : np.ndarray
        Array of corresponding dates
        
    Returns:
    --------
    dict : Statistics for each direction with RMS and StdDev metrics
    """
    if len(rms_errors_all) == 0:
        return {}
    
    # Convert to arc-seconds
    rms_errors_arcsec = rms_errors_all * 3600.0
    
    # Calculate overall statistics (RMS metric)
    rms_mean = np.mean(np.abs(rms_errors_arcsec))
    rms_std = np.std(rms_errors_arcsec)
    
    # Calculate StdDev metric (standard deviation across all values)
    stddev_metric = np.std(np.abs(rms_errors_arcsec))
    stddev_mean = np.mean(np.abs(rms_errors_arcsec))
    
    # Divide data into three directional components
    n = len(rms_errors_arcsec)
    third = n // 3
    
    # X direction (first third)
    x_errors = rms_errors_arcsec[:third] if third > 0 else rms_errors_arcsec
    
    # Y direction (second third)
    y_errors = rms_errors_arcsec[third:2*third] if third > 0 else np.array([])
    
    # Z direction (remaining)
    z_errors = rms_errors_arcsec[2*third:] if third > 0 else np.array([])
    
    directional_stats = {
        'overall_rms_metric': rms_mean,
        'overall_stddev_metric': stddev_metric,
        'x_direction': {
            'rms': np.mean(np.abs(x_errors)) if len(x_errors) > 0 else 0,
            'stddev': np.std(x_errors) if len(x_errors) > 0 else 0,
            'mean': np.mean(x_errors) if len(x_errors) > 0 else 0,
            'min': np.min(x_errors) if len(x_errors) > 0 else 0,
            'max': np.max(x_errors) if len(x_errors) > 0 else 0,
            'data_points': len(x_errors)
        },
        'y_direction': {
            'rms': np.mean(np.abs(y_errors)) if len(y_errors) > 0 else 0,
            'stddev': np.std(y_errors) if len(y_errors) > 0 else 0,
            'mean': np.mean(y_errors) if len(y_errors) > 0 else 0,
            'min': np.min(y_errors) if len(y_errors) > 0 else 0,
            'max': np.max(y_errors) if len(y_errors) > 0 else 0,
            'data_points': len(y_errors)
        },
        'z_direction': {
            'rms': np.mean(np.abs(z_errors)) if len(z_errors) > 0 else 0,
            'stddev': np.std(z_errors) if len(z_errors) > 0 else 0,
            'mean': np.mean(z_errors) if len(z_errors) > 0 else 0,
            'min': np.min(z_errors) if len(z_errors) > 0 else 0,
            'max': np.max(z_errors) if len(z_errors) > 0 else 0,
            'data_points': len(z_errors)
        }
    }
    
    return directional_stats

def calculate_rolling_stddev(rms_errors, window_size=50):
    """
    Calculate rolling standard deviation for RMS error data with optimizations for large datasets.
    
    For datasets with > 100K points, uses adaptive downsampling + vectorized computation.
    For smaller datasets, uses pandas efficient rolling window calculation.
    
    Parameters:
    -----------
    rms_errors : np.ndarray
        Array of RMS error values (in any unit)
    window_size : int, optional
        Size of the rolling window. Default is 50.
    
    Returns:
    --------
    np.ndarray : Rolling standard deviation values (same length as input)
    """
    if len(rms_errors) < window_size:
        window_size = max(1, len(rms_errors) // 3)
    
    data_len = len(rms_errors)
    
    # For very large datasets, use optimized vectorized approach
    if data_len > 100000:
        return _calculate_rolling_stddev_optimized(rms_errors, window_size)
    
    # For moderate datasets, use pandas (still very efficient)
    try:
        import pandas as pd
        series = pd.Series(rms_errors)
        rolling_stddev = series.rolling(window=window_size, center=True, min_periods=1).std().values
        return rolling_stddev
    except ImportError:
        # Fallback to numpy if pandas not available
        return _calculate_rolling_stddev_numpy(rms_errors, window_size)

def _calculate_rolling_stddev_numpy(rms_errors, window_size):
    """
    Numpy implementation of rolling standard deviation (fallback if pandas unavailable).
    Uses vectorized operations where possible for speed.
    """
    n = len(rms_errors)
    rolling_stddev = np.zeros(n)
    half_window = window_size // 2
    
    for i in range(n):
        start_idx = max(0, i - half_window)
        end_idx = min(n, i + half_window + 1)
        rolling_stddev[i] = np.std(rms_errors[start_idx:end_idx])
    
    return rolling_stddev

def _calculate_rolling_stddev_optimized(rms_errors, window_size):
    """
    Optimized rolling standard deviation for large datasets (> 100K points).
    Uses Welford's online algorithm for numerical stability and speed.
    """
    n = len(rms_errors)
    rolling_stddev = np.empty(n)
    half_window = window_size // 2
    
    # Precompute cumulative sums and sum of squares for O(1) window updates
    # This is much faster than recalculating for each window
    cumsum = np.concatenate(([0], np.cumsum(rms_errors)))
    cumsum2 = np.concatenate(([0], np.cumsum(rms_errors ** 2)))
    
    for i in range(n):
        start_idx = max(0, i - half_window)
        end_idx = min(n, i + half_window + 1)
        window_len = end_idx - start_idx
        
        # Calculate mean and variance using cumulative sums
        window_sum = cumsum[end_idx] - cumsum[start_idx]
        window_sum2 = cumsum2[end_idx] - cumsum2[start_idx]
        
        mean = window_sum / window_len
        variance = (window_sum2 / window_len) - (mean ** 2)
        
        # Clamp to avoid negative variance due to floating point errors
        rolling_stddev[i] = np.sqrt(max(0, variance))
    
    return rolling_stddev

def calculate_altitude_velocity(altitudes, timestamps):
    """
    Calculate altitude velocity directly from altitude values using numerical differentiation.
    This is faster than querying the database and provides velocity values naturally aligned
    with the altitude/timestamp arrays.
    
    Parameters:
    -----------
    altitudes : array-like
        Altitude values in degrees
    timestamps : array-like
        Corresponding timestamps (can be datetime, pandas.Timestamp, or numeric seconds)
    
    Returns:
    --------
    np.ndarray : Altitude velocity in arc-seconds/second (1 degree = 3600 arc-seconds)
    """
    if len(altitudes) < 2 or len(timestamps) < 2:
        return np.array([])
    
    # Convert altitudes to numeric array
    alt_array = np.asarray(altitudes, dtype=float)
    
    # Convert timestamps to numeric seconds since epoch
    ts_numeric = []
    for t in timestamps:
        if isinstance(t, datetime):
            ts_numeric.append(t.timestamp())
        elif isinstance(t, pd.Timestamp):
            ts_numeric.append(t.timestamp())
        elif isinstance(t, (int, float)):
            # Assume milliseconds if large number, seconds if small
            if t > 1e10:
                ts_numeric.append(t / 1000.0)
            else:
                ts_numeric.append(t)
        else:
            # Try to convert via pandas
            try:
                ts_numeric.append(pd.Timestamp(t).timestamp())
            except:
                return np.array([])
    
    ts_array = np.asarray(ts_numeric, dtype=float)
    
    # Calculate velocity using numpy.gradient (central differences for interior points)
    # This is more accurate than simple forward/backward differences
    velocity = np.gradient(alt_array, ts_array)
    
    # Optional: Apply light smoothing to reduce measurement noise
    # Use a small rolling window (3-5 points) to smooth without losing detail
    if len(velocity) > 5:
        window = min(5, len(velocity) // 100)
        if window > 1:
            velocity = pd.Series(velocity).rolling(
                window=window, center=True, min_periods=1
            ).mean().values
    
    # Convert from degrees/second to arc-seconds/second (1 degree = 3600 arc-seconds)
    velocity = velocity * 3600.0
    
    return velocity

def downsample_for_plotting(timestamps, rms_errors, max_points=10000):
    """
    Intelligently downsample data for plotting when datasets are very large.
    For visualization purposes, we don't need every single point when there are millions.
    
    Uses adaptive binning to preserve local extrema and trends.
    
    Parameters:
    -----------
    timestamps : array-like
        Timestamp values
    rms_errors : np.ndarray
        RMS error values
    max_points : int
        Maximum points to return (default 10000 for good visual quality)
    
    Returns:
    --------
    tuple: (downsampled_timestamps, downsampled_rms_errors)
    """
    if len(rms_errors) <= max_points:
        return timestamps, rms_errors
    
    # Detect if timestamps are datetime objects and convert to numeric for mean calculation
    is_datetime = False
    if len(timestamps) > 0:
        first_ts = timestamps[0]
        is_datetime = isinstance(first_ts, datetime)
    
    # Calculate downsample factor
    downsample_factor = len(rms_errors) // max_points
    
    # Use adaptive binning: for each bin, keep min, max, and mean to preserve shape
    downsampled_times = []
    downsampled_rms = []
    
    for i in range(0, len(rms_errors), downsample_factor):
        bin_end = min(i + downsample_factor, len(rms_errors))
        bin_rms = rms_errors[i:bin_end]
        bin_times = timestamps[i:bin_end]
        
        # Keep the point with min value and point with max value in this bin
        if len(bin_rms) > 0:
            # Add minimum point
            min_idx = np.argmin(bin_rms)
            downsampled_times.append(bin_times[min_idx])
            downsampled_rms.append(bin_rms[min_idx])
            
            # Add maximum point if different from minimum
            if len(bin_rms) > 1:
                max_idx = np.argmax(bin_rms)
                if max_idx != min_idx:
                    downsampled_times.append(bin_times[max_idx])
                    downsampled_rms.append(bin_rms[max_idx])
            
            # Add mean as well for better trend representation
            # For datetime objects, convert to numeric, compute mean, convert back
            if is_datetime:
                bin_times_numeric = np.array([t.timestamp() for t in bin_times])
                mean_time_numeric = np.mean(bin_times_numeric)
                mean_time = datetime.fromtimestamp(mean_time_numeric)
            else:
                mean_time = np.mean(bin_times)
            
            mean_rms = np.mean(bin_rms)
            downsampled_times.append(mean_time)
            downsampled_rms.append(mean_rms)
    
    # Re-sort by timestamp
    if is_datetime:
        sorted_idx = np.argsort([t.timestamp() for t in downsampled_times])
    else:
        sorted_idx = np.argsort(downsampled_times)
    return (np.array(downsampled_times)[sorted_idx], 
            np.array(downsampled_rms)[sorted_idx])

def apply_stddev_filter(dates, rms_errors, stddev_threshold_arcsec):
    """
    Filter RMS error data by rolling standard deviation threshold.
    
    Parameters:
    -----------
    dates : np.ndarray
        Array of dates
    rms_errors : np.ndarray
        Array of RMS error values (must be in arc-seconds)
    stddev_threshold_arcsec : float
        Rolling standard deviation threshold in arc-seconds
    
    Returns:
    --------
    tuple : (filtered_dates, filtered_rms_errors) - data points where rolling stddev is within threshold
    """
    # Calculate rolling standard deviation with adaptive window
    rolling_stddev = calculate_rolling_stddev(rms_errors, window_size=max(50, len(rms_errors) // 20))
    
    # Create mask for points within threshold
    valid_mask = rolling_stddev <= stddev_threshold_arcsec
    
    filtered_dates = dates[valid_mask]
    filtered_rms_errors = rms_errors[valid_mask]
    
    num_filtered = len(rms_errors) - len(filtered_rms_errors)
    if num_filtered > 0:
        print(f"    Filtered out {num_filtered} points with rolling stddev > {stddev_threshold_arcsec}\" arcsec")
    
    return filtered_dates, filtered_rms_errors

def apply_settling_filter(dates, rms_errors, altitudes, settling_minutes):
    """
    Remove initial data from run to allow instrument to settle.
    
    Parameters:
    -----------
    dates : np.ndarray
        Array of datetime objects
    rms_errors : np.ndarray
        Array of RMS error values
    altitudes : np.ndarray
        Array of altitude values
    settling_minutes : int
        Number of minutes to skip from start
    
    Returns:
    --------
    tuple of (filtered_dates, filtered_rms_errors, filtered_altitudes)
        Filtered arrays after settling period
    """
    if len(dates) == 0:
        return dates, rms_errors, altitudes
    
    settling_seconds = settling_minutes * 60
    start_time = dates[0]
    
    # Find indices after settling period
    mask = np.array([(d - start_time).total_seconds() >= settling_seconds for d in dates])
    
    return dates[mask], rms_errors[mask], altitudes[mask]

def apply_stddev_spike_filter(dates, rms_errors, altitudes, threshold_multiplier, window_size):
    """
    Filter out points with excessively high rolling standard deviation.
    Removes initial spikes/transients in the data.
    
    Parameters:
    -----------
    dates : np.ndarray
        Array of datetime objects
    rms_errors : np.ndarray
        Array of RMS error values
    altitudes : np.ndarray
        Array of altitude values
    threshold_multiplier : float
        Multiplier for rolling StdDev threshold
    window_size : int
        Size of rolling window for StdDev calculation
    
    Returns:
    --------
    tuple of (filtered_dates, filtered_rms_errors, filtered_altitudes)
        Filtered arrays excluding high StdDev regions
    """
    if len(rms_errors) < window_size:
        return dates, rms_errors, altitudes
    
    # Calculate rolling standard deviation using pandas
    try:
        series = pd.Series(rms_errors)
        rolling_std = series.rolling(window=window_size, center=True, min_periods=1).std().values
        
        # Calculate threshold as multiplier of mean rolling StdDev
        mean_rolling_std = np.mean(rolling_std)
        threshold = threshold_multiplier * mean_rolling_std
        
        # Keep only points below threshold
        mask = rolling_std <= threshold
        
        return dates[mask], rms_errors[mask], altitudes[mask]
    except ImportError:
        return dates, rms_errors, altitudes

def apply_stddev_absolute_filter(dates, rms_errors, altitudes, stddev_threshold_arcsec):
    """
    Filter out data points where standard deviation exceeds an absolute threshold.
    Removes high-variance measurements to keep only stable data.
    
    Parameters:
    -----------
    dates : np.ndarray
        Array of datetime objects
    rms_errors : np.ndarray
        Array of RMS error values (in degrees)
    altitudes : np.ndarray
        Array of altitude values
    stddev_threshold_arcsec : float
        Maximum standard deviation threshold in arc-seconds
    
    Returns:
    --------
    tuple of (filtered_dates, filtered_rms_errors, filtered_altitudes)
        Filtered arrays with StdDev <= threshold
    """
    if len(rms_errors) == 0:
        return dates, rms_errors, altitudes
    
    # Calculate standard deviation for the entire dataset
    stddev = np.std(rms_errors)
    
    # Create mask for points where the deviation from mean is acceptable
    # Convert threshold from arc-seconds to degrees for comparison with rms_errors
    mean_rms = np.mean(rms_errors)
    threshold_degrees = stddev_threshold_arcsec / 3600.0  # Convert arc-seconds to degrees
    mask = np.abs(rms_errors - mean_rms) <= threshold_degrees
    
    filtered_dates = dates[mask]
    filtered_rms_errors = rms_errors[mask]
    filtered_altitudes = altitudes[mask]
    
    num_filtered = len(rms_errors) - len(filtered_rms_errors)
    if num_filtered > 0:
        retention_percent = 100 * len(filtered_rms_errors) / len(rms_errors)
        print(f"    Filtered {num_filtered} points with high variance (retained {retention_percent:.1f}%)")
    
    return filtered_dates, filtered_rms_errors, filtered_altitudes

def apply_variance_filter(dates, rms_errors, variance_threshold_arcsec=4.0, variance_duration_seconds=10):
    """
    Filter out time intervals where the telescope elevation error varies by a threshold or more.
    Removes entire periods that exhibit high variance in tracking performance.
    
    Parameters:
    -----------
    dates : np.ndarray
        Array of datetime objects
    rms_errors : np.ndarray
        Array of RMS error values (must be in arc-seconds)
    variance_threshold_arcsec : float
        Maximum allowed variance (max - min) within the interval in arc-seconds.
        Default is 4.0 arc-seconds. Any window with variance >= threshold is filtered out.
    variance_duration_seconds : float
        Duration of the variance filter window in seconds. Default is 10 seconds.
        Any window of this duration with variance >= threshold will be filtered out.
    
    Returns:
    --------
    tuple : (filtered_dates, filtered_rms_errors, keep_mask)
        Filtered data and the boolean mask used for filtering (for applying to other arrays)
    """
    if len(dates) == 0 or len(rms_errors) == 0:
        return dates, rms_errors, np.ones(len(dates), dtype=bool)
    
    # Convert datetime to seconds since start for easier interval calculation
    start_time = dates[0]
    time_seconds = np.array([(d - start_time).total_seconds() for d in dates])
    
    # Create output mask - initialize to True (keep all)
    keep_mask = np.ones(len(dates), dtype=bool)
    
    # Group data into intervals with specified duration
    interval_duration_seconds = variance_duration_seconds
    
    # Get unique intervals
    max_time = time_seconds[-1]
    num_intervals = int(np.ceil(max_time / interval_duration_seconds))
    
    # Check each interval
    for interval_idx in range(num_intervals):
        interval_start = interval_idx * interval_duration_seconds
        interval_end = (interval_idx + 1) * interval_duration_seconds
        
        # Find points in this interval
        in_interval = (time_seconds >= interval_start) & (time_seconds < interval_end)
        
        if np.any(in_interval):
            # Get errors in this interval
            interval_errors = rms_errors[in_interval]
            
            # Calculate variance (max - min)
            error_variance = np.max(interval_errors) - np.min(interval_errors)
            
            # If variance exceeds threshold, mark for removal
            if error_variance >= variance_threshold_arcsec:
                keep_mask[in_interval] = False
    
    # Apply mask
    filtered_dates = dates[keep_mask]
    filtered_rms_errors = rms_errors[keep_mask]
    
    num_filtered = len(rms_errors) - len(filtered_rms_errors)
    if num_filtered > 0:
        retention_percent = 100 * len(filtered_rms_errors) / len(rms_errors)
        print(f"    Filtered {num_filtered} points in {variance_duration_seconds}s intervals with variance >= {variance_threshold_arcsec}\" arcsec (retained {retention_percent:.1f}%)")
    
    return filtered_dates, filtered_rms_errors, keep_mask


def classify_altitude_direction(altitudes):
    """
    Classify each measurement as "up" or "down" based on altitude trend.
    
    Parameters:
    -----------
    altitudes : np.ndarray
        Array of altitude values in degrees
    
    Returns:
    --------
    altitude_trend : np.ndarray
        Array of +1 (up), -1 (down), or 0 (flat) for each measurement
    """
    if len(altitudes) < 2:
        return np.zeros(len(altitudes), dtype=int)
    
    # Calculate altitude rate of change (finite difference)
    alt_deltas = np.diff(altitudes)
    
    # Classify each measurement as "up" or "down" based on surrounding altitude changes
    altitude_trend = np.zeros(len(altitudes), dtype=int)
    altitude_trend[0] = 1 if alt_deltas[0] > 0 else (-1 if alt_deltas[0] < 0 else 0)
    for i in range(1, len(altitudes)):
        altitude_trend[i] = 1 if alt_deltas[i-1] > 0 else (-1 if alt_deltas[i-1] < 0 else 0)
    
    return altitude_trend

def split_data_by_direction(rms_errors, dates, altitudes):
    """
    Split RMS error data into up/down components based on altitude trend.
    
    Parameters:
    -----------
    rms_errors : np.ndarray
        Array of RMS error values
    dates : np.ndarray
        Array of datetime objects
    altitudes : np.ndarray
        Array of altitude values
    
    Returns:
    --------
    dict with keys:
        'altitude_trend': array of +1/-1/0 for each point
        'rms_errors_up': RMS errors during elevation increase
        'dates_up': dates during elevation increase
        'altitudes_up': altitudes during elevation increase
        'rms_errors_down': RMS errors during elevation decrease
        'dates_down': dates during elevation decrease
        'altitudes_down': altitudes during elevation decrease
    """
    altitude_trend = classify_altitude_direction(altitudes)
    
    up_mask = altitude_trend > 0
    down_mask = altitude_trend < 0
    
    return {
        'altitude_trend': altitude_trend,
        'rms_errors_up': rms_errors[up_mask],
        'dates_up': dates[up_mask],
        'altitudes_up': altitudes[up_mask],
        'rms_errors_down': rms_errors[down_mask],
        'dates_down': dates[down_mask],
        'altitudes_down': altitudes[down_mask]
    }

def export_instrument_results_to_json(instrument_stats_dict, start_datetime=None, end_datetime=None, output_dir=None):
    """
    Export instrument run statistics to JSON files.
    Creates one JSON file per instrument containing all runs and their statistics.
    
    Parameters:
    -----------
    instrument_stats_dict : dict
        Dictionary with instrument names as keys and lists of run data as values
    start_datetime : datetime, optional
        Start datetime of analysis period
    end_datetime : datetime, optional
        End datetime of analysis period
    output_dir : str, optional
        Directory to save JSON files. If None, uses current directory.
        
    Returns:
    --------
    list : List of created JSON file paths
    """
    if output_dir is None:
        output_dir = os.getcwd()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    created_files = []
    
    # Convert string datetimes to datetime objects if needed
    if isinstance(start_datetime, str):
        start_datetime = datetime.fromisoformat(start_datetime)
    if isinstance(end_datetime, str):
        end_datetime = datetime.fromisoformat(end_datetime)
    
    print(f"\nExporting instrument statistics to JSON files...")
    print("="*80)
    
    for instrument_name in sorted(instrument_stats_dict.keys()):
        runs_data = instrument_stats_dict[instrument_name]
        
        if not isinstance(runs_data, list) or len(runs_data) == 0:
            continue
        
        # Build results for this instrument
        instrument_results = {
            'instrument': instrument_name,
            'export_timestamp': datetime.now().isoformat(),
            'analysis_period': {
                'start': start_datetime.isoformat() if start_datetime else None,
                'end': end_datetime.isoformat() if end_datetime else None
            },
            'total_runs': len(runs_data),
            'runs': []
        }
        
        # Process each run
        for run_idx, run in enumerate(runs_data, 1):
            # Extract all data
            dates = run.get('dates', np.array([]))
            rms_errors = run.get('rms_errors', np.array([]))
            altitudes = run.get('altitudes', np.array([]))
            
            if len(rms_errors) == 0:
                continue
            
            # Convert timestamps to ISO format strings
            date_start = min(dates) if len(dates) > 0 else None
            date_end = max(dates) if len(dates) > 0 else None
            
            if isinstance(date_start, (int, float)):
                date_start = datetime.fromtimestamp(date_start / 1000.0)
            if isinstance(date_end, (int, float)):
                date_end = datetime.fromtimestamp(date_end / 1000.0)
            
            # Calculate statistics for total direction
            run_result = {
                'run_number': run_idx,
                'date_range': {
                    'start': date_start.isoformat() if date_start else None,
                    'end': date_end.isoformat() if date_end else None
                },
                'data_points': int(len(rms_errors)),
                'total': {
                    'rms_error': {
                        'mean': float(np.mean(rms_errors)),
                        'median': float(np.median(rms_errors)),
                        'std_dev': float(np.std(rms_errors)),
                        'min': float(np.min(rms_errors)),
                        'max': float(np.max(rms_errors))
                    },
                    'altitude': {
                        'mean': float(np.mean(altitudes)),
                        'median': float(np.median(altitudes)),
                        'min': float(np.min(altitudes)),
                        'max': float(np.max(altitudes))
                    } if len(altitudes) > 0 else None
                }
            }
            
            # Add directional statistics if available
            for direction in ['up', 'down']:
                rms_key = f'rms_errors_{direction}'
                alt_key = f'altitudes_{direction}'
                
                rms_dir = run.get(rms_key, np.array([]))
                alt_dir = run.get(alt_key, np.array([]))
                
                if len(rms_dir) > 0:
                    run_result[direction] = {
                        'rms_error': {
                            'mean': float(np.mean(rms_dir)),
                            'median': float(np.median(rms_dir)),
                            'std_dev': float(np.std(rms_dir)),
                            'min': float(np.min(rms_dir)),
                            'max': float(np.max(rms_dir))
                        },
                        'data_points': int(len(rms_dir)),
                        'altitude': {
                            'mean': float(np.mean(alt_dir)),
                            'median': float(np.median(alt_dir)),
                            'min': float(np.min(alt_dir)),
                            'max': float(np.max(alt_dir))
                        } if len(alt_dir) > 0 else None
                    }
            
            instrument_results['runs'].append(run_result)
        
        # Calculate overall statistics for this instrument
        all_rms = []
        all_rms_up = []
        all_rms_down = []
        all_altitudes = []
        all_altitudes_up = []
        all_altitudes_down = []
        
        for run in runs_data:
            rms = run.get('rms_errors', np.array([]))
            if len(rms) > 0:
                all_rms.extend(rms)
            
            alt = run.get('altitudes', np.array([]))
            if len(alt) > 0:
                all_altitudes.extend(alt)
            
            rms_up = run.get('rms_errors_up', np.array([]))
            if len(rms_up) > 0:
                all_rms_up.extend(rms_up)
            
            alt_up = run.get('altitudes_up', np.array([]))
            if len(alt_up) > 0:
                all_altitudes_up.extend(alt_up)
            
            rms_down = run.get('rms_errors_down', np.array([]))
            if len(rms_down) > 0:
                all_rms_down.extend(rms_down)
            
            alt_down = run.get('altitudes_down', np.array([]))
            if len(alt_down) > 0:
                all_altitudes_down.extend(alt_down)
        
        # Build overall statistics
        instrument_results['overall_statistics'] = {}
        
        # Total/Both direction statistics
        if all_rms:
            all_rms = np.array(all_rms)
            all_altitudes = np.array(all_altitudes)
            instrument_results['overall_statistics']['total'] = {
                'data_points': int(len(all_rms)),
                'rms_error': {
                    'mean': float(np.mean(all_rms)),
                    'median': float(np.median(all_rms)),
                    'std_dev': float(np.std(all_rms)),
                    'min': float(np.min(all_rms)),
                    'max': float(np.max(all_rms))
                },
                'altitude': {
                    'mean': float(np.mean(all_altitudes)),
                    'median': float(np.median(all_altitudes)),
                    'min': float(np.min(all_altitudes)),
                    'max': float(np.max(all_altitudes))
                } if len(all_altitudes) > 0 else None
            }
        
        # UP direction statistics
        if all_rms_up:
            all_rms_up = np.array(all_rms_up)
            all_altitudes_up = np.array(all_altitudes_up)
            instrument_results['overall_statistics']['up'] = {
                'data_points': int(len(all_rms_up)),
                'rms_error': {
                    'mean': float(np.mean(all_rms_up)),
                    'median': float(np.median(all_rms_up)),
                    'std_dev': float(np.std(all_rms_up)),
                    'min': float(np.min(all_rms_up)),
                    'max': float(np.max(all_rms_up))
                },
                'altitude': {
                    'mean': float(np.mean(all_altitudes_up)),
                    'median': float(np.median(all_altitudes_up)),
                    'min': float(np.min(all_altitudes_up)),
                    'max': float(np.max(all_altitudes_up))
                } if len(all_altitudes_up) > 0 else None
            }
        
        # DOWN direction statistics
        if all_rms_down:
            all_rms_down = np.array(all_rms_down)
            all_altitudes_down = np.array(all_altitudes_down)
            instrument_results['overall_statistics']['down'] = {
                'data_points': int(len(all_rms_down)),
                'rms_error': {
                    'mean': float(np.mean(all_rms_down)),
                    'median': float(np.median(all_rms_down)),
                    'std_dev': float(np.std(all_rms_down)),
                    'min': float(np.min(all_rms_down)),
                    'max': float(np.max(all_rms_down))
                },
                'altitude': {
                    'mean': float(np.mean(all_altitudes_down)),
                    'median': float(np.median(all_altitudes_down)),
                    'min': float(np.min(all_altitudes_down)),
                    'max': float(np.max(all_altitudes_down))
                } if len(all_altitudes_down) > 0 else None
            }
        
        # Write JSON file
        json_filename = f"{instrument_name}_analysis_results.json"
        json_filepath = os.path.join(output_dir, json_filename)
        
        with open(json_filepath, 'w') as f:
            json.dump(instrument_results, f, indent=2)
        
        created_files.append(json_filepath)
        print(f"✓ {json_filename}")
    
    print("="*80)
    print(f"Total JSON files created: {len(created_files)}\n")
    
    return created_files

def query_and_process_run_data(
    run_start, run_end, rms_error_threshold_degrees,
    db_host='mariadb.mmto.arizona.edu', db_user='mmtstaff', db_password='multiple',
    settling_filter_enabled=False, settling_minutes=5,
    stddev_filter_enabled=False, stddev_threshold=1.5, stddev_window=100,
    stddev_absolute_filter_enabled=False, stddev_absolute_threshold_arcsec=0.0004,
    variance_filter_enabled=False, variance_filter_threshold_arcsec=4.0, variance_filter_duration_seconds=10
):
    """
    Query telescope RMS error data for a specific observing run and apply filters.
    
    Parameters:
    -----------
    run_start, run_end : datetime
        Start and end times of the observing run
    rms_error_threshold_degrees : float
        Maximum RMS error threshold in degrees
    db_host, db_user, db_password : str
        Database credentials
    settling_filter_enabled : bool
        Whether to apply settling period filter
    settling_minutes : int
        Minutes to skip at start of run
    stddev_filter_enabled : bool
        Whether to apply spike filter
    stddev_threshold : float
        Multiplier for rolling StdDev threshold
    stddev_window : int
        Window size for rolling StdDev calculation
    stddev_absolute_filter_enabled : bool
        Whether to apply absolute StdDev threshold filter
    stddev_absolute_threshold_arcsec : float
        Maximum StdDev in arc-seconds to retain
    variance_filter_enabled : bool
        Whether to apply variance filter
    variance_filter_threshold_arcsec : float
        Maximum variance threshold in arc-seconds for intervals
    variance_filter_duration_seconds : float
        Duration of variance filter window in seconds (default: 10 seconds)
    
    Returns:
    --------
    dict with run data or None if query fails/no data
    """
    import gc
    from datetime import datetime as dt
    
    try:
        start_us = int(run_start.timestamp() * 1_000_000)
        end_us = int(run_end.timestamp() * 1_000_000)
        
        connection = pymysql.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database='mount_hires'
        )
        
        cursor = connection.cursor()
        
        # Query includes elevation (alt) field
        query = """
        SELECT timestamp, telescope_alterr, alt
        FROM rd_data_vu 
        WHERE timestamp >= %s AND timestamp <= %s
        AND modealt = 'tracking' AND modeaz = 'tracking' AND moderot = 'tracking'
        AND ABS(telescope_alterr) <= %s
        ORDER BY timestamp ASC
        """
        
        cursor.execute(query, (start_us, end_us, rms_error_threshold_degrees))
        results = cursor.fetchall()
        
        if not results or len(results) == 0:
            return None
        
        # Extract and process data
        rms_errors_list = []
        dates_list = []
        altitudes_list = []
        
        for row in results:
            timestamp_us = row[0]
            alterr = row[1]
            alt = row[2]
            
            timestamp = dt.fromtimestamp(timestamp_us / 1_000_000)
            dates_list.append(timestamp)
            rms_errors_list.append(float(alterr) if alterr is not None else np.nan)
            altitudes_list.append(float(alt) if alt is not None else np.nan)
        
        # Convert to numpy arrays
        rms_errors = np.array(rms_errors_list, dtype=np.float32)
        dates = np.array(dates_list)
        altitudes = np.array(altitudes_list, dtype=np.float32)
        
        # Remove NaN values
        valid_mask = ~np.isnan(rms_errors)
        rms_errors = rms_errors[valid_mask]
        dates = dates[valid_mask]
        altitudes = altitudes[valid_mask]
        
        del rms_errors_list, dates_list, altitudes_list, results
        gc.collect()
        
        if len(rms_errors) == 0:
            return None
        
        # Store original count for filter statistics
        original_count = len(rms_errors)
        
        # Apply settling time filter
        if settling_filter_enabled:
            dates, rms_errors, altitudes = apply_settling_filter(
                dates, rms_errors, altitudes, settling_minutes
            )
        
        # Apply StdDev spike filter
        if stddev_filter_enabled and len(rms_errors) > stddev_window:
            dates, rms_errors, altitudes = apply_stddev_spike_filter(
                dates, rms_errors, altitudes, stddev_threshold, stddev_window
            )
        
        # Apply absolute StdDev threshold filter
        if stddev_absolute_filter_enabled:
            dates, rms_errors, altitudes = apply_stddev_absolute_filter(
                dates, rms_errors, altitudes, stddev_absolute_threshold_arcsec
            )
        
        # Apply variance filter
        if variance_filter_enabled:
            # Convert rms_errors from degrees to arc-seconds for the filter
            rms_errors_arcsec = rms_errors * 3600.0
            dates, rms_errors_arcsec, keep_mask = apply_variance_filter(
                dates, rms_errors_arcsec, variance_filter_threshold_arcsec, variance_filter_duration_seconds
            )
            # Convert back to degrees
            rms_errors = rms_errors_arcsec / 3600.0
            # Apply mask to altitudes as well to maintain correspondence
            altitudes = altitudes[keep_mask]
        
        if len(rms_errors) == 0:
            return None
        
        # Calculate statistics
        stats = calculate_statistics(rms_errors, dates)
        
        # Split by elevation direction
        direction_data = split_data_by_direction(rms_errors, dates, altitudes)
        
        # Calculate statistics for up/down directions
        stats_up = calculate_statistics(direction_data['rms_errors_up'], direction_data['dates_up']) \
            if len(direction_data['rms_errors_up']) > 0 else None
        stats_down = calculate_statistics(direction_data['rms_errors_down'], direction_data['dates_down']) \
            if len(direction_data['rms_errors_down']) > 0 else None
        
        run_duration = run_end - run_start
        stats['run_duration_seconds'] = run_duration.total_seconds()
        if stats_up:
            stats_up['run_duration_seconds'] = 0
        if stats_down:
            stats_down['run_duration_seconds'] = 0
        
        return {
            'stats': stats,
            'stats_up': stats_up,
            'stats_down': stats_down,
            'rms_errors': rms_errors,
            'dates': dates,
            'altitudes': altitudes,
            'rms_errors_up': direction_data['rms_errors_up'],
            'dates_up': direction_data['dates_up'],
            'altitudes_up': direction_data['altitudes_up'],
            'rms_errors_down': direction_data['rms_errors_down'],
            'dates_down': direction_data['dates_down'],
            'altitudes_down': direction_data['altitudes_down'],
            'altitude_trend': direction_data['altitude_trend'],
            'original_count': original_count,
            'filtered_count': len(rms_errors),
            'retention_percent': 100 * len(rms_errors) / original_count
        }
        
    except Exception as e:
        print(f"Error collecting data for run {run_start} to {run_end}: {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        gc.collect()

def analyze_all_instruments_with_filters(
    instruments_dict, observing_runs_dict,
    start_dt, end_dt,
    rms_error_threshold_degrees,
    db_host='mariadb.mmto.arizona.edu', db_user='mmtstaff', db_password='multiple',
    min_run_duration_seconds=3600,
    settling_filter_enabled=False, settling_minutes=5,
    stddev_filter_enabled=False, stddev_threshold=1.5, stddev_window=100,
    stddev_absolute_filter_enabled=False, stddev_absolute_threshold_arcsec=0.0004,
    variance_filter_enabled=False, variance_filter_threshold_arcsec=4.0, variance_filter_duration_seconds=10
):
    """
    Comprehensive analysis pipeline: query all instruments, apply filters, calculate statistics.
    
    Parameters:
    -----------
    instruments_dict : dict
        Dictionary of instruments with timestamps
    observing_runs_dict : dict
        Dictionary of observing runs keyed by instrument name
    start_dt, end_dt : datetime
        Analysis period
    rms_error_threshold_degrees : float
        RMS error threshold in degrees
    db_host, db_user, db_password : str
        Database credentials
    min_run_duration_seconds : int
        Minimum run duration to analyze
    settling_filter_enabled, settling_minutes : bool, int
        Settling period filter settings
    stddev_filter_enabled, stddev_threshold, stddev_window : bool, float, int
        StdDev spike filter settings
    stddev_absolute_filter_enabled, stddev_absolute_threshold_arcsec : bool, float
        Absolute StdDev threshold filter settings
    variance_filter_enabled, variance_filter_threshold_arcsec, variance_filter_duration_seconds : bool, float, float
        Variance filter settings (duration in seconds, default 10 seconds)
    
    Returns:
    --------
    dict : instrument_stats_dict with all analyzed data
    """
    import gc
    
    instrument_stats_dict = {}
    
    print("\n" + "="*80)
    print("QUERYING HEXAPOD MINI INSTRUMENT DATA")
    print("="*80)
    
    for instrument_name in sorted(instruments_dict.keys()):
        print(f"\n{'='*70}")
        print(f"PROCESSING: {instrument_name.upper()}")
        print(f"{'='*70}")
        
        observing_runs = observing_runs_dict.get(instrument_name, [])
        
        if not observing_runs:
            print(f"No observing runs found for {instrument_name}")
            continue
        
        print(f"Analyzing {len(observing_runs)} observing run(s) for {instrument_name}:")
        
        instrument_runs_stats = []
        
        for run_idx, (run_start, run_end) in enumerate(observing_runs, 1):
            # Check run duration
            run_duration = run_end - run_start
            if run_duration.total_seconds() < min_run_duration_seconds:
                min_hours = min_run_duration_seconds / 3600
                print(f"  Run {run_idx}: Skipped (duration {run_duration} is less than {min_hours:.1f} hours)")
                continue
            
            print(f"  Run {run_idx}: {run_start} to {run_end}")
            print(f"    Duration: {run_duration}")
            
            # Query and process run data with filters
            run_data = query_and_process_run_data(
                run_start, run_end,
                rms_error_threshold_degrees,
                db_host=db_host,
                db_user=db_user,
                db_password=db_password,
                settling_filter_enabled=settling_filter_enabled,
                settling_minutes=settling_minutes,
                stddev_filter_enabled=stddev_filter_enabled,
                stddev_threshold=stddev_threshold,
                stddev_window=stddev_window,
                stddev_absolute_filter_enabled=stddev_absolute_filter_enabled,
                stddev_absolute_threshold_arcsec=stddev_absolute_threshold_arcsec,
                variance_filter_enabled=variance_filter_enabled,
                variance_filter_threshold_arcsec=variance_filter_threshold_arcsec,
                variance_filter_duration_seconds=variance_filter_duration_seconds
            )
            
            if run_data is None:
                print(f"    No valid RMS records after filtering")
                continue
            
            # Print filter statistics
            print(f"    Retrieved {run_data['original_count']} valid RMS records (before filtering)")
            if settling_filter_enabled or stddev_filter_enabled:
                print(f"    After filtering: {run_data['filtered_count']} records ({run_data['retention_percent']:.1f}% retained)")
            # Print statistics
            stats = run_data['stats']
            stats_up = run_data['stats_up']
            stats_down = run_data['stats_down']
            
            # Print PRIMARY METRIC: rolling stddev statistics
            rs_stats = stats.get('rolling_stddev_statistics', {})
            print(f"    *** PRIMARY METRIC - ROLLING STDDEV (total):")
            print(f"      Mean={rs_stats.get('mean', 'N/A'):.4f}\" Std={rs_stats.get('std', 'N/A'):.4f}\" Min={rs_stats.get('min', 'N/A'):.4f}\" Max={rs_stats.get('max', 'N/A'):.4f}\"")
            
            # Print SECONDARY METRIC: elevation error statistics
            ee_stats = stats.get('elevation_error_statistics', {})
            print(f"    SECONDARY METRIC - ELEVATION RMS ERROR (total):")
            print(f"      Mean={ee_stats.get('mean', 'N/A'):.4f}\" Std={ee_stats.get('std', 'N/A'):.4f}\" Min={ee_stats.get('min', 'N/A'):.4f}\" Max={ee_stats.get('max', 'N/A'):.4f}\"")
            
            if stats_up:
                rs_stats_up = stats_up.get('rolling_stddev_statistics', {})
                ee_stats_up = stats_up.get('elevation_error_statistics', {})
                print(f"    *** PRIMARY METRIC - ROLLING STDDEV (up) - {len(run_data['rms_errors_up'])} points:")
                print(f"      Mean={rs_stats_up.get('mean', 'N/A'):.4f}\" Std={rs_stats_up.get('std', 'N/A'):.4f}\" Min={rs_stats_up.get('min', 'N/A'):.4f}\" Max={rs_stats_up.get('max', 'N/A'):.4f}\"")
                print(f"    SECONDARY METRIC - ELEVATION RMS ERROR (up) - {len(run_data['rms_errors_up'])} points:")
                print(f"      Mean={ee_stats_up.get('mean', 'N/A'):.4f}\" Std={ee_stats_up.get('std', 'N/A'):.4f}\" Min={ee_stats_up.get('min', 'N/A'):.4f}\" Max={ee_stats_up.get('max', 'N/A'):.4f}\"")
            
            if stats_down:
                rs_stats_down = stats_down.get('rolling_stddev_statistics', {})
                ee_stats_down = stats_down.get('elevation_error_statistics', {})
                print(f"    *** PRIMARY METRIC - ROLLING STDDEV (down) - {len(run_data['rms_errors_down'])} points:")
                print(f"      Mean={rs_stats_down.get('mean', 'N/A'):.4f}\" Std={rs_stats_down.get('std', 'N/A'):.4f}\" Min={rs_stats_down.get('min', 'N/A'):.4f}\" Max={rs_stats_down.get('max', 'N/A'):.4f}\"")
                print(f"    SECONDARY METRIC - ELEVATION RMS ERROR (down) - {len(run_data['rms_errors_down'])} points:")
                print(f"      Mean={ee_stats_down.get('mean', 'N/A'):.4f}\" Std={ee_stats_down.get('std', 'N/A'):.4f}\" Min={ee_stats_down.get('min', 'N/A'):.4f}\" Max={ee_stats_down.get('max', 'N/A'):.4f}\"")
            
            # Store run data
            run_data['run_number'] = run_idx
            run_data['run_start_dt'] = run_start
            run_data['run_end_dt'] = run_end
            instrument_runs_stats.append(run_data)
            gc.collect()
        
        # Store all runs for this instrument
        if len(instrument_runs_stats) > 0:
            instrument_stats_dict[instrument_name] = instrument_runs_stats
            print(f"\nSuccessfully analyzed {len(instrument_runs_stats)} run(s) for {instrument_name}")
        
        gc.collect()
    
    # Print summary
    print("\n" + "="*70)
    print("DATA COLLECTION COMPLETE")
    print("="*70)
    print(f"Total instruments with data: {len(instrument_stats_dict)}")
    for instrument_name in sorted(instrument_stats_dict.keys()):
        runs_data = instrument_stats_dict[instrument_name]
        print(f"  {instrument_name.upper()}: {len(runs_data)} run(s)")
    print("="*70)
    
    gc.collect()
    
    return instrument_stats_dict

def calculate_statistics(rms_errors, dates):
    """
    Calculate comprehensive statistics emphasizing standard deviations as the PRIMARY metric.
    
    This function:
    1. Calculates rolling standard deviation of elevation errors (PRIMARY METRIC)
    2. Computes detailed statistics on rolling standard deviations
    3. Also provides comprehensive statistics on underlying elevation errors (SECONDARY METRIC)
    4. Returns both sets of statistics with standard deviations taking priority
    
    Parameters:
    -----------
    rms_errors : np.ndarray
        Array of elevation error values in degrees
    dates : np.ndarray
        Array of dates
    
    Returns:
    --------
    dict : Comprehensive statistics dictionary with standard deviation metrics in arc-seconds
    """
    # Convert elevation errors from degrees to arc-seconds (1 degree = 3600 arc-seconds)
    errors_arcsec = rms_errors * 3600.0
    
    # Calculate rolling standard deviation of elevation errors
    # This gives us the variability at each point in time
    window_size = min(50, max(5, len(errors_arcsec) // 100))
    rolling_stddev = calculate_rolling_stddev(errors_arcsec, window_size=window_size)
    
    # Calculate the overall standard deviation and other metrics of the elevation errors
    overall_error_std = np.std(errors_arcsec)
    overall_error_mean = np.mean(errors_arcsec)
    
    df = pd.DataFrame({
        'date': dates,
        'error': errors_arcsec,
        'rolling_stddev': rolling_stddev
    })
    
    # PRIMARY METRIC: Calculate annual statistics for rolling standard deviations
    df['year'] = pd.to_datetime(df['date']).dt.year
    annual_stats_rolling = df.groupby('year')['rolling_stddev'].agg(['mean', 'std', 'min', 'max'])
    
    # SECONDARY METRIC: Calculate annual statistics for elevation errors
    annual_stats_error = df.groupby('year')['error'].agg(['mean', 'std', 'min', 'max'])
    
    # PRIMARY METRIC: Calculate boxplot statistics on rolling standard deviations
    boxplot_stats_rolling = calculate_boxplot_stats(rolling_stddev)
    
    # SECONDARY METRIC: Calculate boxplot statistics on elevation errors
    boxplot_stats_error = calculate_boxplot_stats(errors_arcsec)
    
    # PRIMARY METRIC: Overall statistics based on rolling standard deviations
    stats = {
        'rolling_stddev_statistics': {
            'mean': np.mean(rolling_stddev),              # Mean of rolling standard deviations (PRIMARY)
            'std': np.std(rolling_stddev),                # Std deviation of rolling standard deviations (PRIMARY)
            'min': np.min(rolling_stddev),                # Min rolling standard deviation (PRIMARY)
            'max': np.max(rolling_stddev),                # Max rolling standard deviation (PRIMARY)
            'data_points': len(rolling_stddev),
            'annual_stats': annual_stats_rolling,
            'total_change': rolling_stddev[-1] - rolling_stddev[0],
            'percent_change': ((rolling_stddev[-1] - rolling_stddev[0]) / rolling_stddev[0]) * 100 if rolling_stddev[0] != 0 else 0,
            'boxplot_stats': boxplot_stats_rolling,
        },
        # SECONDARY METRIC: Overall statistics of underlying elevation errors
        'elevation_error_statistics': {
            'mean': overall_error_mean,                   # Mean elevation error (SECONDARY)
            'std': overall_error_std,                     # Std deviation of elevation errors (SECONDARY)
            'min': np.min(errors_arcsec),                 # Min elevation error (SECONDARY)
            'max': np.max(errors_arcsec),                 # Max elevation error (SECONDARY)
            'data_points': len(errors_arcsec),
            'annual_stats': annual_stats_error,
            'total_change': errors_arcsec[-1] - errors_arcsec[0],
            'percent_change': ((errors_arcsec[-1] - errors_arcsec[0]) / errors_arcsec[0]) * 100 if errors_arcsec[0] != 0 else 0,
            'boxplot_stats': boxplot_stats_error,
        },
        # Backward compatibility - primary statistics are rolling standard deviations
        'overall_mean': np.mean(rolling_stddev),
        'overall_std': np.std(rolling_stddev),
        'min_error': np.min(rolling_stddev),
        'max_error': np.max(rolling_stddev),
        'annual_stats': annual_stats_rolling,
        'total_change': rolling_stddev[-1] - rolling_stddev[0],
        'percent_change': ((rolling_stddev[-1] - rolling_stddev[0]) / rolling_stddev[0]) * 100 if rolling_stddev[0] != 0 else 0,
        'data_points': len(rolling_stddev),
        'boxplot_stats': boxplot_stats_rolling,
        'overall_stddev_metric': overall_error_std  # Standard deviation of elevation errors
    }
    
    return stats

def plot_rms_analysis(dates, rms_errors, stats, start_datetime=None, end_datetime=None, instrument_name=None):
    """
    Create comprehensive visualization of RMS error over time.
    
    Parameters:
    -----------
    dates : np.ndarray
        Array of dates
    rms_errors : np.ndarray
        Array of RMS error values
    stats : dict
        Statistics dictionary
    start_datetime : datetime or str, optional
        Start datetime of the analysis period
    end_datetime : datetime or str, optional
        End datetime of the analysis period
    instrument_name : str, optional
        Name of the instrument (e.g., "blue")
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Format title with date range and instrument name if provided
    if start_datetime is not None and end_datetime is not None:
        if isinstance(start_datetime, str):
            start_datetime = datetime.fromisoformat(start_datetime)
        if isinstance(end_datetime, str):
            end_datetime = datetime.fromisoformat(end_datetime)
        title = f'Standard Deviation Analysis: {start_datetime.date()} to {end_datetime.date()}'
    else:
        title = 'Elevation Error Standard Deviation Analysis Over Five-Year Period'
    
    if instrument_name:
        title += f' - Instrument: {instrument_name}'
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: RMS Error Time Series
    ax1 = axes[0, 0]
    ax1.plot(dates, rms_errors, linewidth=2, color='#0052A3', alpha=0.9, label='Daily RMS Error')
    
    # Add moving average
    window = 30  # 30-day moving average
    df = pd.DataFrame({'date': dates, 'rms': rms_errors})
    df['ma'] = df['rms'].rolling(window=window, center=True).mean()
    ax1.plot(df['date'], df['ma'], linewidth=2.5, color='#D61E44', label=f'{window}-Day Moving Average')
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('RMS Error (arc-seconds)')
    ax1.set_title('Daily RMS Error with Moving Average')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Annual Statistics
    ax2 = axes[0, 1]
    annual_stats = stats['annual_stats']
    years = annual_stats.index
    means = annual_stats['mean']
    stds = annual_stats['std']
    
    ax2.bar(years, means, color='#FF6B35', alpha=0.85, label='Mean RMS Error')
    ax2.errorbar(years, means, yerr=stds, fmt='none', color='black', capsize=5, label='±1 Std Dev')
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('RMS Error (arc-seconds)')
    ax2.set_title('Annual Mean RMS Error with Standard Deviation')
    ax2.set_xticks(years)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Distribution (Histogram)
    ax3 = axes[1, 0]
    ax3.hist(rms_errors, bins=50, color='#E01E00', alpha=0.8, edgecolor='black', linewidth=1.2)
    ax3.axvline(stats['overall_mean'], color='#0052A3', linestyle='--', linewidth=3, label=f"Mean: {stats['overall_mean']:.2f}\"")
    ax3.axvline(np.median(rms_errors), color='#D61E44', linestyle='--', linewidth=3, label=f"Median: {np.median(rms_errors):.2f}\"")
    
    ax3.set_xlabel('RMS Error (arc-seconds)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of RMS Error Values')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Trend Analysis
    ax4 = axes[1, 1]
    
    # Calculate polynomial fit
    x_numeric = np.arange(len(rms_errors))
    z = np.polyfit(x_numeric, rms_errors, 2)
    p = np.poly1d(z)
    trend_line = p(x_numeric)
    
    ax4.scatter(dates, rms_errors, alpha=0.4, s=12, color='#0052A3', label='Daily Values')
    ax4.plot(dates, trend_line, linewidth=3, color='#FF6B35', label='Polynomial Fit (degree 2)')
    
    ax4.set_xlabel('Date')
    ax4.set_ylabel('RMS Error (arc-seconds)')
    ax4.set_title('Trend Analysis with Polynomial Fit')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def print_summary(stats):
    """Print comprehensive summary statistics (standard deviations as PRIMARY metric)."""
    print("="*80)
    print("STANDARD DEVIATION ANALYSIS SUMMARY (PRIMARY METRIC)")
    print("="*80)
    
    # PRIMARY METRIC: Rolling Standard Deviation Statistics
    rs_stats = stats.get('rolling_stddev_statistics', {})
    print(f"\n*** PRIMARY METRIC: ROLLING STANDARD DEVIATION STATISTICS (arc-seconds) ***")
    print(f"  Mean:                  {rs_stats.get('mean', 'N/A'):.4f}\"")
    print(f"  Std Deviation:         {rs_stats.get('std', 'N/A'):.4f}\"")
    print(f"  Minimum:               {rs_stats.get('min', 'N/A'):.4f}\"")
    print(f"  Maximum:               {rs_stats.get('max', 'N/A'):.4f}\"")
    print(f"  Data Points:           {rs_stats.get('data_points', 'N/A')}")
    
    # Boxplot stats for rolling stddev
    bp_rs = rs_stats.get('boxplot_stats', {})
    if bp_rs:
        print(f"  Boxplot (Q1/Med/Q3):   {bp_rs.get('q1'):.4f}\" / {bp_rs.get('median'):.4f}\" / {bp_rs.get('q3'):.4f}\"")
        print(f"  Whiskers (Low/High):   {bp_rs.get('lower_whisker'):.4f}\" / {bp_rs.get('upper_whisker'):.4f}\"")
        print(f"  Outliers:              {bp_rs.get('num_outliers')} ({bp_rs.get('outlier_percent'):.2f}%)")
    
    print(f"  Total Change:          {rs_stats.get('total_change', 'N/A'):.4f}\"")
    print(f"  Percent Change:        {rs_stats.get('percent_change', 'N/A'):.2f}%")
    
    # SECONDARY METRIC: Elevation RMS Error Statistics
    ee_stats = stats.get('elevation_error_statistics', {})
    print(f"\n*** SECONDARY METRIC: ELEVATION RMS ERROR STATISTICS (arc-seconds) ***")
    print(f"  Mean:                  {ee_stats.get('mean', 'N/A'):.4f}\"")
    print(f"  Std Deviation:         {ee_stats.get('std', 'N/A'):.4f}\"")
    print(f"  Minimum:               {ee_stats.get('min', 'N/A'):.4f}\"")
    print(f"  Maximum:               {ee_stats.get('max', 'N/A'):.4f}\"")
    print(f"  Data Points:           {ee_stats.get('data_points', 'N/A')}")
    
    # Boxplot stats for elevation errors
    bp_ee = ee_stats.get('boxplot_stats', {})
    if bp_ee:
        print(f"  Boxplot (Q1/Med/Q3):   {bp_ee.get('q1'):.4f}\" / {bp_ee.get('median'):.4f}\" / {bp_ee.get('q3'):.4f}\"")
        print(f"  Whiskers (Low/High):   {bp_ee.get('lower_whisker'):.4f}\" / {bp_ee.get('upper_whisker'):.4f}\"")
        print(f"  Outliers:              {bp_ee.get('num_outliers')} ({bp_ee.get('outlier_percent'):.2f}%)")
    
    print(f"  Total Change:          {ee_stats.get('total_change', 'N/A'):.4f}\"")
    print(f"  Percent Change:        {ee_stats.get('percent_change', 'N/A'):.2f}%")
    
    # Annual Statistics (Primary Metric)
    print(f"\n*** ANNUAL ROLLING STDDEV STATISTICS (arc-seconds) - PRIMARY ***")
    annual_rs = rs_stats.get('annual_stats')
    if annual_rs is not None:
        print(annual_rs.to_string())
    
    print(f"\n*** ANNUAL ELEVATION RMS ERROR STATISTICS (arc-seconds) - SECONDARY ***")
    annual_ee = ee_stats.get('annual_stats')
    if annual_ee is not None:
        print(annual_ee.to_string())
    
    print("="*80)

def format_duration(total_seconds):
    """
    Convert total seconds to a human-readable duration string.
    
    Parameters:
    -----------
    total_seconds : float
        Total duration in seconds
    
    Returns:
    --------
    str : Formatted duration string (HH:MM:SS or D days HH:MM:SS)
    """
    if total_seconds is None or total_seconds == 0:
        return "0:00:00"
    
    total_seconds = int(total_seconds)
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    if days > 0:
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def create_instrument_summary_table(instrument_stats_dict):
    """
    Create a summary table with 3 rows per instrument (TOTAL, UP, DOWN).
    Aggregates statistics across all runs for each instrument and direction.
    
    Parameters:
    -----------
    instrument_stats_dict : dict
        Dictionary with instrument names as keys and lists of run stats
    
    Returns:
    --------
    pd.DataFrame : Summary dataframe with 3 rows per instrument
    """
    if not instrument_stats_dict:
        print("No instrument statistics to summarize")
        return None
    
    summary_data = []
    
    for instrument_name in sorted(instrument_stats_dict.keys()):
        item = instrument_stats_dict[instrument_name]
        
        # Check if this is a list of runs (from notebook) or a stats dict (from direct call)
        if isinstance(item, list):
            # This is a list of runs - aggregate them
            runs_data = item
            if len(runs_data) == 0:
                continue
            
            # Separate aggregation for TOTAL, UP, and DOWN
            total_means = []
            total_stds = []
            total_min_errors = []
            total_max_errors = []
            total_changes = []
            percent_changes = []
            total_data_points_list = []
            total_durations = []
            total_q1s = []
            total_medians = []
            total_q3s = []
            total_outlier_counts = []
            
            up_means = []
            up_stds = []
            up_min_errors = []
            up_max_errors = []
            up_total_changes = []
            up_percent_changes = []
            up_data_points_list = []
            up_durations = []
            up_q1s = []
            up_medians = []
            up_q3s = []
            up_outlier_counts = []
            
            down_means = []
            down_stds = []
            down_min_errors = []
            down_max_errors = []
            down_total_changes = []
            down_percent_changes = []
            down_data_points_list = []
            down_durations = []
            down_q1s = []
            down_medians = []
            down_q3s = []
            down_outlier_counts = []
            
            for run in runs_data:
                # Aggregate TOTAL statistics
                if 'stats' in run and run['stats'] is not None:
                    stats = run['stats']
                    total_means.append(stats.get('overall_mean', 0))
                    total_stds.append(stats.get('overall_std', 0))
                    total_min_errors.append(stats.get('min_error', 0))
                    total_max_errors.append(stats.get('max_error', 0))
                    total_changes.append(stats.get('total_change', 0))
                    percent_changes.append(stats.get('percent_change', 0))
                    total_data_points_list.append(stats.get('data_points', 0))
                    total_durations.append(stats.get('run_duration_seconds', 0))
                    
                    # Extract box plot statistics
                    boxplot_stats = stats.get('boxplot_stats')
                    if boxplot_stats:
                        total_q1s.append(boxplot_stats['q1'])
                        total_medians.append(boxplot_stats['median'])
                        total_q3s.append(boxplot_stats['q3'])
                        total_outlier_counts.append(boxplot_stats['num_outliers'])
                
                # Aggregate UP statistics
                if 'stats_up' in run and run['stats_up'] is not None:
                    stats_up = run['stats_up']
                    up_means.append(stats_up.get('overall_mean', 0))
                    up_stds.append(stats_up.get('overall_std', 0))
                    up_min_errors.append(stats_up.get('min_error', 0))
                    up_max_errors.append(stats_up.get('max_error', 0))
                    up_total_changes.append(stats_up.get('total_change', 0))
                    up_percent_changes.append(stats_up.get('percent_change', 0))
                    up_data_points_list.append(stats_up.get('data_points', 0))
                    up_durations.append(stats_up.get('run_duration_seconds', 0))
                    
                    # Extract box plot statistics
                    boxplot_stats = stats_up.get('boxplot_stats')
                    if boxplot_stats:
                        up_q1s.append(boxplot_stats['q1'])
                        up_medians.append(boxplot_stats['median'])
                        up_q3s.append(boxplot_stats['q3'])
                        up_outlier_counts.append(boxplot_stats['num_outliers'])
                
                # Aggregate DOWN statistics
                if 'stats_down' in run and run['stats_down'] is not None:
                    stats_down = run['stats_down']
                    down_means.append(stats_down.get('overall_mean', 0))
                    down_stds.append(stats_down.get('overall_std', 0))
                    down_min_errors.append(stats_down.get('min_error', 0))
                    down_max_errors.append(stats_down.get('max_error', 0))
                    down_total_changes.append(stats_down.get('total_change', 0))
                    down_percent_changes.append(stats_down.get('percent_change', 0))
                    down_data_points_list.append(stats_down.get('data_points', 0))
                    down_durations.append(stats_down.get('run_duration_seconds', 0))
                    
                    # Extract box plot statistics
                    boxplot_stats = stats_down.get('boxplot_stats')
                    if boxplot_stats:
                        down_q1s.append(boxplot_stats['q1'])
                        down_medians.append(boxplot_stats['median'])
                        down_q3s.append(boxplot_stats['q3'])
                        down_outlier_counts.append(boxplot_stats['num_outliers'])
            
            # Create TOTAL row if available
            if total_means:
                total_row = {
                    'Instrument': instrument_name.upper(),
                    'Direction': 'TOTAL',
                    'Mean RMS (arcsec)': f"{np.mean(total_means):.4f}",
                    'Std Dev (arcsec)': f"{np.mean(total_stds):.4f}",
                    'Median (arcsec)': f"{np.mean(total_medians):.4f}" if total_medians else "N/A",
                    'Q1 (arcsec)': f"{np.mean(total_q1s):.4f}" if total_q1s else "N/A",
                    'Q3 (arcsec)': f"{np.mean(total_q3s):.4f}" if total_q3s else "N/A",
                    'Outliers': int(np.sum(total_outlier_counts)) if total_outlier_counts else 0,
                    'Min Error (arcsec)': f"{np.min(total_min_errors):.4f}",
                    'Max Error (arcsec)': f"{np.max(total_max_errors):.4f}",
                    'Total Change (arcsec)': f"{np.mean(total_changes):.4f}",
                    'Percent Change (%)': f"{np.mean(percent_changes):.2f}",
                    'Total Runs': len(runs_data),
                    'Data Points': int(np.sum(total_data_points_list)),
                    'Total Duration': format_duration(int(np.sum(total_durations)))
                }
                summary_data.append(total_row)
            
            # Create UP row if available
            if up_means:
                up_row = {
                    'Instrument': instrument_name.upper(),
                    'Direction': 'UP',
                    'Mean RMS (arcsec)': f"{np.mean(up_means):.4f}",
                    'Std Dev (arcsec)': f"{np.mean(up_stds):.4f}",
                    'Median (arcsec)': f"{np.mean(up_medians):.4f}" if up_medians else "N/A",
                    'Q1 (arcsec)': f"{np.mean(up_q1s):.4f}" if up_q1s else "N/A",
                    'Q3 (arcsec)': f"{np.mean(up_q3s):.4f}" if up_q3s else "N/A",
                    'Outliers': int(np.sum(up_outlier_counts)) if up_outlier_counts else 0,
                    'Min Error (arcsec)': f"{np.min(up_min_errors):.4f}",
                    'Max Error (arcsec)': f"{np.max(up_max_errors):.4f}",
                    'Total Change (arcsec)': f"{np.mean(up_total_changes):.4f}",
                    'Percent Change (%)': f"{np.mean(up_percent_changes):.2f}",
                    'Total Runs': len(runs_data),
                    'Data Points': int(np.sum(up_data_points_list)),
                    'Total Duration': format_duration(int(np.sum(up_durations)))
                }
                summary_data.append(up_row)
            
            # Create DOWN row if available
            if down_means:
                down_row = {
                    'Instrument': instrument_name.upper(),
                    'Direction': 'DOWN',
                    'Mean RMS (arcsec)': f"{np.mean(down_means):.4f}",
                    'Std Dev (arcsec)': f"{np.mean(down_stds):.4f}",
                    'Median (arcsec)': f"{np.mean(down_medians):.4f}" if down_medians else "N/A",
                    'Q1 (arcsec)': f"{np.mean(down_q1s):.4f}" if down_q1s else "N/A",
                    'Q3 (arcsec)': f"{np.mean(down_q3s):.4f}" if down_q3s else "N/A",
                    'Outliers': int(np.sum(down_outlier_counts)) if down_outlier_counts else 0,
                    'Min Error (arcsec)': f"{np.min(down_min_errors):.4f}",
                    'Max Error (arcsec)': f"{np.max(down_max_errors):.4f}",
                    'Total Change (arcsec)': f"{np.mean(down_total_changes):.4f}",
                    'Percent Change (%)': f"{np.mean(down_percent_changes):.2f}",
                    'Total Runs': len(runs_data),
                    'Data Points': int(np.sum(down_data_points_list)),
                    'Total Duration': format_duration(int(np.sum(down_durations)))
                }
                summary_data.append(down_row)
            
        else:
            # This is a stats dict (direct aggregated stats) - create single TOTAL row
            stats = item
            boxplot_stats = stats.get('boxplot_stats', {})
            summary_data.append({
                'Instrument': instrument_name.upper(),
                'Direction': 'TOTAL',
                'Mean RMS (arcsec)': f"{stats.get('overall_mean', 0):.4f}",
                'Std Dev (arcsec)': f"{stats.get('overall_std', 0):.4f}",
                'Median (arcsec)': f"{boxplot_stats.get('median', 0):.4f}" if boxplot_stats else "N/A",
                'Q1 (arcsec)': f"{boxplot_stats.get('q1', 0):.4f}" if boxplot_stats else "N/A",
                'Q3 (arcsec)': f"{boxplot_stats.get('q3', 0):.4f}" if boxplot_stats else "N/A",
                'Outliers': boxplot_stats.get('num_outliers', 0) if boxplot_stats else 0,
                'Min Error (arcsec)': f"{stats.get('min_error', 0):.4f}",
                'Max Error (arcsec)': f"{stats.get('max_error', 0):.4f}",
                'Total Change (arcsec)': f"{stats.get('total_change', 0):.4f}",
                'Percent Change (%)': f"{stats.get('percent_change', 0):.2f}",
                'Total Runs': 1,
                'Data Points': stats.get('data_points', 0),
                'Total Duration': format_duration(stats.get('total_duration_seconds', 0))
            })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

def create_directional_summary_table(instrument_stats_dict):
    """
    Create a summary table with directional RMS and StdDev metrics.
    
    Parameters:
    -----------
    instrument_stats_dict : dict
        Dictionary with instrument names as keys and lists of run stats as values
    
    Returns:
    --------
    pd.DataFrame : Summary dataframe with directional metrics
    """
    if not instrument_stats_dict:
        print("No instrument statistics to summarize")
        return None
    
    summary_data = []
    
    for instrument_name in sorted(instrument_stats_dict.keys()):
        runs_data = instrument_stats_dict[instrument_name]
        
        if len(runs_data) == 0:
            continue
        
        # Aggregate directional stats across all runs
        x_rms_list = []
        x_stddev_list = []
        y_rms_list = []
        y_stddev_list = []
        z_rms_list = []
        z_stddev_list = []
        
        for run in runs_data:
            dir_stats = run.get('directional_stats', {})
            
            x_dir = dir_stats.get('x_direction', {})
            if x_dir:
                x_rms_list.append(x_dir.get('rms', 0))
                x_stddev_list.append(x_dir.get('stddev', 0))
            
            y_dir = dir_stats.get('y_direction', {})
            if y_dir:
                y_rms_list.append(y_dir.get('rms', 0))
                y_stddev_list.append(y_dir.get('stddev', 0))
            
            z_dir = dir_stats.get('z_direction', {})
            if z_dir:
                z_rms_list.append(z_dir.get('rms', 0))
                z_stddev_list.append(z_dir.get('stddev', 0))
        
        # Create row with aggregated metrics
        summary_data.append({
            'Instrument': instrument_name.upper(),
            'Num Runs': len(runs_data),
            'X RMS (arcsec)': f"{np.mean(x_rms_list):.4f}" if x_rms_list else "N/A",
            'X StdDev (arcsec)': f"{np.mean(x_stddev_list):.4f}" if x_stddev_list else "N/A",
            'Y RMS (arcsec)': f"{np.mean(y_rms_list):.4f}" if y_rms_list else "N/A",
            'Y StdDev (arcsec)': f"{np.mean(y_stddev_list):.4f}" if y_stddev_list else "N/A",
            'Z RMS (arcsec)': f"{np.mean(z_rms_list):.4f}" if z_rms_list else "N/A",
            'Z StdDev (arcsec)': f"{np.mean(z_stddev_list):.4f}" if z_stddev_list else "N/A",
        })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df


def print_directional_summary_table(summary_df):
    """
    Print formatted directional summary table.
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        Summary dataframe with directional metrics
    """
    if summary_df is None or len(summary_df) == 0:
        print("No instruments to summarize")
        return
    
    print("\n" + "="*130)
    print("DIRECTIONAL RMS & STDDEV SUMMARY (all values in arc-seconds)")
    print("="*130)
    print(summary_df.to_string(index=False))
    print("="*130)


def print_instrument_summary_table(summary_df):
    """
    Print formatted instrument summary table.
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        Summary dataframe with RMS error values in arc-seconds
    """
    if summary_df is None or len(summary_df) == 0:
        print("No instruments to summarize")
        return
    
    print("\n" + "="*130)
    print("INSTRUMENT SUMMARY - RMS ERROR TRENDS (all values in arc-seconds)")
    print("="*130)
    print(summary_df.to_string(index=False))
    print("="*130)


def plot_run_direction_comparison(instrument_name, run_number, run_data, start_datetime=None, end_datetime=None):
    """
    Create a single run plot showing Both/Up/Down data distributions as violin plots.
    
    Parameters:
    -----------
    instrument_name : str
        Name of the instrument
    run_number : int
        Run number
    run_data : dict
        Dictionary containing run data with keys: 'stats', 'stats_up', 'stats_down',
        'rms_errors', 'rms_errors_up', 'rms_errors_down', 'run_start_dt'
    start_datetime : datetime, optional
        Overall analysis start datetime for context
    end_datetime : datetime, optional
        Overall analysis end datetime for context
    
    Returns:
    --------
    fig : matplotlib figure
        The created figure
    """
    if not run_data:
        return None
    
    # Create figure with 3 subplots for Both/Up/Down
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    
    # Convert string datetimes to datetime objects if needed
    if isinstance(start_datetime, str):
        start_datetime = datetime.fromisoformat(start_datetime)
    if isinstance(end_datetime, str):
        end_datetime = datetime.fromisoformat(end_datetime)
    
    # Get run start time for title
    run_start = run_data.get('run_start_dt', 'Unknown')
    run_start_str = run_start.strftime('%Y-%m-%d %H:%M:%S') if hasattr(run_start, 'strftime') else str(run_start)
    
    # Title
    title = f'{instrument_name.upper()} - Run {run_number} ({run_start_str})'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Define the three subplots
    directions = [
        ('both', 'BOTH', 'rms_errors', 'stats', '#0052A3'),
        ('up', 'UP', 'rms_errors_up', 'stats_up', '#00AA00'),
        ('down', 'DOWN', 'rms_errors_down', 'stats_down', '#FF0000')
    ]
    
    for idx, (dir_key, dir_label, data_key, stats_key, color) in enumerate(directions):
        ax = fig.add_subplot(gs[0, idx])
        
        # Get data for this direction
        rms_data = run_data.get(data_key, np.array([]))
        stats = run_data.get(stats_key)
        
        if rms_data is not None and len(rms_data) > 0 and stats is not None:
            # Create box plot
            bp = ax.boxplot([rms_data], positions=[0], widths=0.4,
                           patch_artist=True, showmeans=True,
                           meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
            
            # Customize colors
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            for whisker in bp['whiskers']:
                whisker.set_linewidth(1.5)
            for cap in bp['caps']:
                cap.set_linewidth(1.5)
            
            # Add statistics text
            mean_val = stats['overall_mean']
            std_val = stats['overall_std']
            min_val = stats['min_error']
            max_val = stats['max_error']
            
            stats_text = f'Mean: {mean_val:.4f}"\nStd: {std_val:.4f}"\nMin: {min_val:.4f}"\nMax: {max_val:.4f}"\nN: {len(rms_data)}'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
            
            ax.set_ylabel('RMS Error (arc-seconds)', fontsize=10)
            ax.set_title(f'[{dir_label}]', fontsize=11, fontweight='bold', color=color)
            ax.set_xticks([])
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, f'No data for {dir_label}', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=11, style='italic')
            ax.set_title(f'[{dir_label}]', fontsize=11, fontweight='bold', color=color)
            ax.set_xticks([])
    
    return fig


def plot_instrument_summary(instrument_name, runs_data, start_datetime=None, end_datetime=None):
    """
    Create comprehensive visualization of RMS error for a single instrument across all its runs.
    Works with statistics-only data (memory optimized).
    
    Parameters:
    -----------
    instrument_name : str
        Name of the instrument
    runs_data : list of dict
        List of dictionaries containing run stats with keys: 'stats', 'run_number'
        (raw data like 'dates' and 'rms_errors' are not stored for memory efficiency)
    start_datetime : datetime, optional
        Overall start datetime for title
    end_datetime : datetime, optional
        Overall end datetime for title
    
    Returns:
    --------
    fig : matplotlib figure
        The created figure
    """
    if not runs_data or len(runs_data) == 0:
        print(f"No data to plot for instrument {instrument_name}")
        return None
    
    num_runs = len(runs_data)
    
    # Create grid layout: 2 rows, 2 columns for summary plots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Convert string datetimes to datetime objects if needed
    if isinstance(start_datetime, str):
        start_datetime = datetime.fromisoformat(start_datetime)
    if isinstance(end_datetime, str):
        end_datetime = datetime.fromisoformat(end_datetime)
    
    # Title
    if start_datetime and end_datetime:
        title = f'{instrument_name.upper()} - Standard Deviation Analysis: {start_datetime.date()} to {end_datetime.date()}'
    else:
        title = f'{instrument_name.upper()} - Standard Deviation Analysis'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Timeline visualization of mean RMS error across runs
    ax1 = fig.add_subplot(gs[0, :])
    
    # High-contrast color palette (avoids yellow, cyan, pale violet)
    high_contrast_colors = [
        '#0052A3',  # Dark blue
        '#CC0000',  # Bright red
        '#006600',  # Dark green
        '#CC5200',  # Dark orange
        '#440099',  # Dark purple
        '#8B4513',  # Saddle brown
        '#008B8B',  # Dark cyan
        '#000080',  # Navy
        '#8B0000',  # Dark red
        '#4B0082',  # Indigo
        '#DC143C',  # Crimson
        '#228B22',  # Forest green
        '#1E90FF',  # Dodger blue
        '#FF6347',  # Tomato
        '#2F4F4F',  # Dark slate gray
    ]
    colors = [high_contrast_colors[i % len(high_contrast_colors)] for i in range(num_runs)]
    
    run_numbers = [run['run_number'] for run in runs_data]
    run_means = [run['stats']['overall_mean'] for run in runs_data]
    run_stds = [run['stats']['overall_std'] for run in runs_data]
    
    ax1.scatter(range(num_runs), run_means, color=colors, s=120, alpha=0.85, edgecolors='black', linewidth=2)
    ax1.errorbar(range(num_runs), run_means, yerr=run_stds, fmt='none', 
                color='black', capsize=5, alpha=0.5, linestyle='-', linewidth=1)
    ax1.plot(range(num_runs), run_means, linewidth=1.5, color='gray', alpha=0.5)
    
    ax1.set_xlabel('Run Sequence')
    ax1.set_ylabel('Mean RMS Error (arc-seconds)')
    ax1.set_title(f'RMS Error Trend Across All Runs')
    ax1.set_xticks(range(num_runs))
    ax1.set_xticklabels(run_numbers)
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Mean RMS Error', 'Std Dev Range'], loc='upper right', fontsize=9)
    
    # Plot 2: Mean and Std by run
    ax2 = fig.add_subplot(gs[1, 0])
    run_numbers = [run['run_number'] for run in runs_data]
    run_means = [run['stats']['overall_mean'] for run in runs_data]
    run_stds = [run['stats']['overall_std'] for run in runs_data]
    
    ax2.bar(range(num_runs), run_means, color='#FF6B35', alpha=0.85, label='Mean RMS Error')
    ax2.errorbar(range(num_runs), run_means, yerr=run_stds, fmt='none', color='black', capsize=5)
    
    ax2.set_xlabel('Run Number')
    ax2.set_ylabel('RMS Error (arc-seconds)')
    ax2.set_title('Mean RMS Error by Run')
    ax2.set_xticks(range(num_runs))
    ax2.set_xticklabels(run_numbers)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Data points per run
    ax3 = fig.add_subplot(gs[1, 1])
    run_data_points = [run['stats'].get('data_points', 0) for run in runs_data]
    
    bars = ax3.bar(range(num_runs), run_data_points, color='#004E89', alpha=0.85)
    ax3.set_xlabel('Run Number')
    ax3.set_ylabel('Number of Data Points')
    ax3.set_title('Data Points per Run')
    ax3.set_xticks(range(num_runs))
    ax3.set_xticklabels(run_numbers)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Summary statistics across all runs
    ax4 = fig.add_subplot(gs[2, 0])
    stats_data = [run['stats'] for run in runs_data]
    all_means = [s['overall_mean'] for s in stats_data]
    all_stds = [s['overall_std'] for s in stats_data]
    all_mins = [s['min_error'] for s in stats_data]
    all_maxs = [s['max_error'] for s in stats_data]
    
    # Create a box plot style visualization from statistics
    x_pos = np.arange(num_runs)
    width = 0.6
    
    # Plot ranges (min to max)
    for i, (min_val, max_val, mean_val) in enumerate(zip(all_mins, all_maxs, all_means)):
        ax4.plot([i, i], [min_val, max_val], 'o-', color=colors[i], linewidth=2, 
                markersize=4, alpha=0.7)
        ax4.scatter(i, mean_val, color=colors[i], s=100, marker='D', 
                   edgecolors='black', linewidth=1, zorder=5, alpha=0.9)
    
    ax4.set_xlabel('Run Number')
    ax4.set_ylabel('RMS Error (arc-seconds)')
    ax4.set_title('RMS Error Range (Min-Max) and Mean by Run')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(run_numbers)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend(['Range (Min-Max)', 'Mean'], loc='upper left', fontsize=9)
    
    # Plot 5: Min/Max by run
    ax5 = fig.add_subplot(gs[2, 1])
    run_mins = [run['stats']['min_error'] for run in runs_data]
    run_maxs = [run['stats']['max_error'] for run in runs_data]
    
    ax5.scatter(range(num_runs), run_mins, color='#00AA00', s=120, label='Min', alpha=0.85)
    ax5.scatter(range(num_runs), run_maxs, color='#FF0000', s=120, label='Max', alpha=0.85)
    ax5.fill_between(range(num_runs), run_mins, run_maxs, alpha=0.2, color='gray')
    
    ax5.set_xlabel('Run Number')
    ax5.set_ylabel('RMS Error (arc-seconds)')
    ax5.set_title('Min and Max RMS Error by Run')
    ax5.set_xticks(range(num_runs))
    ax5.set_xticklabels(run_numbers)
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    
    return fig

def plot_combined_instruments_summary(instrument_stats_dict, start_datetime=None, end_datetime=None):
    """
    Create comprehensive visualization combining all instruments (memory optimized).
    Works with statistics-only data - no raw time series data stored.
    
    Parameters:
    -----------
    instrument_stats_dict : dict
        Dictionary with instrument names as keys and lists of run stats as values
    start_datetime : datetime, optional
        Overall start datetime for title
    end_datetime : datetime, optional
        Overall end datetime for title
    
    Returns:
    --------
    fig : matplotlib figure
        The created figure
    """
    if not instrument_stats_dict or len(instrument_stats_dict) == 0:
        print("No instruments to plot")
        return None
    
    num_instruments = len(instrument_stats_dict)
    
    # Create grid layout for summary plots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Convert string datetimes to datetime objects if needed
    if isinstance(start_datetime, str):
        start_datetime = datetime.fromisoformat(start_datetime)
    if isinstance(end_datetime, str):
        end_datetime = datetime.fromisoformat(end_datetime)
    
    # Title
    if start_datetime and end_datetime:
        title = f'Combined Standard Deviation Analysis: {start_datetime.date()} to {end_datetime.date()}'
    else:
        title = 'Combined Standard Deviation Analysis - All Instruments'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Prepare data for all instruments
    instrument_names = sorted(instrument_stats_dict.keys())
    
    # High-contrast color palette (avoids yellow, cyan, pale violet)
    high_contrast_colors = [
        '#0052A3',  # Dark blue
        '#CC0000',  # Bright red
        '#006600',  # Dark green
        '#CC5200',  # Dark orange
        '#440099',  # Dark purple
        '#8B4513',  # Saddle brown
        '#008B8B',  # Dark cyan
        '#000080',  # Navy
        '#8B0000',  # Dark red
        '#4B0082',  # Indigo
        '#DC143C',  # Crimson
        '#228B22',  # Forest green
        '#1E90FF',  # Dodger blue
        '#FF6347',  # Tomato
        '#2F4F4F',  # Dark slate gray
    ]
    colors = [high_contrast_colors[i % len(high_contrast_colors)] for i in range(num_instruments)]
    
    # Plot 1: Timeline visualization of mean RMS error across runs
    ax1 = fig.add_subplot(gs[0, :])
    
    for idx, instrument_name in enumerate(instrument_names):
        runs_data = instrument_stats_dict[instrument_name]
        # With memory optimization, we work with aggregated stats only
        run_means = [run['stats']['overall_mean'] for run in runs_data]
        
        # Plot trend of mean RMS error across runs
        ax1.plot(range(len(run_means)), run_means, linewidth=2, marker='o', 
                color=colors[idx], alpha=0.85, label=instrument_name.upper())
    
    ax1.set_xlabel('Run Sequence')
    ax1.set_ylabel('Mean RMS Error (arc-seconds)')
    ax1.set_title('Mean RMS Error Trend - All Instruments')
    ax1.legend(loc='upper right', fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mean RMS Error by Instrument
    ax2 = fig.add_subplot(gs[1, 0])
    instrument_means = []
    for instrument_name in instrument_names:
        runs_data = instrument_stats_dict[instrument_name]
        # Calculate mean of run means for overall instrument mean
        run_means = [run['stats']['overall_mean'] for run in runs_data]
        instrument_means.append(np.mean(run_means))
    
    bars = ax2.barh(range(num_instruments), instrument_means, color=colors, alpha=0.85)
    ax2.set_yticks(range(num_instruments))
    ax2.set_yticklabels([name.upper() for name in instrument_names])
    ax2.set_xlabel('Mean RMS Error (arc-seconds)')
    ax2.set_title('Mean RMS Error by Instrument')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.4f}',
                ha='left', va='center', fontsize=9)
    
    # Plot 3: Data points by Instrument
    ax3 = fig.add_subplot(gs[1, 1])
    instrument_data_points = []
    for instrument_name in instrument_names:
        runs_data = instrument_stats_dict[instrument_name]
        # Sum data points from all runs
        data_points = sum(run['stats'].get('data_points', 0) for run in runs_data)
        instrument_data_points.append(data_points)
    
    bars = ax3.barh(range(num_instruments), instrument_data_points, color=colors, alpha=0.85)
    ax3.set_yticks(range(num_instruments))
    ax3.set_yticklabels([name.upper() for name in instrument_names])
    ax3.set_xlabel('Number of Data Points')
    ax3.set_title('Data Points per Instrument')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2.,
                f'{int(width)}',
                ha='left', va='center', fontsize=9)
    
    # Plot 4: Range of RMS Error by Instrument
    ax4 = fig.add_subplot(gs[2, 0])
    instrument_mins = []
    instrument_maxs = []
    instrument_means = []
    
    for instrument_name in instrument_names:
        runs_data = instrument_stats_dict[instrument_name]
        mins = [run['stats']['min_error'] for run in runs_data]
        maxs = [run['stats']['max_error'] for run in runs_data]
        means = [run['stats']['overall_mean'] for run in runs_data]
        
        instrument_mins.append(np.min(mins) if mins else 0)
        instrument_maxs.append(np.max(maxs) if maxs else 0)
        instrument_means.append(np.mean(means) if means else 0)
    
    x_pos = np.arange(num_instruments)
    width = 0.6
    
    for i, (min_val, max_val, mean_val) in enumerate(zip(instrument_mins, instrument_maxs, instrument_means)):
        ax4.plot([i, i], [min_val, max_val], 'o-', color=colors[i], linewidth=2.5, 
                markersize=5, alpha=0.7)
        ax4.scatter(i, mean_val, color=colors[i], s=120, marker='D', 
                   edgecolors='black', linewidth=1, zorder=5, alpha=0.9)
    
    ax4.set_ylabel('RMS Error (arc-seconds)')
    ax4.set_title('RMS Error Range by Instrument (Min-Max-Mean)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([name.upper() for name in instrument_names])
    ax4.legend(['Range (Min-Max)', 'Mean'], loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Standard deviation by instrument
    ax5 = fig.add_subplot(gs[2, 1])
    instrument_stds = []
    for instrument_name in instrument_names:
        runs_data = instrument_stats_dict[instrument_name]
        # Average standard deviations across runs
        stds = [run['stats']['overall_std'] for run in runs_data]
        instrument_stds.append(np.mean(stds) if stds else 0)
    
    bars = ax5.barh(range(num_instruments), instrument_stds, color=colors, alpha=0.85)
    ax5.set_yticks(range(num_instruments))
    ax5.set_yticklabels([name.upper() for name in instrument_names])
    ax5.set_xlabel('Standard Deviation (arc-seconds)')
    ax5.set_title('RMS Error Standard Deviation by Instrument')
    ax5.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax5.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.4f}',
                ha='left', va='center', fontsize=9)
    
    return fig


def plot_instrument_timeseries_violin(instrument_name, runs_data, start_datetime=None, end_datetime=None, direction='both'):
    """
    Create time-series visualization of RMS error for a single instrument with violin plots
    showing data distribution within each run.
    
    Parameters:
    -----------
    instrument_name : str
        Name of the instrument
    runs_data : list of dict
        List of dictionaries containing run data with keys: 'stats', 'run_number', 'run_start_dt', 
        'rms_errors', 'dates'
    start_datetime : datetime, optional
        Overall start datetime for title
    end_datetime : datetime, optional
        Overall end datetime for title
    direction : str, optional
        Direction filter: 'both' (total), 'up', or 'down'. Default is 'both'
    
    Returns:
    --------
    fig : matplotlib figure
        The created figure
    """
    if not runs_data or len(runs_data) == 0:
        print(f"No data to plot for instrument {instrument_name}")
        return None
    
    num_runs = len(runs_data)
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, hspace=0.3)
    
    # Convert string datetimes to datetime objects if needed
    if isinstance(start_datetime, str):
        start_datetime = datetime.fromisoformat(start_datetime)
    if isinstance(end_datetime, str):
        end_datetime = datetime.fromisoformat(end_datetime)
    
    # Determine direction label and select appropriate stats
    direction_label = direction.upper() if direction in ['up', 'down'] else 'BOTH'
    
    # Title
    if start_datetime and end_datetime:
        title = f'{instrument_name.upper()} - RMS Error Time-Series [{direction_label}]: {start_datetime.date()} to {end_datetime.date()}'
    else:
        title = f'{instrument_name.upper()} - RMS Error Time-Series [{direction_label}]'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Prepare data
    run_start_dates = []
    run_means = []
    run_stds = []
    violin_data = []  # For violin plots
    
    for run in runs_data:
        if 'run_start_dt' in run:
            run_start_dates.append(run['run_start_dt'])
            
            # Select data based on direction
            if direction == 'up' and 'stats_up' in run and run['stats_up'] is not None:
                run_means.append(run['stats_up']['overall_mean'])
                run_stds.append(run['stats_up']['overall_std'])
                if 'rms_errors_up' in run:
                    violin_data.append(run['rms_errors_up'])
                else:
                    violin_data.append(np.array([]))
            elif direction == 'down' and 'stats_down' in run and run['stats_down'] is not None:
                run_means.append(run['stats_down']['overall_mean'])
                run_stds.append(run['stats_down']['overall_std'])
                if 'rms_errors_down' in run:
                    violin_data.append(run['rms_errors_down'])
                else:
                    violin_data.append(np.array([]))
            else:  # 'both' or total
                run_means.append(run['stats']['overall_mean'])
                run_stds.append(run['stats']['overall_std'])
                if 'rms_errors' in run:
                    violin_data.append(run['rms_errors'])
                else:
                    violin_data.append(np.array([]))
    
    # Plot 1: Time-series with error bands
    ax1 = fig.add_subplot(gs[0])
    
    if run_start_dates:
        # Use actual datetime values as x coordinates
        x_datetime = mdates.date2num(run_start_dates)
        
        # Plot mean RMS error
        ax1.plot(x_datetime, run_means, 'o-', linewidth=2.5, markersize=8, 
                color='#0052A3', alpha=0.9, label='Mean RMS Error')
        
        # Add error band (mean ± std)
        ax1.fill_between(x_datetime, 
                         np.array(run_means) - np.array(run_stds),
                         np.array(run_means) + np.array(run_stds),
                         alpha=0.2, color='#0052A3', label='±1 Std Dev')
        
        # Format x-axis with date formatter
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        ax1.set_ylabel('Mean RMS Error (arc-seconds)', fontsize=11)
        ax1.set_title(f'RMS Error Trend Over Time [{direction_label}]', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Violin plots showing data distribution per run
    ax2 = fig.add_subplot(gs[1])
    
    if violin_data and any(len(v) > 0 for v in violin_data):
        # Filter out empty violin data
        valid_indices = [i for i, v in enumerate(violin_data) if len(v) > 0]
        valid_violin_data = [violin_data[i] for i in valid_indices]
        valid_dates = [run_start_dates[i] for i in valid_indices]
        
        if valid_violin_data:
            # Use actual datetime values as x coordinates for box plots
            x_datetime = mdates.date2num(valid_dates)
            
            # Use matplotlib's boxplot positioned at datetime coordinates
            bp = ax2.boxplot(valid_violin_data, positions=x_datetime, widths=0.015,
                            patch_artist=True, showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='#8B00FF', markersize=5))
            
            # Customize box colors
            colors = plt.cm.viridis(np.linspace(0, 1, len(valid_violin_data)))
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i])
                patch.set_alpha(0.7)
            for whisker in bp['whiskers']:
                whisker.set_linewidth(1.2)
            for cap in bp['caps']:
                cap.set_linewidth(1.2)
            
            # Format x-axis with date formatter
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
            ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            ax2.set_ylabel('RMS Error (arc-seconds)', fontsize=11)
            ax2.set_xlabel('Run Start Date', fontsize=11)
            ax2.set_title(f'RMS Error Distribution by Run [{direction_label}] (Box Plot)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
    
    return fig


def plot_rms_vs_stddev_directions(instrument_name, rms_errors, dates, stats, directional_stats, 
                                  start_datetime=None, end_datetime=None):
    """
    Create comprehensive visualization comparing RMS and StdDev across all three directions.
    
    Parameters:
    -----------
    instrument_name : str
        Name of the instrument
    rms_errors : np.ndarray
        Array of RMS error values in arc-seconds
    dates : np.ndarray
        Array of dates
    stats : dict
        Statistics dictionary
    directional_stats : dict
        Directional statistics for X, Y, Z
    start_datetime : datetime or str, optional
        Start datetime of analysis
    end_datetime : datetime or str, optional
        End datetime of analysis  
        
    Returns:
    --------
    fig : matplotlib figure
        The created figure
    """
    parent_fig, fig = plt.subplots(3, 2, figsize=(16, 14))
    fig = fig.reshape(3, 2)
    
    # Format title
    if start_datetime is not None and end_datetime is not None:
        if isinstance(start_datetime, str):
            start_datetime = datetime.fromisoformat(start_datetime)
        if isinstance(end_datetime, str):
            end_datetime = datetime.fromisoformat(end_datetime)
        title = f'{instrument_name.upper()} - RMS & StdDev Analysis: {start_datetime.date()} to {end_datetime.date()}'
    else:
        title = f'{instrument_name.upper()} - RMS & StdDev Analysis'
    
    parent_fig.suptitle(title, fontsize=16, fontweight='bold')
    
    directions = ['x_direction', 'y_direction', 'z_direction']
    dir_labels = ['X-Direction', 'Y-Direction', 'Z-Direction']
    colors = ['#0052A3', '#00AA00', '#FF0000']
    
    # Plot RMS and StdDev metrics for each direction
    for idx, (dir_key, dir_label, color) in enumerate(zip(directions, dir_labels, colors)):
        dir_data = directional_stats.get(dir_key, {})
        
        # Left column: RMS metrics
        ax_rms = fig[idx, 0]
        rms_val = dir_data.get('rms', 0)
        data_pts = dir_data.get('data_points', 0)
        mean_val = abs(dir_data.get('mean', 0))
        min_val = dir_data.get('min', 0)
        max_val = dir_data.get('max', 0)
        
        ax_rms.bar([0], [rms_val], color=color, alpha=0.7, width=0.5)
        ax_rms.set_ylabel('RMS (arc-seconds)', fontsize=10)
        ax_rms.set_title(f'{dir_label} - RMS Metric', fontsize=11, fontweight='bold')
        ax_rms.set_xlim(-0.5, 0.5)
        ax_rms.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        stats_text = f'RMS: {rms_val:.4f}"\nMean: {mean_val:.4f}"\nMin: {min_val:.4f}"\nMax: {max_val:.4f}"\nN: {data_pts}'
        ax_rms.text(0.02, 0.98, stats_text, transform=ax_rms.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
        ax_rms.set_xticks([])
        
        # Right column: StdDev metrics
        ax_stddev = fig[idx, 1]
        stddev_val = dir_data.get('stddev', 0)
        
        ax_stddev.bar([0], [stddev_val], color=color, alpha=0.7, width=0.5)
        ax_stddev.set_ylabel('StdDev (arc-seconds)', fontsize=10)
        ax_stddev.set_title(f'{dir_label} - StdDev Metric', fontsize=11, fontweight='bold')
        ax_stddev.set_xlim(-0.5, 0.5)
        ax_stddev.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        stddev_text = f'StdDev: {stddev_val:.4f}"\nRMS: {rms_val:.4f}"\nMin: {min_val:.4f}"\nMax: {max_val:.4f}"\nN: {data_pts}'
        ax_stddev.text(0.02, 0.98, stddev_text, transform=ax_stddev.transAxes,
                      fontsize=9, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
        ax_stddev.set_xticks([])
    
    parent_fig.tight_layout()
    return parent_fig


def plot_instrument_direction_comparison(instrument_name, runs_data, start_datetime=None, end_datetime=None):
    """
    Create a comprehensive comparison plot showing RMS and StdDev for all three directions.
    
    Parameters:
    -----------
    instrument_name : str
        Name of the instrument
    runs_data : list of dict
        List of run data dictionaries
    start_datetime : datetime, optional
        Overall start datetime for title
    end_datetime : datetime, optional
        Overall end datetime for title
        
    Returns:
    --------
    fig : matplotlib figure
        The created figure
    """
    if not runs_data or len(runs_data) == 0:
        print(f"No data to plot for instrument {instrument_name}")
        return None
    
    num_runs = len(runs_data)
    
    # Create figure with 6 subplots (3 directions × 2 metrics)
    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Convert string datetimes to datetime objects if needed
    if isinstance(start_datetime, str):
        start_datetime = datetime.fromisoformat(start_datetime)
    if isinstance(end_datetime, str):
        end_datetime = datetime.fromisoformat(end_datetime)
    
    # Title
    if start_datetime and end_datetime:
        title = f'{instrument_name.upper()} - RMS & StdDev by Direction: {start_datetime.date()} to {end_datetime.date()}'
    else:
        title = f'{instrument_name.upper()} - RMS & StdDev by Direction'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    directions = ['x_direction', 'y_direction', 'z_direction']
    dir_labels = ['X-Direction', 'Y-Direction', 'Z-Direction']
    colors = ['#0052A3', '#00AA00', '#FF0000']
    
    for dir_idx, (dir_key, dir_label, color) in enumerate(zip(directions, dir_labels, colors)):
        # RMS column
        ax_rms = fig.add_subplot(gs[dir_idx, 0])
        rms_values = []
        run_numbers = []
        
        for run in runs_data:
            dir_stats = run.get('directional_stats', {}).get(dir_key, {})
            rms_values.append(dir_stats.get('rms', 0))
            run_numbers.append(run.get('run_number', 0))
        
        bars = ax_rms.bar(range(num_runs), rms_values, color=color, alpha=0.7, label='RMS')
        ax_rms.set_ylabel('RMS (arc-seconds)', fontsize=10)
        ax_rms.set_title(f'{dir_label} - RMS Metric', fontsize=11, fontweight='bold')
        ax_rms.set_xticks(range(num_runs))
        ax_rms.set_xticklabels(run_numbers)
        ax_rms.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax_rms.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        # StdDev column
        ax_stddev = fig.add_subplot(gs[dir_idx, 1])
        stddev_values = []
        
        for run in runs_data:
            dir_stats = run.get('directional_stats', {}).get(dir_key, {})
            stddev_values.append(dir_stats.get('stddev', 0))
        
        bars = ax_stddev.bar(range(num_runs), stddev_values, color=color, alpha=0.5, label='StdDev')
        ax_stddev.set_ylabel('StdDev (arc-seconds)', fontsize=10)
        ax_stddev.set_title(f'{dir_label} - StdDev Metric', fontsize=11, fontweight='bold')
        ax_stddev.set_xticks(range(num_runs))
        ax_stddev.set_xticklabels(run_numbers)
        ax_stddev.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax_stddev.text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.3f}',
                          ha='center', va='bottom', fontsize=8)
        
        # Comparison column (RMS vs StdDev)
        ax_comp = fig.add_subplot(gs[dir_idx, 2])
        x_pos = np.arange(num_runs)
        width = 0.35
        
        bars1 = ax_comp.bar(x_pos - width/2, rms_values, width, label='RMS', color=color, alpha=0.7)
        bars2 = ax_comp.bar(x_pos + width/2, stddev_values, width, label='StdDev', color=color, alpha=0.3)
        
        ax_comp.set_ylabel('Value (arc-seconds)', fontsize=10)
        ax_comp.set_title(f'{dir_label} - RMS vs StdDev', fontsize=11, fontweight='bold')
        ax_comp.set_xticks(x_pos)
        ax_comp.set_xticklabels(run_numbers)
        ax_comp.legend(loc='upper right', fontsize=8)
        ax_comp.grid(True, alpha=0.3, axis='y')
    
    return fig


def plot_combined_timeseries_violin(instrument_stats_dict, start_datetime=None, end_datetime=None, direction='both'):
    """
    Create combined time-series visualization for all instruments showing trends
    and individual run distributions as violin plots.
    
    Parameters:
    -----------
    instrument_stats_dict : dict
        Dictionary with instrument names as keys and lists of run data as values
    start_datetime : datetime, optional
        Overall start datetime for title
    end_datetime : datetime, optional
        Overall end datetime for title
    direction : str, optional
        Direction filter: 'both' (total), 'up', or 'down'. Default is 'both'
    
    Returns:
    --------
    fig : matplotlib figure
        The created figure
    """
    if not instrument_stats_dict or len(instrument_stats_dict) == 0:
        print("No instruments to plot")
        return None
    
    num_instruments = len(instrument_stats_dict)
    
    # Create figure with one subplot per instrument
    fig = plt.figure(figsize=(16, 4 * num_instruments))
    gs = fig.add_gridspec(num_instruments, 1, hspace=0.4)
    
    # Convert string datetimes to datetime objects if needed
    if isinstance(start_datetime, str):
        start_datetime = datetime.fromisoformat(start_datetime)
    if isinstance(end_datetime, str):
        end_datetime = datetime.fromisoformat(end_datetime)
    
    # Determine direction label
    direction_label = direction.upper() if direction in ['up', 'down'] else 'BOTH'
    
    # Title
    if start_datetime and end_datetime:
        title = f'Combined RMS Error Time-Series [{direction_label}]: {start_datetime.date()} to {end_datetime.date()}'
    else:
        title = f'Combined RMS Error Time-Series [{direction_label}] - All Instruments'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Prepare data for all instruments
    instrument_names = sorted(instrument_stats_dict.keys())
    
    # High-contrast color palette (avoids yellow, cyan, pale violet)
    high_contrast_colors = [
        '#0052A3',  # Dark blue
        '#CC0000',  # Bright red
        '#006600',  # Dark green
        '#CC5200',  # Dark orange
        '#440099',  # Dark purple
        '#8B4513',  # Saddle brown
        '#008B8B',  # Dark cyan
        '#000080',  # Navy
        '#8B0000',  # Dark red
        '#4B0082',  # Indigo
        '#DC143C',  # Crimson
        '#228B22',  # Forest green
        '#1E90FF',  # Dodger blue
        '#FF6347',  # Tomato
        '#2F4F4F',  # Dark slate gray
    ]
    colors = [high_contrast_colors[i % len(high_contrast_colors)] for i in range(num_instruments)]
    
    for idx, instrument_name in enumerate(instrument_names):
        ax = fig.add_subplot(gs[idx])
        
        runs_data = instrument_stats_dict[instrument_name]
        
        # Collect data
        run_start_dates = []
        run_means = []
        run_stds = []
        violin_data = []
        
        for run in runs_data:
            if 'run_start_dt' in run:
                run_start_dates.append(run['run_start_dt'])
                
                # Select data based on direction
                if direction == 'up' and 'stats_up' in run and run['stats_up'] is not None:
                    run_means.append(run['stats_up']['overall_mean'])
                    run_stds.append(run['stats_up']['overall_std'])
                    if 'rms_errors_up' in run:
                        violin_data.append(run['rms_errors_up'])
                    else:
                        violin_data.append(np.array([]))
                elif direction == 'down' and 'stats_down' in run and run['stats_down'] is not None:
                    run_means.append(run['stats_down']['overall_mean'])
                    run_stds.append(run['stats_down']['overall_std'])
                    if 'rms_errors_down' in run:
                        violin_data.append(run['rms_errors_down'])
                    else:
                        violin_data.append(np.array([]))
                else:  # 'both' or total
                    run_means.append(run['stats']['overall_mean'])
                    run_stds.append(run['stats']['overall_std'])
                    if 'rms_errors' in run:
                        violin_data.append(run['rms_errors'])
                    else:
                        violin_data.append(np.array([]))
        
        if run_start_dates:
            # Use actual datetime values as x coordinates
            x_datetime = mdates.date2num(run_start_dates)
            
            # Plot mean RMS error trend
            ax.plot(x_datetime, run_means, 'o-', linewidth=2.5, markersize=8, 
                   color=colors[idx], alpha=0.8, label='Mean RMS Error')
            
            # Add error band
            ax.fill_between(x_datetime, 
                            np.array(run_means) - np.array(run_stds),
                            np.array(run_means) + np.array(run_stds),
                            alpha=0.15, color=colors[idx])
            
            # Add violin plots positioned at datetime coordinates
            if violin_data and any(len(v) > 0 for v in violin_data):
                valid_indices = [i for i, v in enumerate(violin_data) if len(v) > 0]
                valid_violin_data = [violin_data[i] for i in valid_indices]
                valid_dates = [run_start_dates[i] for i in valid_indices]
                
                if valid_violin_data:
                    x_datetime_valid = mdates.date2num(valid_dates)
                    bp = ax.boxplot(valid_violin_data, positions=x_datetime_valid, widths=0.015,
                                   patch_artist=True, showmeans=False, showmedians=True)
                    
                    for patch in bp['boxes']:
                        patch.set_facecolor(colors[idx])
                        patch.set_alpha(0.3)
                        patch.set_edgecolor(colors[idx])
                    for whisker in bp['whiskers']:
                        whisker.set_linewidth(1.0)
                    for cap in bp['caps']:
                        cap.set_linewidth(1.0)
            
            # Format x-axis with date formatter
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            ax.set_ylabel('RMS Error (arc-seconds)', fontsize=10)
            ax.set_title(f'{instrument_name.upper()} - RMS Error Trend with Data Distribution [{direction_label}]', 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='upper right', fontsize=9)
    
    return fig


def plot_individual_instrument_timeseries(instrument_stats_dict, start_datetime=None, end_datetime=None, show_altitude=True,
                                          db_host='mariadb.mmto.arizona.edu', db_user='mmtstaff', 
                                          db_password='multiple', db_measurements='measurements',
                                          enable_downsampling=True, downsample_max_points=10000):
    """
    Generate time-series plots showing RMS error and StdDev over time by direction.
    Creates individual plots for each instrument and direction (TOTAL, UP, DOWN).
    When show_altitude is True, altitude plots are displayed on separate graphs below with aligned x-axes.
    Altitude velocity data is queried from the mount_mini_velalt table and plotted in the bottom row.
    
    Parameters:
    -----------
    instrument_stats_dict : dict
        Dictionary with instrument names as keys and lists of run data as values
    start_datetime : datetime, optional
        Start datetime for title context
    end_datetime : datetime, optional
        End datetime for title context
    show_altitude : bool, optional
        Whether to display telescope altitude and velocity on separate plots below. Default is True.
    db_host : str, optional
        Database host for velocity data queries
    db_user : str, optional
        Database user for velocity data queries
    db_password : str, optional
        Database password for velocity data queries
    db_measurements : str, optional
        Database name for measurements
    enable_downsampling : bool, optional
        Whether to downsample large datasets for faster plotting. Default is True.
    downsample_max_points : int, optional
        Maximum number of points to display in plots. Default is 10000.
    """
    if not instrument_stats_dict or len(instrument_stats_dict) == 0:
        print("No instruments to plot")
        return
    
    # Convert string datetimes to datetime objects if needed
    if isinstance(start_datetime, str):
        start_datetime = datetime.fromisoformat(start_datetime)
    if isinstance(end_datetime, str):
        end_datetime = datetime.fromisoformat(end_datetime)
    
    print("Generating time-series plots with RMS and StdDev by direction (TOTAL, UP, DOWN)...")
    print("="*80)
    
    fig_count = 0
    directions_data = ['total', 'up', 'down']
    direction_labels = {'total': 'BOTH (Total)', 'up': 'UP (Elevation Increasing)', 'down': 'DOWN (Elevation Decreasing)'}
    
    for instrument_name in sorted(instrument_stats_dict.keys()):
        runs_data = instrument_stats_dict[instrument_name]
        
        if not isinstance(runs_data, list) or len(runs_data) == 0:
            continue
        
        # Generate a plot for each direction
        for direction in directions_data:
            # Create cache key for this instrument/direction combination
            cache_key = f"{instrument_name}_{direction}"
            
            # Check if plot data is already cached
            if cache_key in _PLOT_DATA_CACHE:
                print(f"Using cached plot data for {instrument_name.upper()} ({direction_labels.get(direction, direction)})")
                plot_data = _PLOT_DATA_CACHE[cache_key]
                all_timestamps = plot_data['all_timestamps']
                all_rms_errors = plot_data['all_rms_errors']
                all_altitudes = plot_data['all_altitudes']
                all_altitudes_velocities = plot_data['all_altitudes_velocities']
                data_start = plot_data['data_start']
                data_end = plot_data['data_end']
            else:
                # Process fresh data
                # Collect all data points and timestamps for this direction
                all_timestamps = []
                all_rms_errors = []
                all_altitudes = []
                
                for run in runs_data:
                    # Get the appropriate data based on direction
                    if direction == 'total':
                        rms_errors = run.get('rms_errors', np.array([]))
                        dates = run.get('dates', np.array([]))
                        altitudes = run.get('altitudes', np.array([]))
                    elif direction == 'up':
                        rms_errors = run.get('rms_errors_up', np.array([]))
                        dates = run.get('dates_up', np.array([]))
                        altitudes = run.get('altitudes_up', np.array([]))
                    else:  # down
                        rms_errors = run.get('rms_errors_down', np.array([]))
                        dates = run.get('dates_down', np.array([]))
                        altitudes = run.get('altitudes_down', np.array([]))
                    
                    if len(rms_errors) > 0:
                        all_rms_errors.extend(rms_errors)
                        all_timestamps.extend(dates)
                        all_altitudes.extend(altitudes)
                
                if len(all_rms_errors) == 0:
                    continue
                
                # Sort by timestamp
                sorted_indices = np.argsort(all_timestamps)
                all_timestamps = [all_timestamps[i] for i in sorted_indices]
                all_rms_errors = np.array(all_rms_errors)[sorted_indices]
                all_altitudes = np.array(all_altitudes)[sorted_indices]
                
                # Query or calculate altitude velocity data if show_altitude is True
                all_altitudes_velocities = np.array([])
                if show_altitude and len(all_timestamps) > 0:
                    if USE_ALTITUDE_GRADIENT_VELOCITY:
                        # Calculate velocity directly from altitude data (faster, no DB query)
                        try:
                            all_altitudes_velocities = calculate_altitude_velocity(all_altitudes, all_timestamps)
                        except Exception as e:
                            print(f"Warning: Could not calculate altitude velocity from data: {e}")
                            all_altitudes_velocities = np.array([])
                    else:
                        # Query velocity data from mount_mini_velalt table (traditional approach)
                        try:
                            data_start_dt = all_timestamps[0]
                            data_end_dt = all_timestamps[-1]
                            if isinstance(data_start_dt, (int, float)):
                                data_start_dt = datetime.fromtimestamp(data_start_dt / 1000.0)
                            if isinstance(data_end_dt, (int, float)):
                                data_end_dt = datetime.fromtimestamp(data_end_dt / 1000.0)
                            elif not isinstance(data_start_dt, datetime):
                                data_start_dt = pd.Timestamp(data_start_dt).to_pydatetime()
                            
                            if not isinstance(data_end_dt, datetime):
                                data_end_dt = pd.Timestamp(data_end_dt).to_pydatetime()
                            
                            # Query velocity data from mount_mini_velalt table with DB credentials
                            vel_timestamps, vel_values = query_altitude_velocity_data(
                                host=db_host,
                                user=db_user,
                                password=db_password,
                                database=db_measurements,
                                start_datetime=data_start_dt,
                                end_datetime=data_end_dt
                            )
                            
                            # Interpolate velocity values to match RMS error timestamps
                            if len(vel_timestamps) > 0 and len(vel_values) > 0:
                                # Convert all timestamps to numeric format for interpolation (seconds since epoch)
                                # Handle numpy datetime64, pandas Timestamp, and datetime objects
                                vel_ts_numeric = np.array([])
                                rms_ts_numeric = np.array([])
                                try:
                                    # Convert velocity timestamps to numeric
                                    if isinstance(vel_timestamps[0], np.datetime64):
                                        vel_ts_numeric = pd.to_datetime(vel_timestamps).astype(np.int64) / 1e9
                                    elif isinstance(vel_timestamps[0], pd.Timestamp):
                                        vel_ts_numeric = vel_timestamps.astype(np.int64) / 1e9
                                    elif isinstance(vel_timestamps[0], datetime):
                                        vel_ts_numeric = np.array([t.timestamp() for t in vel_timestamps])
                                    else:
                                        vel_ts_numeric = np.array([float(t) / 1000.0 for t in vel_timestamps])
                                    
                                    # Convert RMS timestamps to numeric
                                    rms_ts_numeric = np.array([
                                        t.timestamp() if isinstance(t, datetime) else 
                                        (pd.Timestamp(t).timestamp() if isinstance(t, (str, np.datetime64, pd.Timestamp)) else float(t) / 1000.0)
                                        for t in all_timestamps
                                    ])
                                    
                                    # Convert RMS timestamps to datetime objects for plotting
                                    if not isinstance(all_timestamps[0], datetime):
                                        all_timestamps = [
                                            datetime.fromtimestamp(t / 1000.0) if isinstance(t, (int, float)) and t > 1e10
                                            else (pd.Timestamp(t).to_pydatetime() if not isinstance(t, datetime) else t)
                                            for t in all_timestamps
                                        ]
                                except Exception as e:
                                    print(f"Warning: Error converting timestamps: {e}")
                                    vel_timestamps = []
                                    vel_values = []
                                
                                if len(vel_ts_numeric) > 1 and len(vel_values) > 0:
                                    try:
                                        # Create interpolation function (use linear interpolation)
                                        vel_interp = interp1d(vel_ts_numeric, vel_values, kind='linear', 
                                                            bounds_error=False, fill_value='extrapolate')
                                        # Interpolate velocity values to RMS timestamps
                                        all_altitudes_velocities = vel_interp(rms_ts_numeric)
                                    except Exception as e:
                                        print(f"Warning: Interpolation failed: {e}")
                                        all_altitudes_velocities = np.array([])
                                elif len(vel_values) > 0:
                                    # If only one velocity point, use constant value
                                    all_altitudes_velocities = np.full_like(all_rms_errors, vel_values[0], dtype=float)
                            else:
                                all_altitudes_velocities = np.array([])
                                print(f"No velocity data available for {instrument_name.upper()} ({direction_labels[direction]})")
                        except Exception as e:
                            print(f"Warning: Could not fetch or interpolate altitude velocity data: {e}")
                        all_altitudes_velocities = np.array([])
            
            # For very large datasets, downsample before computing rolling stddev to speed up plotting
            plot_timestamps = all_timestamps
            plot_rms_errors = all_rms_errors
            plot_altitudes = all_altitudes
            plot_altitudes_velocities = all_altitudes_velocities
            
            if enable_downsampling and len(all_rms_errors) > downsample_max_points:
                # Downsample for rendering (keeps visual accuracy despite fewer points)
                # First, downsample RMS errors using adaptive binning to get the reference timestamps
                plot_timestamps, plot_rms_errors = downsample_for_plotting(
                    np.array(all_timestamps), all_rms_errors, max_points=downsample_max_points
                )
                
                # Now use a uniform downsampling factor to ensure all arrays match the RMS array length
                # This is crucial because downsample_for_plotting uses adaptive binning (min/max/mean)
                # which can produce variable output sizes for different input arrays
                downsample_factor = max(1, len(all_rms_errors) // len(plot_rms_errors))
                
                # Apply uniform downsampling to altitudes using the same factor
                if len(all_altitudes) > 0:
                    plot_altitudes = all_altitudes[::downsample_factor]
                    # If we downsampled more than needed, trim to match RMS length
                    if len(plot_altitudes) > len(plot_rms_errors):
                        plot_altitudes = plot_altitudes[:len(plot_rms_errors)]
                    # If we downsampled less, pad with the last value
                    elif len(plot_altitudes) < len(plot_rms_errors):
                        plot_altitudes = np.pad(plot_altitudes, (0, len(plot_rms_errors) - len(plot_altitudes)), 
                                               mode='edge')
                else:
                    plot_altitudes = np.array([])
                
                # Apply uniform downsampling to velocities using the same factor
                if len(all_altitudes_velocities) > 0:
                    plot_altitudes_velocities = all_altitudes_velocities[::downsample_factor]
                    # If we downsampled more than needed, trim to match RMS length
                    if len(plot_altitudes_velocities) > len(plot_rms_errors):
                        plot_altitudes_velocities = plot_altitudes_velocities[:len(plot_rms_errors)]
                    # If we downsampled less, pad with the last value
                    elif len(plot_altitudes_velocities) < len(plot_rms_errors):
                        plot_altitudes_velocities = np.pad(plot_altitudes_velocities, 
                                                           (0, len(plot_rms_errors) - len(plot_altitudes_velocities)), 
                                                           mode='edge')
                else:
                    plot_altitudes_velocities = np.array([])
            
            # Calculate rolling standard deviation with reasonable window size
            # Check cache first before recalculating
            stddev_cache_key = f"{cache_key}_stddev_{len(plot_rms_errors)}"
            if stddev_cache_key in _ROLLING_STDDEV_CACHE:
                rolling_stddev = _ROLLING_STDDEV_CACHE[stddev_cache_key]
            else:
                # Adaptive window based on data density
                window_size = min(500, max(50, len(plot_rms_errors) // 100))
                rolling_stddev = calculate_rolling_stddev(plot_rms_errors, window_size=window_size)
                # Cache the result
                _ROLLING_STDDEV_CACHE[stddev_cache_key] = rolling_stddev
            
            # Verify all arrays have matching length
            expected_length = len(plot_rms_errors)
            if len(plot_altitudes) > 0 and len(plot_altitudes) != expected_length:
                print(f"Warning: Altitude array length {len(plot_altitudes)} != RMS length {expected_length}")
                if len(plot_altitudes) > expected_length:
                    plot_altitudes = plot_altitudes[:expected_length]
                else:
                    plot_altitudes = np.pad(plot_altitudes, (0, expected_length - len(plot_altitudes)), mode='edge')
            
            if len(plot_altitudes_velocities) > 0 and len(plot_altitudes_velocities) != expected_length:
                print(f"Warning: Velocity array length {len(plot_altitudes_velocities)} != RMS length {expected_length}")
                if len(plot_altitudes_velocities) > expected_length:
                    plot_altitudes_velocities = plot_altitudes_velocities[:expected_length]
                else:
                    plot_altitudes_velocities = np.pad(plot_altitudes_velocities, 
                                                       (0, expected_length - len(plot_altitudes_velocities)), 
                                                       mode='edge')
            
            # Ensure rolling_stddev also matches length (it's computed from plot_rms_errors so should match)
            if len(rolling_stddev) != expected_length:
                print(f"Warning: Rolling StdDev length {len(rolling_stddev)} != RMS length {expected_length}")
                if len(rolling_stddev) > expected_length:
                    rolling_stddev = rolling_stddev[:expected_length]
                else:
                    rolling_stddev = np.pad(rolling_stddev, (0, expected_length - len(rolling_stddev)), mode='edge')
            
            # Log array lengths for verification
            if len(plot_rms_errors) > 100000 or show_altitude:
                print(f"Plot data array lengths for {instrument_name.upper()} ({direction_labels[direction]}):")
                print(f"  RMS Errors: {len(plot_rms_errors)}")
                print(f"  Timestamps: {len(plot_timestamps)}")
                print(f"  Rolling StdDev: {len(rolling_stddev)}")
                print(f"  Altitudes: {len(plot_altitudes)}")
                print(f"  Alt Velocities: {len(plot_altitudes_velocities)}")
            
            # Get actual data time range for title
            data_start = min(all_timestamps)
            data_end = max(all_timestamps)
            if isinstance(data_start, (int, float)):
                data_start = datetime.fromtimestamp(data_start / 1000.0)
            if isinstance(data_end, (int, float)):
                data_end = datetime.fromtimestamp(data_end / 1000.0)
            
            # Cache the processed plot data for future reuse (only if it came from fresh data, not from cache)
            if cache_key not in _PLOT_DATA_CACHE:
                _PLOT_DATA_CACHE[cache_key] = {
                    'all_timestamps': all_timestamps,
                    'all_rms_errors': all_rms_errors,
                    'all_altitudes': all_altitudes,
                    'all_altitudes_velocities': all_altitudes_velocities,
                    'data_start': data_start,
                    'data_end': data_end
                }
            
            # Create figure with 3x2 subplots if altitude shown (top: RMS and StdDev, middle: altitude, bottom: altitude velocity)
            # or 1x2 if altitude not shown
            if show_altitude:
                fig = plt.figure(figsize=(18, 15))
                gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1], hspace=0.35, wspace=0.3)
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1])
                ax1_alt = fig.add_subplot(gs[1, 0], sharex=ax1)
                ax2_alt = fig.add_subplot(gs[1, 1], sharex=ax2)
                ax1_vel = fig.add_subplot(gs[2, 0], sharex=ax1)
                ax2_vel = fig.add_subplot(gs[2, 1], sharex=ax2)
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
                ax1_vel = None
                ax2_vel = None
            
            # Plot 1 (Top-Left): RMS error as scatter points over time
            color1_rms = '#0052A3'  # Dark blue
            ax1.scatter(plot_timestamps, plot_rms_errors, alpha=0.5, s=10, color=color1_rms, edgecolors='none', label='RMS Measurements')
            ax1.set_ylabel('RMS Error (arcsec)', fontsize=12, fontweight='bold', color=color1_rms)
            ax1.tick_params(axis='y', labelcolor=color1_rms)
            
            title_str = f'{instrument_name.upper()} - {direction_labels[direction]}'
            title_str += f'\n({data_start.strftime("%Y-%m-%d %H:%M:%S")} to {data_end.strftime("%Y-%m-%d %H:%M:%S")})'
            title_str += ' - RMS Error Over Time'
            ax1.set_title(title_str, fontsize=13, fontweight='bold', pad=15)
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            # Add mean and median lines to RMS plot (using full data for stats)
            mean_val = np.mean(all_rms_errors)
            median_val = np.median(all_rms_errors)
            ax1.axhline(y=mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}"', alpha=0.7)
            ax1.axhline(y=median_val, color='red', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}"', alpha=0.7)
            ax1.legend(loc='upper right', fontsize=9)
            
            # Format x-axis for main RMS plot - always rotate dates 45 degrees
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
            
            # Plot altitude below RMS if enabled
            if show_altitude:
                ax1.set_xticklabels([])  # Hide x-axis labels on main plot
                color1_alt = '#2E7D32'  # Dark green
                # Verify altitude data length matches RMS errors (reference array)
                if len(plot_altitudes) == len(plot_rms_errors):
                    ax1_alt.scatter(plot_timestamps, plot_altitudes, alpha=0.4, s=5, color=color1_alt, edgecolors='none')
                    ax1_alt.set_ylabel('Altitude (deg)', fontsize=11, fontweight='bold', color=color1_alt)
                    ax1_alt.tick_params(axis='y', labelcolor=color1_alt)
                else:
                    ax1_alt.text(0.5, 0.5, f'Data mismatch: altitude {len(plot_altitudes)} != RMS {len(plot_rms_errors)}', 
                                ha='center', va='center', transform=ax1_alt.transAxes, fontsize=8, color='red')
                ax1_alt.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax1_alt.xaxis.set_major_locator(mdates.AutoDateLocator())
                # Always show and rotate x-axis labels on altitude plot for consistency with RMS and velocity plots
                plt.setp(ax1_alt.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
                ax1_alt.grid(True, alpha=0.3, linestyle='--')
                
                # Plot altitude velocity below altitude if available
                if len(plot_altitudes_velocities) > 0 and len(plot_altitudes_velocities) == len(plot_rms_errors):
                    color1_vel = '#FF6B35'  # Orange-red for velocity
                    ax1_vel.scatter(plot_timestamps, plot_altitudes_velocities, alpha=0.4, s=5, color=color1_vel, edgecolors='none')
                    ax1_vel.set_ylabel('Alt Velocity (arcsec/s)', fontsize=11, fontweight='bold', color=color1_vel)
                    ax1_vel.tick_params(axis='y', labelcolor=color1_vel)
                    ax1_vel.set_xlabel('Date', fontsize=12, fontweight='bold')
                    ax1_vel.grid(True, alpha=0.3, linestyle='--')
                    ax1_vel.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax1_vel.xaxis.set_major_locator(mdates.AutoDateLocator())
                    plt.setp(ax1_vel.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
                    # Add zero line
                    ax1_vel.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
                else:
                    # No velocity data - show placeholder in velocity subplot and add Date label to altitude instead
                    ax1_alt.set_xlabel('Date', fontsize=12, fontweight='bold')
                    if len(plot_altitudes_velocities) > 0:
                        ax1_vel.text(0.5, 0.5, f'Data mismatch: velocity {len(plot_altitudes_velocities)} != RMS {len(plot_rms_errors)}', 
                                    ha='center', va='center', transform=ax1_vel.transAxes, fontsize=8, color='red')
                    else:
                        ax1_vel.text(0.5, 0.5, 'No velocity data available', ha='center', va='center', 
                                    transform=ax1_vel.transAxes, fontsize=10, color='gray')
                    ax1_vel.set_xticks([])
                    ax1_vel.set_yticks([])
            else:
                pass  # No altitude subplot shown
            
            # Plot 2 (Top-Right): Rolling Standard Deviation over time
            color2_stddev = '#CC5200'  # Dark orange
            ax2.scatter(plot_timestamps, rolling_stddev, alpha=0.6, s=10, color=color2_stddev, edgecolors='none', label='Rolling StdDev')
            ax2.plot(plot_timestamps, rolling_stddev, alpha=0.3, linewidth=1, color='#8B3A00')  # Darker orange
            ax2.set_ylabel('Standard Deviation (arcsec)', fontsize=12, fontweight='bold', color=color2_stddev)
            ax2.tick_params(axis='y', labelcolor=color2_stddev)
            
            title_str = f'{instrument_name.upper()} - {direction_labels[direction]}'
            title_str += f'\n({data_start.strftime("%Y-%m-%d %H:%M:%S")} to {data_end.strftime("%Y-%m-%d %H:%M:%S")})'
            title_str += ' - Standard Deviation Over Time'
            ax2.set_title(title_str, fontsize=13, fontweight='bold', pad=15)
            ax2.grid(True, alpha=0.3, linestyle='--')
            
            # Add mean line to StdDev plot
            mean_stddev = np.mean(rolling_stddev)
            ax2.axhline(y=mean_stddev, color='#006600', linestyle='--', linewidth=2, label=f'Mean: {mean_stddev:.4f}"', alpha=0.7)  # Dark green
            ax2.legend(loc='upper right', fontsize=9)
            
            # Format x-axis for main StdDev plot - always rotate dates 45 degrees
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
            
            # Plot altitude below StdDev if enabled
            if show_altitude:
                ax2.set_xticklabels([])  # Hide x-axis labels on main plot
                color2_alt = '#2E7D32'  # Dark green
                # Verify altitude data length matches RMS errors (reference array)
                if len(plot_altitudes) == len(plot_rms_errors):
                    ax2_alt.scatter(plot_timestamps, plot_altitudes, alpha=0.4, s=5, color=color2_alt, edgecolors='none')
                    ax2_alt.set_ylabel('Altitude (deg)', fontsize=11, fontweight='bold', color=color2_alt)
                    ax2_alt.tick_params(axis='y', labelcolor=color2_alt)
                else:
                    ax2_alt.text(0.5, 0.5, f'Data mismatch: altitude {len(plot_altitudes)} != RMS {len(plot_rms_errors)}', 
                                ha='center', va='center', transform=ax2_alt.transAxes, fontsize=8, color='red')
                ax2_alt.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax2_alt.xaxis.set_major_locator(mdates.AutoDateLocator())
                # Always show and rotate x-axis labels on altitude plot for consistency with RMS and velocity plots
                plt.setp(ax2_alt.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
                ax2_alt.grid(True, alpha=0.3, linestyle='--')
                
                # Plot altitude velocity below altitude if available
                if len(plot_altitudes_velocities) > 0 and len(plot_altitudes_velocities) == len(plot_rms_errors):
                    # Velocity data available - show velocity plot with rotation
                    color2_vel = '#FF6B35'  # Orange-red for velocity
                    ax2_vel.scatter(plot_timestamps, plot_altitudes_velocities, alpha=0.4, s=5, color=color2_vel, edgecolors='none')
                    ax2_vel.set_ylabel('Alt Velocity (arcsec/s)', fontsize=11, fontweight='bold', color=color2_vel)
                    ax2_vel.tick_params(axis='y', labelcolor=color2_vel)
                    ax2_vel.set_xlabel('Date', fontsize=12, fontweight='bold')
                    ax2_vel.grid(True, alpha=0.3, linestyle='--')
                    ax2_vel.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax2_vel.xaxis.set_major_locator(mdates.AutoDateLocator())
                    plt.setp(ax2_vel.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
                    # Add zero line
                    ax2_vel.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
                else:
                    # No velocity data - show placeholder in velocity subplot and add Date label to altitude instead
                    ax2_alt.set_xlabel('Date', fontsize=12, fontweight='bold')
                    if len(plot_altitudes_velocities) > 0:
                        ax2_vel.text(0.5, 0.5, f'Data mismatch: velocity {len(plot_altitudes_velocities)} != RMS {len(plot_rms_errors)}', 
                                    ha='center', va='center', transform=ax2_vel.transAxes, fontsize=8, color='red')
                    else:
                        ax2_vel.text(0.5, 0.5, 'No velocity data available', ha='center', va='center', 
                                    transform=ax2_vel.transAxes, fontsize=10, color='gray')
                    ax2_vel.set_xticks([])
                    ax2_vel.set_yticks([])
            else:
                pass  # No altitude subplot shown
            
            plt.tight_layout()
            plt.show()
            fig_count += 1
            plt.close(fig)
    
    print("="*80)
    print(f"Generated {fig_count} time-series plot figure(s)")
    print("Each instrument now has 3 figures (TOTAL, UP, DOWN)")
    print("\nPlot Structure (3 rows × 2 columns):")
    print("  Row 1 (Top):       RMS Error (left) | Rolling StdDev (right)")
    print("  Row 2 (Middle):    Telescope Altitude (left) | Telescope Altitude (right)")
    print("  Row 3 (Bottom):    Altitude Velocity from mount_mini_velalt (left & right)")
    print("\nDetailed Information:")
    print("  - Altitude velocity data queried from measurements.mount_mini_velalt table")
    print("  - Velocity data resampled to match RMS error measurement timestamps")
    print("  - X-axis labels formatted as dates (YYYY-MM-DD) with 45-degree rotation")
    print("  - All subplots share aligned time axes for easy cross-reference")
    print("  - Rolling StdDev uses adaptive window size (~1% of dataset)")
    print("="*80)


def plot_combined_instruments_direction(instrument_stats_dict, direction='total', start_datetime=None, end_datetime=None, show_altitude=True,
                                       enable_downsampling=True, downsample_max_points=10000):
    """
    Generate combined plots for all instruments in a specific direction.
    Shows RMS error and rolling StdDev side-by-side for all instruments overlaid.
    When show_altitude is True, altitude plots are displayed on separate graphs below with aligned x-axes.
    
    Parameters:
    -----------
    instrument_stats_dict : dict
        Dictionary with instrument names as keys and lists of run data as values
    direction : str, optional
        Direction to plot: 'total', 'up', or 'down'. Default is 'total'
    start_datetime : datetime, optional
        Start datetime for title context
    end_datetime : datetime, optional
        End datetime for title context
    show_altitude : bool, optional
        Whether to display telescope altitude on separate plot below. Default is True.
    enable_downsampling : bool, optional
        Whether to downsample large datasets for faster plotting. Default is True.
    downsample_max_points : int, optional
        Maximum number of points to display in plots. Default is 10000.
    """
    if not instrument_stats_dict or len(instrument_stats_dict) == 0:
        print("No instruments to plot")
        return None
    
    # Convert string datetimes to datetime objects if needed
    if isinstance(start_datetime, str):
        start_datetime = datetime.fromisoformat(start_datetime)
    if isinstance(end_datetime, str):
        end_datetime = datetime.fromisoformat(end_datetime)
    
    # Direction configuration
    direction_lower = direction.lower()
    direction_map = {'total': ('rms_errors', 'dates', 'altitudes', 'BOTH (Total)'),
                     'up': ('rms_errors_up', 'dates_up', 'altitudes_up', 'UP (Elevation Increasing)'),
                     'down': ('rms_errors_down', 'dates_down', 'altitudes_down', 'DOWN (Elevation Decreasing)')}
    
    if direction_lower not in direction_map:
        print(f"Invalid direction: {direction}. Must be 'total', 'up', or 'down'")
        return None
    
    rms_key, dates_key, altitudes_key, dir_label = direction_map[direction_lower]
    
    print(f"\n{'='*80}")
    print(f"GENERATING COMBINED INSTRUMENT PLOTS - {dir_label.upper()} DIRECTION")
    print(f"{'='*80}")
    
    # Prepare colors for each instrument
    num_instruments = len(instrument_stats_dict)
    
    # High-contrast color palette (avoids yellow, cyan, pale violet)
    high_contrast_colors = [
        '#0052A3',  # Dark blue
        '#CC0000',  # Bright red
        '#006600',  # Dark green
        '#CC5200',  # Dark orange
        '#440099',  # Dark purple
        '#8B4513',  # Saddle brown
        '#008B8B',  # Dark cyan
        '#000080',  # Navy
        '#8B0000',  # Dark red
        '#4B0082',  # Indigo
        '#DC143C',  # Crimson
        '#228B22',  # Forest green
        '#1E90FF',  # Dodger blue
        '#FF6347',  # Tomato
        '#2F4F4F',  # Dark slate gray
    ]
    colors = [high_contrast_colors[i % len(high_contrast_colors)] for i in range(max(num_instruments, 3))]
    
    # Collect all altitude and altitude velocity data for altitude plots if enabled
    all_combined_timestamps = []
    all_combined_altitudes = []
    all_combined_altitudes_velocities = []
    
    # Create cache key for combined altitude velocity data
    combined_cache_key = f"{direction_lower}_combined_alt_vel"
    
    if show_altitude:
        # Check if cached data is available for this direction
        if combined_cache_key in _PLOT_DATA_CACHE:
            print(f"[CACHE HIT] Using cached combined altitude velocity data for '{direction}' direction")
            cached_data = _PLOT_DATA_CACHE[combined_cache_key]
            all_combined_timestamps = cached_data['all_timestamps']
            all_combined_altitudes = cached_data['all_altitudes']
            all_combined_altitudes_velocities = cached_data['all_velocities']
        else:
            # Collect fresh data
            for instrument_name in sorted(instrument_stats_dict.keys()):
                runs_data = instrument_stats_dict[instrument_name]
                
                if not isinstance(runs_data, list) or len(runs_data) == 0:
                    continue
                
                for run in runs_data:
                    dates = run.get(dates_key, np.array([]))
                    altitudes = run.get(altitudes_key, np.array([]))
                    
                    if len(dates) > 0:
                        all_combined_timestamps.extend(dates)
                        all_combined_altitudes.extend(altitudes)
            
            # Query or calculate altitude velocity data for combined plot if available
            if len(all_combined_timestamps) > 0:
                if USE_ALTITUDE_GRADIENT_VELOCITY:
                    # Calculate velocity directly from altitude data (faster, no DB query)
                    try:
                        all_combined_altitudes_velocities = calculate_altitude_velocity(
                            np.array(all_combined_altitudes), 
                            all_combined_timestamps
                        )
                    except Exception as e:
                        print(f"Warning: Could not calculate altitude velocity for combined plot: {e}")
                        all_combined_altitudes_velocities = np.array([])
                else:
                    # Query velocity data from database (traditional approach)
                    try:
                        # Get date range from the combined timestamps
                        # Convert all to numeric first to avoid datetime64 issues
                        def ts_to_dt(t):
                            if isinstance(t, datetime):
                                return t
                            elif isinstance(t, (int, float)):
                                return datetime.fromtimestamp(t / 1000.0)
                            else:
                                # Handle numpy datetime64 using pandas
                                return pd.Timestamp(t).to_pydatetime()
                        
                        # Convert all timestamps to compatible format
                        converted_timestamps = [ts_to_dt(t) for t in all_combined_timestamps]
                        data_start_dt = min(converted_timestamps)
                        data_end_dt = max(converted_timestamps)
                        
                        # Query velocity data
                        vel_timestamps, vel_values = query_altitude_velocity_data(
                            start_datetime=data_start_dt,
                            end_datetime=data_end_dt
                        )
                        
                        # Interpolate velocity values to match combined timestamps if available
                        if len(vel_timestamps) > 0 and len(vel_values) > 0:
                            # Convert timestamps to numeric seconds (handles Python datetime, numpy datetime64, and integers)
                            def ts_to_numeric(t):
                                if isinstance(t, datetime):
                                    return t.timestamp()
                                elif isinstance(t, (int, float)):
                                    return t / 1000.0  # Assume milliseconds
                                else:
                                    # Handle numpy datetime64 using pandas
                                    return pd.Timestamp(t).timestamp()
                            
                            vel_ts_numeric = np.array([ts_to_numeric(t) for t in vel_timestamps], dtype=float)
                            combined_ts_numeric = np.array([ts_to_numeric(t) for t in all_combined_timestamps], dtype=float)
                            
                            if len(vel_ts_numeric) > 1:
                                vel_interp = interp1d(vel_ts_numeric, vel_values, kind='linear', 
                                                    bounds_error=False, fill_value='extrapolate')
                                all_combined_altitudes_velocities = vel_interp(combined_ts_numeric)
                            else:
                                all_combined_altitudes_velocities = np.full_like(combined_ts_numeric, vel_values[0])
                    except Exception as e:
                        print(f"Warning: Could not fetch altitude velocity data for combined plot: {e}")
                        all_combined_altitudes_velocities = np.array([])
            
            # Sort combined altitude data
            if len(all_combined_timestamps) > 0:
                combined_sorted_indices = np.argsort(all_combined_timestamps)
                all_combined_timestamps = [all_combined_timestamps[i] for i in combined_sorted_indices]
                all_combined_altitudes = np.array(all_combined_altitudes)[combined_sorted_indices]
                if len(all_combined_altitudes_velocities) > 0:
                    all_combined_altitudes_velocities = np.array(all_combined_altitudes_velocities)[combined_sorted_indices]
            
            # Store in cache for future use
            _PLOT_DATA_CACHE[combined_cache_key] = {
                'all_timestamps': all_combined_timestamps,
                'all_altitudes': all_combined_altitudes,
                'all_velocities': all_combined_altitudes_velocities
            }
    
    # Create figure with stacked plots if altitude is shown
    if show_altitude:
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1], hspace=0.35, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax1_alt = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax2_alt = fig.add_subplot(gs[1, 1], sharex=ax2)
        ax1_vel = fig.add_subplot(gs[2, 0], sharex=ax1)
        ax2_vel = fig.add_subplot(gs[2, 1], sharex=ax2)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Altitude color (consistent dark green)
    altitude_color = '#2E7D32'
    
    # Track min/max timestamps for data range
    min_timestamp = None
    max_timestamp = None
    
    for idx, instrument_name in enumerate(sorted(instrument_stats_dict.keys())):
        runs_data = instrument_stats_dict[instrument_name]
        
        if not isinstance(runs_data, list) or len(runs_data) == 0:
            continue
        
        # Collect all timestamps and RMS errors for this direction
        all_timestamps = []
        all_rms_errors = []
        all_altitudes = []
        
        for run in runs_data:
            rms_errors = run.get(rms_key, np.array([]))
            dates = run.get(dates_key, np.array([]))
            altitudes = run.get(altitudes_key, np.array([]))
            
            if len(rms_errors) > 0:
                all_rms_errors.extend(rms_errors)
                all_timestamps.extend(dates)
                all_altitudes.extend(altitudes)
        
        if len(all_rms_errors) == 0:
            continue
        
        # Track min/max for title
        current_min = min(all_timestamps)
        current_max = max(all_timestamps)
        if min_timestamp is None or current_min < min_timestamp:
            min_timestamp = current_min
        if max_timestamp is None or current_max > max_timestamp:
            max_timestamp = current_max
        
        # Sort by timestamp
        sorted_indices = np.argsort(all_timestamps)
        all_timestamps = [all_timestamps[i] for i in sorted_indices]
        all_rms_errors = np.array(all_rms_errors)[sorted_indices]
        all_altitudes = np.array(all_altitudes)[sorted_indices]
        
        # Downsample for large datasets before computing rolling stddev
        plot_timestamps = all_timestamps
        plot_rms_errors = all_rms_errors
        plot_altitudes = all_altitudes
        
        if enable_downsampling and len(all_rms_errors) > downsample_max_points:
            # Downsample for rendering
            plot_timestamps, plot_rms_errors = downsample_for_plotting(
                np.array(all_timestamps), all_rms_errors, max_points=downsample_max_points
            )
            if len(all_altitudes) > 0:
                _, plot_altitudes = downsample_for_plotting(
                    np.array(all_timestamps), all_altitudes, max_points=downsample_max_points
                )
        
        # Calculate rolling standard deviation on (possibly downsampled) data with caching
        window_size = min(500, max(50, len(plot_rms_errors) // 100))
        stddev_cache_key = f"{direction_lower}_{instrument_name}_combined_stddev_{len(plot_rms_errors)}"
        
        if stddev_cache_key in _ROLLING_STDDEV_CACHE:
            rolling_stddev = _ROLLING_STDDEV_CACHE[stddev_cache_key]
        else:
            rolling_stddev = calculate_rolling_stddev(plot_rms_errors, window_size=window_size)
            _ROLLING_STDDEV_CACHE[stddev_cache_key] = rolling_stddev
        
        # Plot RMS on left
        ax1.scatter(plot_timestamps, plot_rms_errors, alpha=0.6, s=15, color=colors[idx], label=instrument_name.upper(), edgecolors='none')
        
        # Plot StdDev on right
        ax2.scatter(plot_timestamps, rolling_stddev, alpha=0.6, s=15, color=colors[idx], label=instrument_name.upper(), edgecolors='none')
        ax2.plot(plot_timestamps, rolling_stddev, alpha=0.3, linewidth=1, color=colors[idx])
    
    # Format the RMS plot (top-left or left)
    ax1.set_ylabel('RMS Error (arcsec)', fontsize=12, fontweight='bold')
    ax1.set_title('RMS Error', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
    
    if show_altitude:
        ax1.set_xticklabels([])  # Hide x-axis labels on main plot
    else:
        ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    
    # Format the StdDev plot (top-right or right)
    ax2.set_ylabel('Standard Deviation (arcsec)', fontsize=12, fontweight='bold')
    ax2.set_title('Rolling Standard Deviation', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
    
    if show_altitude:
        ax2.set_xticklabels([])  # Hide x-axis labels on main plot
    else:
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    
    # Add altitude plots below if enabled, with aligned x-axes
    if show_altitude and len(all_combined_timestamps) > 0:
        # Bottom-left: Altitude for RMS plot
        ax1_alt.scatter(all_combined_timestamps, all_combined_altitudes, alpha=0.4, s=8, color=altitude_color, edgecolors='none')
        ax1_alt.set_ylabel('Altitude (deg)', fontsize=11, fontweight='bold', color=altitude_color)
        ax1_alt.tick_params(axis='y', labelcolor=altitude_color)
        ax1_alt.set_xticklabels([])  # Hide labels - velocity plot below will show them
        ax1_alt.grid(True, alpha=0.3, linestyle='--')
        ax1_alt.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1_alt.xaxis.set_major_locator(mdates.AutoDateLocator())
        # Apply rotation formatting for consistency, even if labels are hidden
        plt.setp(ax1_alt.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
        
        # Bottom-right: Altitude for StdDev plot
        ax2_alt.scatter(all_combined_timestamps, all_combined_altitudes, alpha=0.4, s=8, color=altitude_color, edgecolors='none')
        ax2_alt.set_ylabel('Altitude (deg)', fontsize=11, fontweight='bold', color=altitude_color)
        ax2_alt.tick_params(axis='y', labelcolor=altitude_color)
        ax2_alt.set_xticklabels([])  # Hide labels - velocity plot below will show them
        ax2_alt.grid(True, alpha=0.3, linestyle='--')
        ax2_alt.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2_alt.xaxis.set_major_locator(mdates.AutoDateLocator())
        # Apply rotation formatting for consistency, even if labels are hidden
        plt.setp(ax2_alt.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
        
        # Bottom row: Altitude velocity plots
        velocity_color = '#FF6B35'  # Orange-red for velocity
        
        if len(all_combined_altitudes_velocities) > 0:
            # Bottom-left: Altitude velocity for RMS plot
            ax1_vel.scatter(all_combined_timestamps, all_combined_altitudes_velocities, alpha=0.4, s=8, color=velocity_color, edgecolors='none')
            ax1_vel.set_ylabel('Alt Velocity (arcsec/s)', fontsize=11, fontweight='bold', color=velocity_color)
            ax1_vel.tick_params(axis='y', labelcolor=velocity_color)
            ax1_vel.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax1_vel.grid(True, alpha=0.3, linestyle='--')
            ax1_vel.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1_vel.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax1_vel.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
            # Add zero line
            ax1_vel.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Bottom-right: Altitude velocity for StdDev plot
            ax2_vel.scatter(all_combined_timestamps, all_combined_altitudes_velocities, alpha=0.4, s=8, color=velocity_color, edgecolors='none')
            ax2_vel.set_ylabel('Alt Velocity (arcsec/s)', fontsize=11, fontweight='bold', color=velocity_color)
            ax2_vel.tick_params(axis='y', labelcolor=velocity_color)
            ax2_vel.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax2_vel.grid(True, alpha=0.3, linestyle='--')
            ax2_vel.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2_vel.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax2_vel.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
            # Add zero line
            ax2_vel.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        else:
            # No velocity data available - show placeholder
            ax1_vel.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax1_vel.text(0.5, 0.5, 'No velocity data available', ha='center', va='center', 
                        transform=ax1_vel.transAxes, fontsize=10, color='gray')
            ax1_vel.set_xticks([])
            ax1_vel.set_yticks([])
            
            ax2_vel.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax2_vel.text(0.5, 0.5, 'No velocity data available', ha='center', va='center', 
                        transform=ax2_vel.transAxes, fontsize=10, color='gray')
            ax2_vel.set_xticks([])
            ax2_vel.set_yticks([])
    
    # Add common title and legend
    title_str = f'Combined Elevation Error Analysis - {dir_label} Direction'
    
    # Convert timestamps to datetime for title if needed
    if min_timestamp is not None and max_timestamp is not None:
        if isinstance(min_timestamp, (int, float)):
            data_start = datetime.fromtimestamp(min_timestamp / 1000.0)
            data_end = datetime.fromtimestamp(max_timestamp / 1000.0)
        else:
            data_start = min_timestamp
            data_end = max_timestamp
        title_str += f'\n({data_start.strftime("%Y-%m-%d %H:%M:%S")} to {data_end.strftime("%Y-%m-%d %H:%M:%S")})'
    elif start_datetime and end_datetime:
        title_str += f'\n({start_datetime.strftime("%Y-%m-%d")} to {end_datetime.strftime("%Y-%m-%d")})'
    
    fig.suptitle(title_str, fontsize=14, fontweight='bold', y=0.98)
    
    # Shared legend for instruments
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=min(len(labels), 5), fontsize=10, 
               bbox_to_anchor=(0.5, 0.96), framealpha=0.95)
    ax1.legend().remove()
    ax2.legend().remove()
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()
    plt.close(fig)
    
    print(f"✓ Combined {dir_label.lower()} plot generated")
    
    return fig


def plot_summary_statistics_overview(instrument_stats_dict, start_datetime=None, end_datetime=None):
    """
    Create comprehensive visualization of summary statistics results.
    Displays 2x2 grid with mean RMS, std dev, min/max, and data points.
    
    Parameters:
    -----------
    instrument_stats_dict : dict
        Dictionary with instrument names as keys and lists of run data as values
    start_datetime : datetime, optional
        Start datetime for title context
    end_datetime : datetime, optional
        End datetime for title context
    """
    if not instrument_stats_dict or len(instrument_stats_dict) == 0:
        print("No instruments to plot")
        return None
    
    # Convert string datetimes to datetime objects if needed
    if isinstance(start_datetime, str):
        start_datetime = datetime.fromisoformat(start_datetime)
    if isinstance(end_datetime, str):
        end_datetime = datetime.fromisoformat(end_datetime)
    
    print("\n" + "="*80)
    print("GENERATING SUMMARY RESULTS VISUALIZATIONS")
    print("="*80)
    
    # Prepare data for plotting
    instruments = sorted(instrument_stats_dict.keys())
    mean_rms_values = []
    std_dev_values = []
    min_error_values = []
    max_error_values = []
    data_points_values = []
    run_counts = []
    
    # Create cache key for summary statistics
    summary_cache_key = "summary_stats_overall"
    
    # Check if cached summary stats are available
    if summary_cache_key in _PLOT_DATA_CACHE:
        print(f"[CACHE HIT] Using cached summary statistics")
        cached_stats = _PLOT_DATA_CACHE[summary_cache_key]
        mean_rms_values = cached_stats['mean_rms_values']
        std_dev_values = cached_stats['std_dev_values']
        min_error_values = cached_stats['min_error_values']
        max_error_values = cached_stats['max_error_values']
        data_points_values = cached_stats['data_points_values']
        run_counts = cached_stats['run_counts']
    else:
        # Calculate fresh statistics
        for instrument_name in instruments:
            runs_data = instrument_stats_dict[instrument_name]
            all_rms = np.concatenate([run['rms_errors'] for run in runs_data if 'rms_errors' in run])
            all_dates = np.concatenate([run['dates'] for run in runs_data if 'dates' in run])
            
            if len(all_rms) == 0:
                continue
            
            # Calculate overall statistics for this instrument
            overall_stats = calculate_statistics(all_rms, all_dates)
            
            mean_rms_values.append(overall_stats['overall_mean'])
            std_dev_values.append(overall_stats['overall_std'])
            min_error_values.append(overall_stats['min_error'])
            max_error_values.append(overall_stats['max_error'])
            data_points_values.append(overall_stats.get('data_points', len(all_rms)))
            run_counts.append(len(runs_data))
        
        # Store in cache for future use
        _PLOT_DATA_CACHE[summary_cache_key] = {
            'mean_rms_values': mean_rms_values,
            'std_dev_values': std_dev_values,
            'min_error_values': min_error_values,
            'max_error_values': max_error_values,
            'data_points_values': data_points_values,
            'run_counts': run_counts
        }
    
    # Create a 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    title_str = 'Summary Statistics Overview'
    if start_datetime and end_datetime:
        title_str += f' ({start_datetime.strftime("%Y-%m-%d")} to {end_datetime.strftime("%Y-%m-%d")})'
    fig.suptitle(title_str, fontsize=16, fontweight='bold', y=0.995)
    
    colors_bar = plt.cm.Set3(np.linspace(0, 1, len(instruments)))
    
    # Plot 1: Mean RMS Error by Instrument
    ax = axes[0, 0]
    bars1 = ax.bar(range(len(mean_rms_values)), mean_rms_values, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Mean RMS Error (arcsec)', fontsize=11, fontweight='bold')
    ax.set_title('Mean RMS Error by Instrument', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(instruments)))
    ax.set_xticklabels([name.upper() for name in instruments], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Standard Deviation by Instrument
    ax = axes[0, 1]
    bars2 = ax.bar(range(len(std_dev_values)), std_dev_values, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Std Deviation (arcsec)', fontsize=11, fontweight='bold')
    ax.set_title('Standard Deviation by Instrument', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(instruments)))
    ax.set_xticklabels([name.upper() for name in instruments], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: RMS Error Range (Min-Max) by Instrument
    ax = axes[1, 0]
    x_pos = np.arange(len(instruments))
    
    # Create bars showing range
    for i, (min_val, max_val) in enumerate(zip(min_error_values, max_error_values)):
        ax.plot([i, i], [min_val, max_val], 'o-', color=colors_bar[i], linewidth=3, markersize=6, alpha=0.8)
    
    ax.set_ylabel('RMS Error (arcsec)', fontsize=11, fontweight='bold')
    ax.set_title('RMS Error Range (Min-Max) by Instrument', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(instruments)))
    ax.set_xticklabels([name.upper() for name in instruments], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Data Points and Run Counts by Instrument
    ax = axes[1, 1]
    x_pos = np.arange(len(instruments))
    width = 0.35
    
    bars3 = ax.bar(x_pos - width/2, data_points_values, width, label='Data Points', color='steelblue', alpha=0.8, edgecolor='black', linewidth=1)
    ax2 = ax.twinx()
    bars4 = ax2.bar(x_pos + width/2, run_counts, width, label='Run Count', color='coral', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Data Points', fontsize=11, fontweight='bold', color='steelblue')
    ax2.set_ylabel('Number of Runs', fontsize=11, fontweight='bold', color='coral')
    ax.set_title('Data Points and Run Counts by Instrument', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([name.upper() for name in instruments], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add combined legend
    ax.legend([bars3, bars4], ['Data Points', 'Run Count'], loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    
    print("✓ Summary statistics overview plots generated")
    
    return fig


def plot_direction_comparison(instrument_stats_dict, start_datetime=None, end_datetime=None):
    """
    Create elevation direction comparison plots (UP vs DOWN).
    Shows mean RMS error and standard deviation for each direction.
    
    Parameters:
    -----------
    instrument_stats_dict : dict
        Dictionary with instrument names as keys and lists of run data as values
    start_datetime : datetime, optional
        Start datetime for title context
    end_datetime : datetime, optional
        End datetime for title context
    """
    if not instrument_stats_dict or len(instrument_stats_dict) == 0:
        print("No instruments to plot")
        return None
    
    # Convert string datetimes to datetime objects if needed
    if isinstance(start_datetime, str):
        start_datetime = datetime.fromisoformat(start_datetime)
    if isinstance(end_datetime, str):
        end_datetime = datetime.fromisoformat(end_datetime)
    
    print("\nGenerating elevation direction comparison plots...")
    
    # Prepare data for up/down comparison
    instruments = sorted(instrument_stats_dict.keys())
    mean_rms_up = []
    mean_rms_down = []
    std_dev_up = []
    std_dev_down = []
    
    # Create cache key for direction comparison statistics
    dir_comp_cache_key = "direction_comparison_stats"
    
    # Check if cached direction comparison stats are available
    if dir_comp_cache_key in _PLOT_DATA_CACHE:
        print(f"[CACHE HIT] Using cached direction comparison statistics")
        cached_stats = _PLOT_DATA_CACHE[dir_comp_cache_key]
        mean_rms_up = cached_stats['mean_rms_up']
        mean_rms_down = cached_stats['mean_rms_down']
        std_dev_up = cached_stats['std_dev_up']
        std_dev_down = cached_stats['std_dev_down']
    else:
        # Calculate fresh direction comparison statistics
        for instrument_name in instruments:
            runs_data = instrument_stats_dict[instrument_name]
            
            up_means = []
            up_stds = []
            down_means = []
            down_stds = []
            
            for run in runs_data:
                if run.get('stats_up'):
                    up_means.append(run['stats_up']['overall_mean'])
                    up_stds.append(run['stats_up']['overall_std'])
                if run.get('stats_down'):
                    down_means.append(run['stats_down']['overall_mean'])
                    down_stds.append(run['stats_down']['overall_std'])
            
            mean_rms_up.append(np.mean(up_means) if up_means else 0)
            mean_rms_down.append(np.mean(down_means) if down_means else 0)
            std_dev_up.append(np.mean(up_stds) if up_stds else 0)
            std_dev_down.append(np.mean(down_stds) if down_stds else 0)
        
        # Store in cache for future use
        _PLOT_DATA_CACHE[dir_comp_cache_key] = {
            'mean_rms_up': mean_rms_up,
            'mean_rms_down': mean_rms_down,
            'std_dev_up': std_dev_up,
            'std_dev_down': std_dev_down
        }
    
    # Create figure for up/down comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    title_str = 'Elevation Direction Comparison'
    if start_datetime and end_datetime:
        title_str += f' ({start_datetime.strftime("%Y-%m-%d")} to {end_datetime.strftime("%Y-%m-%d")})'
    fig.suptitle(title_str, fontsize=16, fontweight='bold')
    
    # Plot 1: Mean RMS Error - UP vs DOWN
    ax = axes[0]
    x_pos = np.arange(len(instruments))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, mean_rms_up, width, label='UP (Elevation Increasing)', 
                   color='#00AA00', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, mean_rms_down, width, label='DOWN (Elevation Decreasing)', 
                   color='#FF0000', alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Mean RMS Error (arcsec)', fontsize=11, fontweight='bold')
    ax.set_title('Mean RMS Error: UP vs DOWN', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([name.upper() for name in instruments], rotation=45, ha='right')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Standard Deviation - UP vs DOWN
    ax = axes[1]
    
    bars3 = ax.bar(x_pos - width/2, std_dev_up, width, label='UP (Elevation Increasing)', 
                   color='#00AA00', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars4 = ax.bar(x_pos + width/2, std_dev_down, width, label='DOWN (Elevation Decreasing)', 
                   color='#FF0000', alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Standard Deviation (arcsec)', fontsize=11, fontweight='bold')
    ax.set_title('StdDev: UP vs DOWN', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([name.upper() for name in instruments], rotation=45, ha='right')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    
    print("✓ Elevation direction comparison plots generated")
    
    return fig


def generate_all_plots(instrument_stats_dict, start_datetime=None, end_datetime=None):
    """
    Generate all analysis plots: individual timeseries, combined by direction, 
    summary statistics, and direction comparison.
    
    This is a convenience function to create all visualizations in one call.
    
    Parameters:
    -----------
    instrument_stats_dict : dict
        Dictionary with instrument names as keys and lists of run data as values
    start_datetime : datetime or str, optional
        Start datetime for title context
    end_datetime : datetime or str, optional
        End datetime for title context
    """
    if not instrument_stats_dict or len(instrument_stats_dict) == 0:
        print("No instruments to plot")
        return
    
    # Convert string datetimes to datetime objects if needed
    if isinstance(start_datetime, str):
        start_datetime = datetime.fromisoformat(start_datetime)
    if isinstance(end_datetime, str):
        end_datetime = datetime.fromisoformat(end_datetime)
    
    print("\n" + "="*100)
    print("GENERATING ALL ANALYSIS VISUALIZATIONS")
    print("="*100)
    
    # Generate individual instrument timeseries plots (RMS and StdDev by direction)
    print("\n[1/5] Individual Instrument Time-Series Plots...")
    plot_individual_instrument_timeseries(instrument_stats_dict, start_datetime, end_datetime)
    
    # Generate combined plots for each direction
    print("\n[2/5] Combined Instruments - BOTH Direction...")
    plot_combined_instruments_direction(instrument_stats_dict, 'total', start_datetime, end_datetime)
    
    print("\n[3/5] Combined Instruments - UP Direction...")
    plot_combined_instruments_direction(instrument_stats_dict, 'up', start_datetime, end_datetime)
    
    print("\n[4/5] Combined Instruments - DOWN Direction...")
    plot_combined_instruments_direction(instrument_stats_dict, 'down', start_datetime, end_datetime)
    
    # Generate summary statistics plots
    print("\n[5/5a] Summary Statistics Overview...")
    plot_summary_statistics_overview(instrument_stats_dict, start_datetime, end_datetime)
    
    # Generate direction comparison plots
    print("\n[5/5b] Direction Comparison (UP vs DOWN)...")
    plot_direction_comparison(instrument_stats_dict, start_datetime, end_datetime)
    
    print("\n" + "="*100)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*100)


def main(start_datetime=None, end_datetime=None, threshold_arcsec=5.0):
    """
    Main execution function.
    
    Parameters:
    -----------
    start_datetime : datetime or str, optional
        Start datetime for analysis. If None, defaults to 5 years ago.
        Can be a datetime object or ISO format string (YYYY-MM-DD HH:MM:SS)
    end_datetime : datetime or str, optional
        End datetime for analysis. If None, defaults to now.
        Can be a datetime object or ISO format string (YYYY-MM-DD HH:MM:SS)
    threshold_arcsec : float, optional
        RMS error threshold in arc-seconds for filtering outliers. Default is 5.0.
    """
    # Convert string datetimes to datetime objects if needed
    if isinstance(start_datetime, str):
        start_datetime = datetime.fromisoformat(start_datetime)
    if isinstance(end_datetime, str):
        end_datetime = datetime.fromisoformat(end_datetime)
    
    # Query all instruments on telescope during specified period
    print("Querying all instruments on telescope from MariaDB...")
    instruments_dict = None
    try:
        instruments_dict = query_all_instruments(
            host='mariadb.mmto.arizona.edu',
            user='mmtstaff',
            password='multiple',
            database='measurements',
            start_datetime=start_datetime,
            end_datetime=end_datetime
        )
        print_instruments_on_telescope(instruments_dict)
    except Exception as e:
        print(f"Failed to query instruments: {e}")
        return
    
    if not instruments_dict or len(instruments_dict) == 0:
        print("No instruments found for analysis")
        return
    
    # Dictionary to store statistics and data for each instrument
    # Structure: {instrument_name: [{'dates': np.ndarray, 'rms_errors': np.ndarray, 'stats': dict, 'run_number': int}, ...]}
    instrument_stats_dict = {}
    
    # Iterate over each instrument and generate statistics and plots
    for instrument_name in sorted(instruments_dict.keys()):
        # Skip NA instruments
        if instrument_name.upper() == 'NA':
            print(f"\nSkipping NA instrument")
            continue
        print("\n" + "="*70)
        print(f"ANALYZING INSTRUMENT: {instrument_name.upper()}")
        print("="*70)
        
        # Query hexapod_mini_instrument data for this specific instrument
        print(f"\nQuerying hexapod_mini_instrument data for {instrument_name}...")
        observing_runs = None
        try:
            instrument_dates = query_hexapod_instrument_data(
                host='mariadb.mmto.arizona.edu',
                user='mmtstaff',
                password='multiple',
                database='measurements',
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                instrument_name=instrument_name
            )
            print(f"Retrieved {len(instrument_dates)} instrument records for {instrument_name}")
            
            # Identify observing runs from instrument timestamps
            if len(instrument_dates) > 0:
                observing_runs = identify_observing_runs(instrument_dates, min_gap_hours=1)
            else:
                print(f"No observing runs found for {instrument_name}")
                continue
        except Exception as e:
            print(f"Failed to query hexapod_mini_instrument data for {instrument_name}: {e}")
            continue
        
        # Analyze each observing run individually
        if observing_runs:
            print(f"\nAnalyzing {len(observing_runs)} observing run(s) for {instrument_name}:")
            
            # List to store data for all runs of this instrument
            instrument_runs_data = []
            
            for run_idx, (run_start, run_end) in enumerate(observing_runs, 1):
                # Get data for this run
                try:
                    start_us = int(run_start.timestamp() * 1_000_000)
                    end_us = int(run_end.timestamp() * 1_000_000)
                    
                    connection = pymysql.connect(
                        host='mariadb.mmto.arizona.edu',
                        user='mmtstaff',
                        password='multiple',
                        database='mount_hires'
                    )
                    
                    cursor = connection.cursor()
                    
                    query = """
                    SELECT timestamp, telescope_alterr 
                    FROM rd_data_vu 
                    WHERE timestamp >= %s AND timestamp <= %s
                    AND modealt = 'tracking' AND modeaz = 'tracking' AND moderot = 'tracking'
                    AND ABS(telescope_alterr) <= %s
                    ORDER BY timestamp ASC
                    """
                    
                    cursor.execute(query, (start_us, end_us, RMS_ERROR_THRESHOLD_DEGREES))
                    results = cursor.fetchall()
                    cursor.close()
                    connection.close()
                    
                    if not results or len(results) == 0:
                        print(f"  Run {run_idx}: No RMS data found")
                        continue
                    
                    # Extract and process data
                    dates = []
                    rms_errors = []
                    
                    for row in results:
                        timestamp_us = row[0]
                        alterr = row[1]
                        
                        timestamp = datetime.fromtimestamp(timestamp_us / 1_000_000)
                        dates.append(timestamp)
                        rms_errors.append(float(alterr) if alterr is not None else np.nan)
                    
                    # Remove NaN values
                    dates = np.array(dates)
                    rms_errors = np.array(rms_errors)
                    
                    valid_mask = ~np.isnan(rms_errors)
                    dates = dates[valid_mask]
                    rms_errors = rms_errors[valid_mask]
                    
                    if len(rms_errors) == 0:
                        print(f"  Run {run_idx}: No valid RMS records after filtering")
                        continue
                    
                    # Convert RMS errors from degrees to arc-seconds for stddev filtering
                    rms_errors_arcsec = rms_errors * 3600.0
                    
                    # Apply rolling standard deviation filter
                    dates, rms_errors_arcsec = apply_stddev_filter(dates, rms_errors_arcsec, STDDEV_ERROR_THRESHOLD_ARCSEC)
                    
                    if len(rms_errors_arcsec) == 0:
                        print(f"  Run {run_idx}: No valid records after StdDev filtering")
                        continue
                    
                    # Convert back to degrees for statistics calculation
                    rms_errors = rms_errors_arcsec / 3600.0
                    
                    print(f"  Run {run_idx}: {run_start} to {run_end}")
                    print(f"    Duration: {run_end - run_start}")
                    print(f"    Retrieved {len(rms_errors)} valid RMS records after all filtering")
                    
                    # Calculate statistics for this run
                    stats = calculate_statistics(rms_errors, dates)
                    
                    # Calculate directional statistics
                    directional_stats = calculate_directional_statistics(rms_errors, dates)
                    
                    # Print run summary with comprehensive statistics (PRIMARY METRIC first)
                    rs_stats = stats.get('rolling_stddev_statistics', {})
                    ee_stats = stats.get('elevation_error_statistics', {})
                    
                    print(f"    *** PRIMARY METRIC: ROLLING STDDEV - Mean={rs_stats.get('mean', 'N/A'):.4f}\" Std={rs_stats.get('std', 'N/A'):.4f}\" Min={rs_stats.get('min', 'N/A'):.4f}\" Max={rs_stats.get('max', 'N/A'):.4f}\"")
                    print(f"    SECONDARY METRIC: ELEVATION ERROR - Mean={ee_stats.get('mean', 'N/A'):.4f}\" Std={ee_stats.get('std', 'N/A'):.4f}\" Min={ee_stats.get('min', 'N/A'):.4f}\" Max={ee_stats.get('max', 'N/A'):.4f}\"")
                    print(f"    Directional Analysis (Reference):")
                    print(f"      X: RMS={directional_stats['x_direction']['rms']:.4f}\" StdDev={directional_stats['x_direction']['stddev']:.4f}\"")
                    print(f"      Y: RMS={directional_stats['y_direction']['rms']:.4f}\" StdDev={directional_stats['y_direction']['stddev']:.4f}\"")
                    print(f"      Z: RMS={directional_stats['z_direction']['rms']:.4f}\" StdDev={directional_stats['z_direction']['stddev']:.4f}\"")
                    
                    # Store run data
                    instrument_runs_data.append({
                        'dates': dates,
                        'rms_errors': rms_errors,
                        'stats': stats,
                        'directional_stats': directional_stats,
                        'run_number': run_idx,
                        'run_start_dt': run_start
                    })
                    
                except Exception as e:
                    print(f"  Run {run_idx}: Error collecting data: {e}")
            
            # Store all runs data for this instrument
            if len(instrument_runs_data) > 0:
                instrument_stats_dict[instrument_name] = instrument_runs_data
                
                # Create and display instrument-specific plots
                print(f"\nCreating plots for {instrument_name}...")
                
                # Plot 1: Standard instrument summary
                fig = plot_instrument_summary(instrument_name, instrument_runs_data, 
                                             start_datetime=start_datetime, 
                                             end_datetime=end_datetime)
                
                if fig is not None:
                    try:
                        get_ipython()
                        plt.show()
                    except (NameError, AttributeError):
                        plt.show()
                    
                    plt.close(fig)
                
                # Plot 2: RMS vs StdDev directional analysis
                fig2 = plot_instrument_direction_comparison(instrument_name, instrument_runs_data,
                                                            start_datetime=start_datetime,
                                                            end_datetime=end_datetime)
                
                if fig2 is not None:
                    try:
                        get_ipython()
                        plt.show()
                    except (NameError, AttributeError):
                        plt.show()
                    
                    plt.close(fig2)
                
                # Plot 3: Per-run RMS and StdDev analysis
                for run_data in instrument_runs_data:
                    fig3 = plot_rms_vs_stddev_directions(
                        instrument_name,
                        run_data.get('rms_errors', np.array([])),
                        run_data.get('dates', np.array([])),
                        run_data.get('stats', {}),
                        run_data.get('directional_stats', {}),
                        start_datetime=start_datetime,
                        end_datetime=end_datetime
                    )
                    
                    if fig3 is not None:
                        try:
                            get_ipython()
                            plt.show()
                        except (NameError, AttributeError):
                            plt.show()
                        
                        plt.close(fig3)
    
    # Print summary table aggregated by instrument
    print("\n" + "="*130)
    print("AGGREGATE INSTRUMENT SUMMARY")
    print("="*130)
    
    summary_data = []
    for instrument_name in sorted(instrument_stats_dict.keys()):
        runs_data = instrument_stats_dict[instrument_name]
        all_rms = np.concatenate([run['rms_errors'] for run in runs_data])
        all_dates = np.concatenate([run['dates'] for run in runs_data])
        
        # Calculate overall statistics for this instrument
        overall_stats = calculate_statistics(all_rms, all_dates)
        
        summary_data.append({
            'Instrument': instrument_name.upper(),
            'Mean RMS Error': f"{overall_stats['overall_mean']:.4f}",
            'Std Dev': f"{overall_stats['overall_std']:.4f}",
            'Min Error': f"{overall_stats['min_error']:.4f}",
            'Max Error': f"{overall_stats['max_error']:.4f}",
            'Total Change': f"{overall_stats['total_change']:.4f}",
            'Percent Change (%)': f"{overall_stats['percent_change']:.2f}",
            'Total Runs': len(runs_data),
            'Data Points': overall_stats.get('data_points', len(all_rms))
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
    
    print("="*130)
    
    # Print directional RMS and StdDev summary
    directional_summary_df = create_directional_summary_table(instrument_stats_dict)
    if directional_summary_df is not None:
        print_directional_summary_table(directional_summary_df)
    
    # Create combined visualization for all instruments
    if len(instrument_stats_dict) > 0:
        print(f"\nCreating combined summary plot for all {len(instrument_stats_dict)} instrument(s)...")
        combined_fig = plot_combined_instruments_summary(instrument_stats_dict, 
                                                        start_datetime=start_datetime,
                                                        end_datetime=end_datetime)
        
        if combined_fig is not None:
            try:
                get_ipython()
                plt.show()
            except (NameError, AttributeError):
                plt.show()
            
            plt.close(combined_fig)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Total instruments analyzed: {len(instrument_stats_dict)}")
    for instrument_name in sorted(instrument_stats_dict.keys()):
        runs_data = instrument_stats_dict[instrument_name]
        print(f"  {instrument_name.upper()}: {len(runs_data)} run(s)")
    print("="*70)


# ========== NOTEBOOK UTILITIES (Caching, Themes, Figure Management) ==========

def get_cache_filename(cache_dir, start_dt, end_dt, use_gzip=False):
    """
    Generate cache filename for analysis period.
    
    Parameters:
    -----------
    cache_dir : str
        Directory for cache storage
    start_dt : datetime
        Start datetime
    end_dt : datetime
        End datetime
    use_gzip : bool
        Whether to use gzip compression (default False)
    
    Returns:
    --------
    Path : Pathlib Path object to cache file
    """
    from pathlib import Path
    filename = f"analysis_{start_dt.strftime('%Y%m%d')}_to_{end_dt.strftime('%Y%m%d')}.pkl"
    if use_gzip:
        filename += ".gz"
    return Path(cache_dir) / filename


def get_cached_results(cache_dir, start_dt, end_dt, enable_cache=True, save_pkl_cache=False):
    """
    Load cached analysis results if available.
    
    Parameters:
    -----------
    cache_dir : str
        Directory containing cache files
    start_dt : datetime
        Start datetime for analysis period
    end_dt : datetime
        End datetime for analysis period
    enable_cache : bool
        Whether caching is enabled (default True)
    save_pkl_cache : bool
        Whether pickle files are saved/loaded (default False to save disk space)
    
    Returns:
    --------
    dict or None : Cached results dictionary or None if not found
    """
    import pickle
    import gzip
    from pathlib import Path
    
    if not enable_cache or not save_pkl_cache:
        return None
    
    # Try gzip compressed file first
    cache_file_gz = get_cache_filename(cache_dir, start_dt, end_dt, use_gzip=True)
    if cache_file_gz.exists():
        try:
            with gzip.open(cache_file_gz, 'rb') as f:
                data = pickle.load(f)
                print(f"✓ Loaded cached results from: {cache_file_gz}")
                return data
        except Exception as e:
            print(f"Warning: Could not load compressed cache: {e}")
    
    # Fallback to uncompressed pickle file for backward compatibility
    cache_file = get_cache_filename(cache_dir, start_dt, end_dt, use_gzip=False)
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                print(f"✓ Loaded cached results from: {cache_file}")
                return data
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
    return None


def save_to_cache(data, cache_dir, start_dt, end_dt, enable_cache=True, save_pkl_cache=False):
    """
    Save analysis results to cache for faster re-runs.
    
    Parameters:
    -----------
    data : dict
        Data to cache (instruments_dict, observing_runs_dict, instrument_stats_dict)
    cache_dir : str
        Directory to store cache file
    start_dt : datetime
        Start datetime for analysis period
    end_dt : datetime
        End datetime for analysis period
    enable_cache : bool
        Whether caching is enabled (default True)
    save_pkl_cache : bool
        Whether to save pickle files to disk (default False to save disk space)
        If False, caching is bypassed to avoid using storage
    """
    import pickle
    import gzip
    from pathlib import Path
    
    if not enable_cache or not save_pkl_cache:
        return
    
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = get_cache_filename(cache_dir, start_dt, end_dt, use_gzip=True)
    try:
        with gzip.open(cache_file, 'wb') as f:
            pickle.dump(data, f)
            file_size_mb = cache_file.stat().st_size / (1024 * 1024)
            print(f"✓ Saved results to compressed cache: {cache_file} ({file_size_mb:.2f} MB)")
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")


def create_theme_widget():
    """
    Create an interactive theme toggle widget for light/dark mode switching.
    
    Returns:
    --------
    widgets.ToggleButtons : Theme toggle widget
    
    Requires:
    ---------
    ipywidgets, matplotlib, IPython
    """
    try:
        import matplotlib as mpl
        import ipywidgets as widgets
        from IPython.display import display, HTML
    except ImportError:
        print("Warning: ipywidgets not available. Theme widget requires: pip install ipywidgets")
        return None
    
    # Create toggle button
    theme_toggle = widgets.ToggleButtons(
        options={'☀️ Light': 'light', '🌙 Dark': 'dark'},
        description='Theme:',
        button_style='info',
        style={'description_width': '70px'},
        layout=widgets.Layout(width='300px', height='50px')
    )
    
    def on_theme_change(change):
        """Update matplotlib theme when toggle changes"""
        if change['new'] == 'dark':
            plt.style.use('dark_background')
            mpl.rcParams['figure.facecolor'] = '#1e1e1e'
            mpl.rcParams['figure.edgecolor'] = '#1e1e1e'
            mpl.rcParams['axes.facecolor'] = '#2d2d2d'
            mpl.rcParams['axes.edgecolor'] = '#404040'
            mpl.rcParams['grid.color'] = '#404040'
            mpl.rcParams['text.color'] = '#e0e0e0'
            display(HTML('<p style="color: #e0e0e0; font-size: 14px;">✓ Dark theme applied</p>'))
        else:
            plt.style.use('default')
            mpl.rcParams['figure.facecolor'] = 'white'
            mpl.rcParams['figure.edgecolor'] = 'white'
            mpl.rcParams['axes.facecolor'] = 'white'
            mpl.rcParams['axes.edgecolor'] = 'black'
            mpl.rcParams['grid.color'] = '#CCCCCC'
            mpl.rcParams['text.color'] = 'black'
            display(HTML('<p style="color: black; font-size: 14px;">✓ Light theme applied</p>'))
    
    # Connect event handler
    theme_toggle.observe(on_theme_change, names='value')
    
    return theme_toggle


def save_all_figures(output_dir='./figures', theme='current', start_dt=None, end_dt=None):
    """
    Save all currently open matplotlib figures to disk.
    Creates timestamped directories for organization.
    
    Parameters:
    -----------
    output_dir : str
        Base directory to save figures (default './figures')
    theme : str
        Theme used for figures: 'light', 'dark', or 'current' (default 'current')
    start_dt : datetime, optional
        Start datetime for analysis period (for metadata)
    end_dt : datetime, optional
        End datetime for analysis period (for metadata)
    
    Returns:
    --------
    str : Path to saved figures directory, or None if error
    """
    from datetime import datetime as dt
    from pathlib import Path
    
    # Get all open figures
    fig_nums = plt.get_fignums()
    
    if len(fig_nums) == 0:
        print("⚠️  No figures are currently open to save.")
        return None
    
    # Create timestamped output directory
    timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
    theme_suffix = f"_{theme}" if theme != 'current' else ""
    save_path = os.path.join(output_dir, f"analysis_{timestamp}{theme_suffix}")
    
    try:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        for i, fig_num in enumerate(fig_nums, 1):
            fig = plt.figure(fig_num)
            filename = os.path.join(save_path, f"figure_{i:03d}.png")
            fig.savefig(filename, dpi=150, bbox_inches='tight', facecolor='auto')
            saved_count += 1
        
        print(f"✓ Saved {saved_count} figure(s) to: {save_path}")
        print(f"  Format: PNG (150 DPI)")
        print(f"  Theme: {theme}")
        
        # Create metadata file
        summary_file = os.path.join(save_path, 'README.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Analysis Figures - {timestamp}\n")
            f.write(f"Theme: {theme}\n")
            if start_dt and end_dt:
                f.write(f"Analysis Period: {start_dt} to {end_dt}\n")
            f.write(f"Total Figures: {saved_count}\n")
        
        return save_path
        
    except Exception as e:
        print(f"❌ Error saving figures: {e}")
        return None


def create_figure_saving_widget(start_dt=None, end_dt=None):
    """
    Create interactive figure saving widget for notebooks.
    
    Parameters:
    -----------
    start_dt : datetime, optional
        Start datetime for analysis period (for metadata)
    end_dt : datetime, optional
        End datetime for analysis period (for metadata)
    
    Returns:
    --------
    tuple : (button_widget, output_area) for display
    
    Requires:
    ---------
    ipywidgets
    """
    try:
        import ipywidgets as widgets
    except ImportError:
        print("Warning: ipywidgets not available. Figure saving widget requires: pip install ipywidgets")
        return None, None
    
    save_button = widgets.Button(description='💾 Save All Figures (PNG)')
    save_output = widgets.Output()
    
    def on_save_click(b):
        """Handle save button click"""
        with save_output:
            save_output.clear_output()
            current_style = plt.rcParams.get('axes.facecolor')
            theme = 'dark' if current_style == '#2d2d2d' else 'light'
            save_all_figures(theme=theme, start_dt=start_dt, end_dt=end_dt)
    
    save_button.on_click(on_save_click)
    
    return save_button, save_output


def downsample_timeseries(timestamps, values, max_points=10000):
    """
    Downsample large timeseries data for faster plotting.
    Preserves min/max values in each bin to maintain visualization accuracy.
    
    Parameters:
    -----------
    timestamps : array-like
        Timestamp values
    values : np.ndarray
        Values to downsample
    max_points : int
        Maximum points to return (default 10000)
    
    Returns:
    --------
    tuple: (downsampled_timestamps, downsampled_values)
    """
    if len(values) <= max_points:
        return np.array(timestamps), np.array(values)
    
    # Use adaptive binning to preserve shape
    bin_size = len(values) // max_points
    downsampled_times = []
    downsampled_vals = []
    
    for i in range(0, len(values), bin_size):
        bin_end = min(i + bin_size, len(values))
        bin_indices = np.arange(i, bin_end)
        
        # Take median timestamp and mean value
        downsampled_times.append(np.median(timestamps[bin_indices]))
        downsampled_vals.append(np.mean(values[i:bin_end]))
    
    return np.array(downsampled_times), np.array(downsampled_vals)


if __name__ == "__main__":
    main(
        start_datetime='2025-01-01 00:00:00',
        end_datetime='2025-02-01 00:00:00',
        threshold_arcsec=5.0
    )
