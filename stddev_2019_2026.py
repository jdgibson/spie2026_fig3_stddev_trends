#!/usr/bin/env python3
"""
Standalone script to analyze and plot median standard deviation of elevation error from 2019-2026.

This script loads JSON files from the ./notebooks/results/ directory and creates a time-series plot
of the median standard deviation of telescope elevation errors.

Run from directory containing the ./notebooks/results/ folder with JSON output files.
"""

import json
import os
import sys
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy import stats as scipy_stats
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', message='.*One or more sample arguments is too small.*')

# Configure matplotlib for interactive display in VS Code
HAS_IPYTHON = False
try:
    get_ipython()  # Check if running in interactive environment
    from IPython.display import display, Image
    import matplotlib
    # Force Agg backend for rendering to image
    matplotlib.use('Agg')
    HAS_IPYTHON = True
    print("✓ Running in interactive IPython environment")
except (NameError, ImportError):
    Image = None
    display = lambda x: None
    print("Running as standalone script")


def load_json_data(json_dir='./notebooks/results'):
    """Load all JSON files from the notebooks/results directory."""
    json_path = Path(json_dir)
    
    if not json_path.exists():
        print(f"Error: Directory '{json_dir}' not found")
        print("Make sure you're in a directory with a ./notebooks/results/ folder containing JSON output files")
        sys.exit(1)
    
    all_data = {}
    json_files = sorted([f for f in json_path.glob('*_20*.json')])
    
    if not json_files:
        print(f"Error: No JSON files with '_20' in filename found in '{json_dir}'")
        sys.exit(1)
    
    print(f"Found {len(json_files)} JSON files with '_20' in the filename\n")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                all_data[json_file.stem] = json.load(f)
            print(f"  ✓ {json_file.name}")
        except Exception as e:
            print(f"  ✗ Error loading {json_file.name}: {e}")
    
    return all_data


def extract_data_points(all_data):
    """Extract mean and median standard deviations with timestamps and 1-sigma errors from JSON data."""
    data_points = []
    
    for filename, data in all_data.items():
        try:
            metadata = data.get('metadata', {})
            stats = data.get('statistics_by_direction', {}).get('BOTH', {}).get('standard_deviation', {})
            
            run_start = metadata.get('run_start_datetime')
            mean_stddev = stats.get('mean_stddev_arcsec')
            median_stddev = stats.get('median_stddev_arcsec', mean_stddev)  # Use median if available, fall back to mean
            std_stddev = stats.get('std_stddev_arcsec', 0)  # Extract 1-sigma error
            
            if run_start and mean_stddev is not None:
                data_points.append({
                    'filename': filename,
                    'datetime': pd.to_datetime(run_start),
                    'mean_stddev_arcsec': mean_stddev,
                    'median_stddev_arcsec': median_stddev if median_stddev is not None else mean_stddev,
                    'std_stddev_arcsec': std_stddev,
                    'instrument': metadata.get('instrument', 'UNKNOWN')
                })
        except Exception as e:
            print(f"  Warning: Error extracting data from {filename}: {e}")
    
    return pd.DataFrame(data_points).sort_values('datetime').reset_index(drop=True)


def plot_stddev_timeseries(df, output_file=None):
    """Create time-series plot of median standard deviation."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot median standard deviation as primary metric
    ax.plot(df['datetime'], df['median_stddev_arcsec'], 
            marker='o', linestyle='-', linewidth=2.5, markersize=5, 
            color='#2E86AB', label='Median Std Dev', zorder=3)
    
    # Fill area under curve
    ax.fill_between(df['datetime'], df['median_stddev_arcsec'], 
                     alpha=0.2, color='#2E86AB', zorder=2)
    
    # Add mean standard deviation as reference
    ax.plot(df['datetime'], df['mean_stddev_arcsec'], 
            marker='s', linestyle='--', linewidth=1.5, markersize=3, 
            color='#A23B72', alpha=0.6, label='Mean Std Dev (Reference)', zorder=2)
    
    ax.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax.set_ylabel('Standard Deviation of Elevation Error (arcsec)', fontsize=13, fontweight='bold')
    ax.set_title('Tracking Stability Trends (2019-2026)\nMedian Standard Deviation of Elevation Error', 
                 fontsize=15, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=1)
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    
    # Format x-axis
    fig.autofmt_xdate(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save to file
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
    
    # Display in interactive environment
    if HAS_IPYTHON and Image:
        # Render to PNG in memory
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        display(Image(data=buf.getvalue()))
        print("  ✓ Displayed in interactive window")
    
    plt.close(fig)
    
    return fig, ax


def plot_by_instrument(df, output_file=None):
    """Create subplots by instrument."""
    instruments = sorted(df['instrument'].unique())
    n_instruments = len(instruments)
    
    fig, axes = plt.subplots(n_instruments, 1, figsize=(16, 4 * n_instruments))
    if n_instruments == 1:
        axes = [axes]
    
    for ax, instrument in zip(axes, instruments):
        df_inst = df[df['instrument'] == instrument]
        
        ax.plot(df_inst['datetime'], df_inst['median_stddev_arcsec'], 
                marker='o', linestyle='-', linewidth=2, markersize=4, 
                color='#2E86AB', label='Median Std Dev')
        ax.fill_between(df_inst['datetime'], df_inst['median_stddev_arcsec'], 
                         alpha=0.2, color='#2E86AB')
        
        ax.set_ylabel('Std Dev (arcsec)', fontsize=11, fontweight='bold')
        ax.set_title(f'{instrument} - Median Standard Deviation Over Time', 
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9)
    
    axes[-1].set_xlabel('Date', fontsize=12, fontweight='bold')
    
    fig.autofmt_xdate(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save to file
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
    
    # Display in interactive environment
    if HAS_IPYTHON and Image:
        # Render to PNG in memory
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        display(Image(data=buf.getvalue()))
        print("  ✓ Displayed in interactive window")
    
    plt.close(fig)
    
    return fig, axes


def plot_combined_comparison(df, output_file=None):
    """Create figure with two subplots:
    Top: Overall tracking stability trends (2019-2026) with linear regression
    Bottom: Combined blue, binospec, hecto, mmirs data with linear regressions for each
    """
    # Define instrument colors
    instrument_colors = {
        'blue': '#1f77b4',
        'binospec': '#ff7f0e',
        'hecto': '#2ca02c',
        'mmirs': '#d62728'
    }
    
    # Create figure with 2 subplots stacked vertically
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(18, 14), sharex=True)
    
    # Convert datetime to numeric days for regression
    df_copy = df.copy()
    df_copy['time_numeric'] = (df_copy['datetime'] - df_copy['datetime'].min()).dt.days
    
    # ===== TOP SUBPLOT: Overall trend with linear regression =====
    ax_top.errorbar(df_copy['datetime'], df_copy['median_stddev_arcsec'], 
                    yerr=df_copy['std_stddev_arcsec'], fmt='none', elinewidth=1.5, capsize=3, 
                    capthick=1, color='#2E86AB', alpha=0.4, zorder=2.5, label='1σ Error')
    ax_top.plot(df_copy['datetime'], df_copy['median_stddev_arcsec'], 
                marker='o', linestyle='-', linewidth=3, markersize=8, 
                color='#2E86AB', label='All Instruments - Median', zorder=3)
    ax_top.fill_between(df_copy['datetime'], df_copy['median_stddev_arcsec'], 
                        alpha=0.2, color='#2E86AB', zorder=2)
    
    # Calculate linear regression for overall data
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*One or more sample arguments is too small.*')
        slope_overall, intercept_overall, r_value_overall, p_value_overall, std_err_overall = scipy_stats.linregress(
            df_copy['time_numeric'], df_copy['median_stddev_arcsec'])
    
    # Plot regression line for overall data
    x_reg = np.array([df_copy['time_numeric'].min(), df_copy['time_numeric'].max()])
    y_reg_overall = slope_overall * x_reg + intercept_overall
    datetime_reg = df_copy['datetime'].min() + pd.to_timedelta(x_reg, unit='D')
    ax_top.plot(datetime_reg, y_reg_overall, '--', linewidth=2.5, color='red', 
                label=f'Regression (R²={r_value_overall**2:.4f})', zorder=4, alpha=0.8)
    
    # Add regression equation to top subplot
    equation_overall = f'y = {slope_overall:.6f}x + {intercept_overall:.4f}'
    ax_top.text(0.02, 0.95, equation_overall, transform=ax_top.transAxes, 
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.8), family='monospace')
    
    ax_top.set_ylabel('Std Dev (arcsec)', fontsize=16, fontweight='bold')
    ax_top.set_title('Tracking Stability Trends (2019-2026)\nOverall Mean of All Instruments', 
                     fontsize=18, fontweight='bold', pad=20)
    ax_top.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax_top.legend(loc='best', fontsize=13, framealpha=0.95)
    ax_top.tick_params(axis='both', which='major', labelsize=13)
    
    # ===== BOTTOM SUBPLOT: Individual instruments with linear regressions =====
    target_instruments = ['blue', 'binospec', 'hecto', 'mmirs']
    equations_bottom = []
    
    for instrument in target_instruments:
        df_inst = df_copy[df_copy['instrument'].str.lower() == instrument]
        if len(df_inst) > 0:
            color = instrument_colors.get(instrument, '#000000')
            ax_bottom.plot(df_inst['datetime'], df_inst['median_stddev_arcsec'], 
                          marker='o', linestyle='-', linewidth=3, markersize=8, 
                          color=color, label=instrument.upper(), zorder=2)
            
            # Calculate linear regression for this instrument
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*One or more sample arguments is too small.*')
                slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                    df_inst['time_numeric'], df_inst['median_stddev_arcsec'])
            
            # Plot regression line for this instrument
            x_reg_inst = np.array([df_inst['time_numeric'].min(), df_inst['time_numeric'].max()])
            y_reg = slope * x_reg_inst + intercept
            datetime_reg_inst = df_inst['datetime'].min() + pd.to_timedelta(x_reg_inst, unit='D')
            ax_bottom.plot(datetime_reg_inst, y_reg, '--', linewidth=2, color=color, 
                          alpha=0.7, zorder=1)
            
            # Store equation for display
            equations_bottom.append(f'{instrument.upper()}: y = {slope:.6f}x + {intercept:.4f} (R²={r_value**2:.4f})')
    
    # Add all regression equations to bottom subplot
    equations_text = '\n'.join(equations_bottom)
    ax_bottom.text(0.02, 0.95, equations_text, transform=ax_bottom.transAxes, 
                   fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='lightblue', alpha=0.8), family='monospace')
    
    ax_bottom.set_xlabel('Date', fontsize=16, fontweight='bold')
    ax_bottom.set_ylabel('Std Dev (arcsec)', fontsize=16, fontweight='bold')
    ax_bottom.set_title('Instrument Comparison: BLUE, BINOSPEC, HECTO, MMIRS', 
                        fontsize=18, fontweight='bold', pad=20)
    ax_bottom.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax_bottom.legend(loc='best', fontsize=13, framealpha=0.95, ncol=4)
    ax_bottom.tick_params(axis='both', which='major', labelsize=13)
    
    # Format x-axis for date display
    fig.autofmt_xdate(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save to file
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
    
    # Display in interactive environment
    if HAS_IPYTHON and Image:
        # Render to PNG in memory
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        display(Image(data=buf.getvalue()))
        print("  ✓ Displayed in interactive window")
    
    plt.close(fig)
    
    return fig, (ax_top, ax_bottom)


def extract_up_down_data(all_data):
    """Extract UP vs DOWN standard deviation data with timestamps and 1-sigma errors from JSON data."""
    data_points = []
    
    for filename, data in all_data.items():
        try:
            metadata = data.get('metadata', {})
            stats_both = data.get('statistics_by_direction', {}).get('BOTH', {}).get('standard_deviation', {})
            stats_up = data.get('statistics_by_direction', {}).get('UP', {}).get('standard_deviation', {})
            stats_down = data.get('statistics_by_direction', {}).get('DOWN', {}).get('standard_deviation', {})
            
            run_start = metadata.get('run_start_datetime')
            mean_stddev_both = stats_both.get('mean_stddev_arcsec')
            mean_stddev_up = stats_up.get('mean_stddev_arcsec')
            mean_stddev_down = stats_down.get('mean_stddev_arcsec')
            std_stddev_up = stats_up.get('std_stddev_arcsec', 0)
            std_stddev_down = stats_down.get('std_stddev_arcsec', 0)
            
            if run_start and mean_stddev_both is not None and mean_stddev_up is not None and mean_stddev_down is not None:
                data_points.append({
                    'filename': filename,
                    'datetime': pd.to_datetime(run_start),
                    'mean_stddev_up': mean_stddev_up,
                    'mean_stddev_down': mean_stddev_down,
                    'std_stddev_up': std_stddev_up,
                    'std_stddev_down': std_stddev_down,
                    'instrument': metadata.get('instrument', 'UNKNOWN')
                })
        except Exception as e:
            print(f"  Warning: Error extracting UP/DOWN data from {filename}: {e}")
    
    return pd.DataFrame(data_points).sort_values('datetime').reset_index(drop=True)


def plot_up_down_comparison(all_data, output_file=None):
    """Create figure comparing UP vs DOWN standard deviations with linear regressions."""
    df = extract_up_down_data(all_data)
    
    if len(df) == 0:
        print("  Warning: No UP/DOWN comparison data available")
        return None, None
    
    # Convert datetime to numeric days for regression
    df_copy = df.copy()
    df_copy['time_numeric'] = (df_copy['datetime'] - df_copy['datetime'].min()).dt.days
    
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(18, 14), sharex=True)
    
    # ===== TOP SUBPLOT: UP vs DOWN on same plot =====
    ax_top.errorbar(df_copy['datetime'], df_copy['mean_stddev_up'], 
                    yerr=df_copy['std_stddev_up'], fmt='none', elinewidth=1.5, capsize=3, 
                    capthick=1, color='#2E86AB', alpha=0.3, zorder=2.5)
    ax_top.plot(df_copy['datetime'], df_copy['mean_stddev_up'], 
                marker='o', linestyle='-', linewidth=3, markersize=8, 
                color='#2E86AB', label='UP (Elevation Increase)', zorder=3)
    ax_top.fill_between(df_copy['datetime'], df_copy['mean_stddev_up'], 
                        alpha=0.15, color='#2E86AB', zorder=2)
    
    ax_top.errorbar(df_copy['datetime'], df_copy['mean_stddev_down'], 
                    yerr=df_copy['std_stddev_down'], fmt='none', elinewidth=1.5, capsize=3, 
                    capthick=1, color='#D62728', alpha=0.3, zorder=2.5)
    ax_top.plot(df_copy['datetime'], df_copy['mean_stddev_down'], 
                marker='s', linestyle='-', linewidth=3, markersize=8, 
                color='#D62728', label='DOWN (Elevation Decrease)', zorder=3)
    ax_top.fill_between(df_copy['datetime'], df_copy['mean_stddev_down'], 
                        alpha=0.15, color='#D62728', zorder=2)
    
    # Calculate linear regressions
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*One or more sample arguments is too small.*')
        slope_up, intercept_up, r_value_up, _, _ = scipy_stats.linregress(
            df_copy['time_numeric'], df_copy['mean_stddev_up'])
        slope_down, intercept_down, r_value_down, _, _ = scipy_stats.linregress(
            df_copy['time_numeric'], df_copy['mean_stddev_down'])
    
    # Plot regression lines
    x_reg = np.array([df_copy['time_numeric'].min(), df_copy['time_numeric'].max()])
    y_reg_up = slope_up * x_reg + intercept_up
    y_reg_down = slope_down * x_reg + intercept_down
    datetime_reg = df_copy['datetime'].min() + pd.to_timedelta(x_reg, unit='D')
    
    ax_top.plot(datetime_reg, y_reg_up, '--', linewidth=2.5, color='#2E86AB', alpha=0.8, zorder=4)
    ax_top.plot(datetime_reg, y_reg_down, '--', linewidth=2.5, color='#D62728', alpha=0.8, zorder=4)
    
    ax_top.set_ylabel('Std Dev (arcsec)', fontsize=16, fontweight='bold')
    ax_top.set_title('UP vs DOWN Direction Comparison\nStandard Deviation of Elevation Error', 
                     fontsize=18, fontweight='bold', pad=20)
    ax_top.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax_top.legend(loc='best', fontsize=13, framealpha=0.95)
    ax_top.tick_params(axis='both', which='major', labelsize=13)
    
    # Regression info
    regression_text = f'UP: y = {slope_up:.6f}x + {intercept_up:.4f} (R²={r_value_up**2:.4f})\n'
    regression_text += f'DOWN: y = {slope_down:.6f}x + {intercept_down:.4f} (R²={r_value_down**2:.4f})'
    ax_top.text(0.02, 0.95, regression_text, transform=ax_top.transAxes, 
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.8), family='monospace')
    
    # ===== BOTTOM SUBPLOT: UP - DOWN difference =====
    df_copy['difference'] = df_copy['mean_stddev_up'] - df_copy['mean_stddev_down']
    df_copy['difference_std'] = np.sqrt(df_copy['std_stddev_up']**2 + df_copy['std_stddev_down']**2)
    
    ax_bottom.errorbar(df_copy['datetime'], df_copy['difference'], 
                       yerr=df_copy['difference_std'], fmt='none', elinewidth=1.5, capsize=3, 
                       capthick=1, color='#FF7F0E', alpha=0.4, zorder=2.5)
    ax_bottom.plot(df_copy['datetime'], df_copy['difference'], 
                   marker='D', linestyle='-', linewidth=2.5, markersize=7, 
                   color='#FF7F0E', label='UP - DOWN Difference', zorder=3)
    ax_bottom.fill_between(df_copy['datetime'], df_copy['difference'], 
                           alpha=0.15, color='#FF7F0E', zorder=2)
    
    # Zero reference line
    ax_bottom.axhline(y=0, color='black', linestyle=':', linewidth=1.5, alpha=0.5, zorder=1)
    
    # Regression for difference
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*One or more sample arguments is too small.*')
        slope_diff, intercept_diff, r_value_diff, _, _ = scipy_stats.linregress(
            df_copy['time_numeric'], df_copy['difference'])
    
    y_reg_diff = slope_diff * x_reg + intercept_diff
    ax_bottom.plot(datetime_reg, y_reg_diff, '--', linewidth=2.5, color='#FF7F0E', alpha=0.8, zorder=4)
    
    ax_bottom.set_xlabel('Date', fontsize=16, fontweight='bold')
    ax_bottom.set_ylabel('Std Dev Difference (arcsec)', fontsize=16, fontweight='bold')
    ax_bottom.set_title('UP - DOWN Difference (Positive = UP has higher variability)', 
                        fontsize=16, fontweight='bold', pad=20)
    ax_bottom.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax_bottom.legend(loc='best', fontsize=13, framealpha=0.95)
    ax_bottom.tick_params(axis='both', which='major', labelsize=13)
    
    # Regression info for difference
    diff_text = f'Trend: y = {slope_diff:.6f}x + {intercept_diff:.4f} (R²={r_value_diff**2:.4f})'
    ax_bottom.text(0.02, 0.95, diff_text, transform=ax_bottom.transAxes, 
                   fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='lightblue', alpha=0.8), family='monospace')
    
    fig.autofmt_xdate(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save to file
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
    
    # Display in interactive environment
    if HAS_IPYTHON and Image:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        display(Image(data=buf.getvalue()))
        print("  ✓ Displayed in interactive window")
    
    plt.close(fig)
    
    return fig, (ax_top, ax_bottom)


def print_summary_statistics(df):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nDate Range:")
    print(f"  First observation: {df['datetime'].min().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Last observation:  {df['datetime'].max().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration: {(df['datetime'].max() - df['datetime'].min()).days} days")
    
    print(f"\nMedian Standard Deviation Statistics:")
    print(f"  Mean of medians:  {df['median_stddev_arcsec'].mean():.6f} arcsec")
    print(f"  Std Dev:          {df['median_stddev_arcsec'].std():.6f} arcsec")
    print(f"  Min:              {df['median_stddev_arcsec'].min():.6f} arcsec")
    print(f"  25th percentile:  {df['median_stddev_arcsec'].quantile(0.25):.6f} arcsec")
    print(f"  Median:           {df['median_stddev_arcsec'].median():.6f} arcsec")
    print(f"  75th percentile:  {df['median_stddev_arcsec'].quantile(0.75):.6f} arcsec")
    print(f"  Max:              {df['median_stddev_arcsec'].max():.6f} arcsec")
    
    # Find min/max with timestamps
    min_idx = df['median_stddev_arcsec'].idxmin()
    max_idx = df['median_stddev_arcsec'].idxmax()
    print(f"\n  Best (min):  {df.loc[min_idx, 'median_stddev_arcsec']:.6f} arcsec on {df.loc[min_idx, 'datetime'].strftime('%Y-%m-%d')} ({df.loc[min_idx, 'instrument']})")
    print(f"  Worst (max): {df.loc[max_idx, 'median_stddev_arcsec']:.6f} arcsec on {df.loc[max_idx, 'datetime'].strftime('%Y-%m-%d')} ({df.loc[max_idx, 'instrument']})")
    
    # By instrument
    print(f"\nStatistics by Instrument:")
    for instrument in sorted(df['instrument'].unique()):
        df_inst = df[df['instrument'] == instrument]
        print(f"\n  {instrument}:")
        print(f"    Count:  {len(df_inst)} observations")
        print(f"    Mean:   {df_inst['median_stddev_arcsec'].mean():.6f} arcsec")
        print(f"    Median: {df_inst['median_stddev_arcsec'].median():.6f} arcsec")
        print(f"    Std:    {df_inst['median_stddev_arcsec'].std():.6f} arcsec")
        print(f"    Range:  [{df_inst['median_stddev_arcsec'].min():.6f}, {df_inst['median_stddev_arcsec'].max():.6f}] arcsec")
    
    print(f"\n" + "="*80)


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("ELEVATION ERROR STANDARD DEVIATION ANALYSIS (2019-2026)")
    print("="*80 + "\n")
    
    # Load JSON data
    print("Step 1: Loading JSON files...\n")
    all_data = load_json_data('./notebooks/results')
    print(f"\n✓ Total files loaded: {len(all_data)}\n")
    
    # Extract data points
    print("Step 2: Extracting data points...")
    df = extract_data_points(all_data)
    print(f"✓ Extracted {len(df)} data points\n")
    
    if len(df) == 0:
        print("Error: No data could be extracted from JSON files")
        sys.exit(1)
    
    # Print summary statistics
    print_summary_statistics(df)
    
    # Create plots
    print("\nStep 3: Generating visualizations...\n")
    print("Creating time-series plot...")
    plot_stddev_timeseries(df, output_file='stddev_timeseries_2019_2026.png')
    
    print("\nCreating plots by instrument...")
    plot_by_instrument(df, output_file='stddev_by_instrument_2019_2026.png')
    
    print("\nCreating combined comparison plot (Overall + Target Instruments)...")
    plot_combined_comparison(df, output_file='stddev_combined_comparison_2019_2026.png')
    
    print("\nCreating UP vs DOWN direction comparison plot...")
    plot_up_down_comparison(all_data, output_file='stddev_up_vs_down_2019_2026.png')
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print("\n📊 Generated files:")
    print("  • stddev_timeseries_2019_2026.png")
    print("  • stddev_by_instrument_2019_2026.png")
    print("  • stddev_combined_comparison_2019_2026.png")
    print("  • stddev_up_vs_down_2019_2026.png")
    print("\n📈 Plots displayed in the VS Code interactive window above.\n")


if __name__ == '__main__':
    main()
