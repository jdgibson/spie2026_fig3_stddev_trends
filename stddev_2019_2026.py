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
    excluded_instruments = ['red', 'spol']  # Exclude these instruments
    
    for filename, data in all_data.items():
        try:
            metadata = data.get('metadata', {})
            stats = data.get('statistics_by_direction', {}).get('BOTH', {}).get('standard_deviation', {})
            
            run_start = metadata.get('run_start_datetime')
            mean_stddev = stats.get('mean_stddev_arcsec')
            median_stddev = stats.get('median_stddev_arcsec', mean_stddev)  # Use median if available, fall back to mean
            std_stddev = stats.get('std_stddev_arcsec', 0)  # Extract 1-sigma error
            instrument = metadata.get('instrument', 'UNKNOWN')
            
            # Skip excluded instruments
            if instrument.lower() in excluded_instruments:
                continue
            
            if run_start and mean_stddev is not None:
                data_points.append({
                    'filename': filename,
                    'datetime': pd.to_datetime(run_start),
                    'mean_stddev_arcsec': mean_stddev,
                    'median_stddev_arcsec': median_stddev if median_stddev is not None else mean_stddev,
                    'std_stddev_arcsec': std_stddev,
                    'instrument': instrument
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
    excluded_instruments = ['red', 'spol']  # Exclude these instruments
    
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
            instrument = metadata.get('instrument', 'UNKNOWN')
            
            # Skip excluded instruments
            if instrument.lower() in excluded_instruments:
                continue
            
            if run_start and mean_stddev_both is not None and mean_stddev_up is not None and mean_stddev_down is not None:
                data_points.append({
                    'filename': filename,
                    'datetime': pd.to_datetime(run_start),
                    'mean_stddev_up': mean_stddev_up,
                    'mean_stddev_down': mean_stddev_down,
                    'std_stddev_up': std_stddev_up,
                    'std_stddev_down': std_stddev_down,
                    'instrument': instrument
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


def plot_violin_by_instrument(df, output_file=None):
    """Create violin plot comparing standard deviations by instrument."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for violin plot
    instruments = sorted(df['instrument'].unique())
    data_to_plot = [df[df['instrument'] == inst]['median_stddev_arcsec'].values for inst in instruments]
    
    # Create violin plot
    parts = ax.violinplot(data_to_plot, positions=range(len(instruments)), 
                          showmeans=True, showmedians=True)
    
    # Customize violin plot colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, pc in enumerate(parts['bodies']):
        if i < len(colors):
            pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    # Customize median and mean lines
    parts['cmedians'].set_edgecolor('red')
    parts['cmedians'].set_linewidth(2)
    parts['cmeans'].set_edgecolor('blue')
    parts['cmeans'].set_linewidth(2)
    
    # Add individual points as scatter
    for i, instrument in enumerate(instruments):
        y = df[df['instrument'] == instrument]['median_stddev_arcsec'].values
        x = np.random.normal(i, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.4, s=30, color='gray', zorder=2)
    
    ax.set_xticks(range(len(instruments)))
    ax.set_xticklabels([inst.upper() for inst in instruments], fontsize=12, fontweight='bold')
    ax.set_ylabel('Standard Deviation of Elevation Error (arcsec)', fontsize=14, fontweight='bold')
    ax.set_title('Distribution of Pointing Stability by Instrument\n(Violin Plot with Individual Observations)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=0.5)
    
    # Add legend for mean and median lines
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='red', linewidth=2, label='Median'),
                       Line2D([0], [0], color='blue', linewidth=2, label='Mean')]
    ax.legend(handles=legend_elements, loc='best', fontsize=11, framealpha=0.95)
    
    ax.tick_params(axis='both', which='major', labelsize=11)
    
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
    
    return fig, ax


def plot_violin_up_down(all_data, output_file=None):
    """Create violin plot comparing UP and DOWN standard deviations."""
    df = extract_up_down_data(all_data)
    
    if len(df) == 0:
        print("  Warning: No UP/DOWN comparison data available for violin plot")
        return None, None
    
    fig, (ax_overall, ax_by_instrument) = plt.subplots(1, 2, figsize=(16, 8))
    
    # ===== LEFT PLOT: Overall UP vs DOWN =====
    data_overall = [df['mean_stddev_up'].values, df['mean_stddev_down'].values]
    parts_overall = ax_overall.violinplot(data_overall, positions=[0, 1], 
                                          showmeans=True, showmedians=True, widths=0.7)
    
    # Customize colors
    colors_overall = ['#2E86AB', '#D62728']
    for i, pc in enumerate(parts_overall['bodies']):
        pc.set_facecolor(colors_overall[i])
        pc.set_alpha(0.7)
    
    parts_overall['cmedians'].set_edgecolor('black')
    parts_overall['cmedians'].set_linewidth(2)
    parts_overall['cmeans'].set_edgecolor('orange')
    parts_overall['cmeans'].set_linewidth(2)
    
    # Add scatter points
    x_up = np.random.normal(0, 0.04, size=len(df))
    x_down = np.random.normal(1, 0.04, size=len(df))
    ax_overall.scatter(x_up, df['mean_stddev_up'].values, alpha=0.4, s=30, color='gray', zorder=2)
    ax_overall.scatter(x_down, df['mean_stddev_down'].values, alpha=0.4, s=30, color='gray', zorder=2)
    
    ax_overall.set_xticks([0, 1])
    ax_overall.set_xticklabels(['UP (Elevation Increase)', 'DOWN (Elevation Decrease)'], 
                               fontsize=12, fontweight='bold')
    ax_overall.set_ylabel('Standard Deviation of Elevation Error (arcsec)', fontsize=13, fontweight='bold')
    ax_overall.set_title('Overall UP vs DOWN Comparison', fontsize=14, fontweight='bold', pad=15)
    ax_overall.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=0.5)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='black', linewidth=2, label='Median'),
                       Line2D([0], [0], color='orange', linewidth=2, label='Mean')]
    ax_overall.legend(handles=legend_elements, loc='best', fontsize=11, framealpha=0.95)
    
    ax_overall.tick_params(axis='both', which='major', labelsize=11)
    
    # ===== RIGHT PLOT: UP vs DOWN by instrument =====
    instruments = sorted(df['instrument'].unique())
    n_instruments = len(instruments)
    
    # Prepare data and positions
    data_by_inst = []
    positions = []
    colors_by_inst = []
    pos_counter = 0
    
    for i, instrument in enumerate(instruments):
        df_inst = df[df['instrument'] == instrument]
        if len(df_inst) > 0:
            data_by_inst.append(df_inst['mean_stddev_up'].values)
            positions.append(pos_counter)
            colors_by_inst.append('#2E86AB')
            pos_counter += 1
            
            data_by_inst.append(df_inst['mean_stddev_down'].values)
            positions.append(pos_counter)
            colors_by_inst.append('#D62728')
            pos_counter += 1
            
            pos_counter += 0.5  # Add spacing between instruments
    
    parts_by_inst = ax_by_instrument.violinplot(data_by_inst, positions=positions, 
                                                showmeans=True, showmedians=True, widths=0.6)
    
    # Customize colors
    for i, pc in enumerate(parts_by_inst['bodies']):
        if i < len(colors_by_inst):
            pc.set_facecolor(colors_by_inst[i])
        pc.set_alpha(0.7)
    
    parts_by_inst['cmedians'].set_edgecolor('black')
    parts_by_inst['cmedians'].set_linewidth(2)
    parts_by_inst['cmeans'].set_edgecolor('orange')
    parts_by_inst['cmeans'].set_linewidth(2)
    
    # Add scatter points
    for i, data in enumerate(data_by_inst):
        x = np.random.normal(positions[i], 0.03, size=len(data))
        ax_by_instrument.scatter(x, data, alpha=0.3, s=20, color='gray', zorder=2)
    
    # Set x-axis labels
    label_positions = []
    label_texts = []
    pos_counter = 0
    for instrument in instruments:
        label_positions.append(pos_counter + 0.5)
        label_texts.append(instrument.upper())
        pos_counter += 2.5
    
    ax_by_instrument.set_xticks(label_positions)
    ax_by_instrument.set_xticklabels(label_texts, fontsize=11, fontweight='bold')
    ax_by_instrument.set_ylabel('Standard Deviation of Elevation Error (arcsec)', fontsize=13, fontweight='bold')
    ax_by_instrument.set_title('UP vs DOWN by Instrument\n(Blue=UP, Red=DOWN)', fontsize=14, fontweight='bold', pad=15)
    ax_by_instrument.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=0.5)
    ax_by_instrument.tick_params(axis='both', which='major', labelsize=11)
    
    fig.tight_layout()
    
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
    
    return fig, (ax_overall, ax_by_instrument)


def plot_violin_by_year_and_instrument(df, output_file=None):
    """Create time-series violin plot showing annual trends by instrument."""
    # Add year column
    df_copy = df.copy()
    df_copy['year'] = df_copy['datetime'].dt.year
    
    years = sorted(df_copy['year'].unique())
    instruments = sorted(df_copy['instrument'].unique())
    
    # Create subplots - one per instrument
    n_cols = 2
    n_rows = (len(instruments) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    if n_rows == 1 and n_cols > 1:
        axes = axes.flatten()
    elif n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Hide extra subplots
    for idx in range(len(instruments), len(axes)):
        axes[idx].set_visible(False)
    
    for ax_idx, instrument in enumerate(instruments):
        ax = axes[ax_idx]
        df_inst = df_copy[df_copy['instrument'] == instrument]
        
        if len(df_inst) == 0:
            continue
        
        # Prepare data for each year
        data_by_year = [df_inst[df_inst['year'] == year]['median_stddev_arcsec'].values 
                       for year in years]
        
        # Create violin plot with years on x-axis
        parts = ax.violinplot(data_by_year, positions=range(len(years)), 
                              showmeans=True, showmedians=True, widths=0.7)
        
        # Customize colors
        for pc in parts['bodies']:
            pc.set_facecolor('#2E86AB')
            pc.set_alpha(0.7)
        
        parts['cmedians'].set_edgecolor('red')
        parts['cmedians'].set_linewidth(2)
        parts['cmeans'].set_edgecolor('blue')
        parts['cmeans'].set_linewidth(2)
        
        # Add scatter points
        for year_idx, year in enumerate(years):
            y = df_inst[df_inst['year'] == year]['median_stddev_arcsec'].values
            if len(y) > 0:
                x = np.random.normal(year_idx, 0.04, size=len(y))
                ax.scatter(x, y, alpha=0.3, s=30, color='gray', zorder=2)
        
        ax.set_xticks(range(len(years)))
        ax.set_xticklabels(years, fontsize=10, fontweight='bold')
        ax.set_xlabel('Year', fontsize=11, fontweight='bold')
        ax.set_ylabel('Std Dev (arcsec)', fontsize=11, fontweight='bold')
        ax.set_title(f'{instrument.upper()}: Annual Trends', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=0.5)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='red', linewidth=2, label='Median'),
                       Line2D([0], [0], color='blue', linewidth=2, label='Mean')]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.95)
    
    fig.suptitle('Annual Trends: Pointing Stability by Instrument (Time-Series Violin Plot)', 
                 fontsize=14, fontweight='bold', y=0.995)
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
    
    return fig, axes



def plot_violin_up_down_by_year(all_data, output_file=None):
    """Create time-series violin plot showing annual UP vs DOWN trends."""
    df = extract_up_down_data(all_data)
    
    if len(df) == 0:
        print("  Warning: No UP/DOWN comparison data available for annual violin plot")
        return None, None
    
    # Add year column
    df_copy = df.copy()
    df_copy['year'] = df_copy['datetime'].dt.year
    
    years = sorted(df_copy['year'].unique())
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Prepare data - interleave UP and DOWN for each year
    data_to_plot = []
    positions = []
    colors = []
    pos_counter = 0
    
    for year in years:
        df_year = df_copy[df_copy['year'] == year]
        
        if len(df_year) > 0:
            # UP data
            data_to_plot.append(df_year['mean_stddev_up'].values)
            positions.append(pos_counter)
            colors.append('#2E86AB')
            pos_counter += 0.9
            
            # DOWN data
            data_to_plot.append(df_year['mean_stddev_down'].values)
            positions.append(pos_counter)
            colors.append('#D62728')
            pos_counter += 0.9
            
            pos_counter += 0.2  # Add spacing between years
    
    # Create violin plot with narrower widths
    parts = ax.violinplot(data_to_plot, positions=positions, 
                          showmeans=True, showmedians=True, widths=0.35)
    
    # Customize colors
    for i, pc in enumerate(parts['bodies']):
        if i < len(colors):
            pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    parts['cmedians'].set_edgecolor('black')
    parts['cmedians'].set_linewidth(2)
    parts['cmeans'].set_edgecolor('orange')
    parts['cmeans'].set_linewidth(2)
    
    # Add scatter points
    for i, data in enumerate(data_to_plot):
        x = np.random.normal(positions[i], 0.03, size=len(data))
        ax.scatter(x, data, alpha=0.3, s=20, color='gray', zorder=2)
    
    # Set x-axis labels
    label_positions = []
    label_texts = []
    pos_counter = 0
    for year in years:
        label_positions.append(pos_counter + 0.45)
        label_texts.append(str(year))
        pos_counter += 1.8 + 0.2  # 0.9 + 0.9 + 0.2
    
    ax.set_xticks(label_positions)
    ax.set_xticklabels(label_texts, fontsize=11, fontweight='bold')
    ax.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax.set_ylabel('Standard Deviation of Elevation Error (arcsec)', fontsize=13, fontweight='bold')
    ax.set_title('Annual UP vs DOWN Direction Comparison (Blue=UP, Red=DOWN)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=0.5)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='#2E86AB', linewidth=8, label='UP (Elevation Increase)'),
                       Line2D([0], [0], color='#D62728', linewidth=8, label='DOWN (Elevation Decrease)'),
                       Line2D([0], [0], color='black', linewidth=2, label='Median'),
                       Line2D([0], [0], color='orange', linewidth=2, label='Mean')]
    ax.legend(handles=legend_elements, loc='best', fontsize=11, framealpha=0.95)
    
    ax.tick_params(axis='both', which='major', labelsize=11)
    
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
    
    return fig, ax



def plot_violin_up_down_recent_years(all_data, output_file=None):
    """Create violin plot showing UP vs DOWN for 2024 and 2025 only."""
    df = extract_up_down_data(all_data)
    
    if len(df) == 0:
        print("  Warning: No UP/DOWN comparison data available")
        return None, None
    
    # Add year column
    df_copy = df.copy()
    df_copy['year'] = df_copy['datetime'].dt.year
    
    # Filter to 2024 and 2025
    df_recent = df_copy[df_copy['year'].isin([2024, 2025])]
    
    if len(df_recent) == 0:
        print("  Warning: No data available for 2024 and 2025")
        return None, None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data - separate UP and DOWN for 2024 and 2025
    data_to_plot = []
    positions = []
    colors = []
    labels = []
    
    # 2024 UP and DOWN
    df_2024 = df_recent[df_recent['year'] == 2024]
    if len(df_2024) > 0:
        data_to_plot.append(df_2024['mean_stddev_up'].values)
        positions.append(0)
        colors.append('#2E86AB')
        labels.append('2024 UP')
        
        data_to_plot.append(df_2024['mean_stddev_down'].values)
        positions.append(1)
        colors.append('#D62728')
        labels.append('2024 DOWN')
    
    # 2025 UP and DOWN
    df_2025 = df_recent[df_recent['year'] == 2025]
    if len(df_2025) > 0:
        data_to_plot.append(df_2025['mean_stddev_up'].values)
        positions.append(2.5)
        colors.append('#2E86AB')
        labels.append('2025 UP')
        
        data_to_plot.append(df_2025['mean_stddev_down'].values)
        positions.append(3.5)
        colors.append('#D62728')
        labels.append('2025 DOWN')
    
    if len(data_to_plot) == 0:
        print("  Warning: No valid data for violin plot")
        return None, None
    
    # Create violin plot with generous spacing
    parts = ax.violinplot(data_to_plot, positions=positions, 
                          showmeans=True, showmedians=True, widths=0.5)
    
    # Customize colors
    for i, pc in enumerate(parts['bodies']):
        if i < len(colors):
            pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    parts['cmedians'].set_edgecolor('black')
    parts['cmedians'].set_linewidth(2)
    parts['cmeans'].set_edgecolor('orange')
    parts['cmeans'].set_linewidth(2)
    
    # Add scatter points
    for i, data in enumerate(data_to_plot):
        x = np.random.normal(positions[i], 0.05, size=len(data))
        ax.scatter(x, data, alpha=0.4, s=40, color='gray', zorder=2)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylabel('Standard Deviation of Elevation Error (arcsec)', fontsize=13, fontweight='bold')
    ax.set_title('UP vs DOWN Direction Comparison: 2024 and 2025', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=0.5)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='#2E86AB', linewidth=8, label='UP (Elevation Increase)'),
                       Line2D([0], [0], color='#D62728', linewidth=8, label='DOWN (Elevation Decrease)'),
                       Line2D([0], [0], color='black', linewidth=2, label='Median'),
                       Line2D([0], [0], color='orange', linewidth=2, label='Mean')]
    ax.legend(handles=legend_elements, loc='best', fontsize=11, framealpha=0.95)
    
    ax.tick_params(axis='both', which='major', labelsize=11)
    
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
    
    return fig, ax



def print_up_down_comparison_summary(all_data):
    """Print summary statistics comparing UP vs DOWN directions."""
    df = extract_up_down_data(all_data)
    
    if len(df) == 0:
        print("\n" + "="*80)
        print("UP vs DOWN COMPARISON - NO DATA AVAILABLE")
        print("="*80)
        return
    
    print("\n" + "="*80)
    print("UP vs DOWN DIRECTION COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\nTotal observations: {len(df)}")
    print(f"Date range: {df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')}")
    
    print(f"\n{'Metric':<40} {'UP':<15} {'DOWN':<15} {'Difference':<15}")
    print("-" * 85)
    
    mean_up = df['mean_stddev_up'].mean()
    mean_down = df['mean_stddev_down'].mean()
    mean_diff = mean_up - mean_down
    pct_diff = (mean_diff / mean_down * 100) if mean_down != 0 else 0
    
    print(f"{'Mean Std Dev (arcsec)':<40} {mean_up:<15.6f} {mean_down:<15.6f} {mean_diff:<15.6f}")
    print(f"{'Percent Difference (%)':<40} {'':<15} {'':<15} {pct_diff:<15.2f}")
    
    std_up = df['mean_stddev_up'].std()
    std_down = df['mean_stddev_down'].std()
    print(f"{'Std Dev of measurements':<40} {std_up:<15.6f} {std_down:<15.6f}")
    
    min_up = df['mean_stddev_up'].min()
    min_down = df['mean_stddev_down'].min()
    print(f"{'Min Std Dev (arcsec)':<40} {min_up:<15.6f} {min_down:<15.6f}")
    
    max_up = df['mean_stddev_up'].max()
    max_down = df['mean_stddev_down'].max()
    print(f"{'Max Std Dev (arcsec)':<40} {max_up:<15.6f} {max_down:<15.6f}")
    
    median_up = df['mean_stddev_up'].median()
    median_down = df['mean_stddev_down'].median()
    median_diff = median_up - median_down
    print(f"{'Median Std Dev (arcsec)':<40} {median_up:<15.6f} {median_down:<15.6f} {median_diff:<15.6f}")
    
    # Perform paired t-test
    from scipy import stats as scipy_stats
    t_stat, p_value = scipy_stats.ttest_rel(df['mean_stddev_up'], df['mean_stddev_down'])
    print(f"\nPaired t-test (UP vs DOWN):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    if p_value < 0.05:
        print(f"  ✓ Statistically significant difference (p < 0.05)")
    else:
        print(f"  ✗ No statistically significant difference (p ≥ 0.05)")
    
    # Count cases where UP > DOWN
    count_up_greater = (df['mean_stddev_up'] > df['mean_stddev_down']).sum()
    count_down_greater = (df['mean_stddev_down'] > df['mean_stddev_up']).sum()
    
    print(f"\nDirection comparison:")
    print(f"  UP > DOWN: {count_up_greater} times ({count_up_greater/len(df)*100:.1f}%)")
    print(f"  DOWN > UP: {count_down_greater} times ({count_down_greater/len(df)*100:.1f}%)")
    
    # By instrument
    print(f"\nComparison by Instrument:")
    for instrument in sorted(df['instrument'].unique()):
        df_inst = df[df['instrument'] == instrument]
        if len(df_inst) > 0:
            mean_up_inst = df_inst['mean_stddev_up'].mean()
            mean_down_inst = df_inst['mean_stddev_down'].mean()
            mean_diff_inst = mean_up_inst - mean_down_inst
            pct_diff_inst = (mean_diff_inst / mean_down_inst * 100) if mean_down_inst != 0 else 0
            
            print(f"\n  {instrument.upper()}:")
            print(f"    Observations: {len(df_inst)}")
            print(f"    UP mean:      {mean_up_inst:.6f} arcsec")
            print(f"    DOWN mean:    {mean_down_inst:.6f} arcsec")
            print(f"    Difference:   {mean_diff_inst:.6f} arcsec ({pct_diff_inst:+.2f}%)")
            
            # Paired t-test for this instrument
            if len(df_inst) > 1:
                t_stat_inst, p_value_inst = scipy_stats.ttest_rel(df_inst['mean_stddev_up'], df_inst['mean_stddev_down'])
                print(f"    t-test p-value: {p_value_inst:.6f}", end="")
                print(f" (significant)" if p_value_inst < 0.05 else " (not significant)")
    
    print(f"\n" + "="*85)


def print_prose_summary(df, all_data):
    """Print a comprehensive prose summary of the analysis."""
    print("\n" + "="*80)
    print("ANALYSIS NARRATIVE SUMMARY")
    print("="*80 + "\n")
    
    # Overall summary
    date_range_days = (df['datetime'].max() - df['datetime'].min()).days
    print(f"TRACKING STABILITY ANALYSIS (2019-2026)\n")
    print(f"This analysis examines telescope pointing stability over a {date_range_days}-day period from "
          f"{df['datetime'].min().strftime('%B %d, %Y')} to {df['datetime'].max().strftime('%B %d, %Y')}. "
          f"The dataset comprises {len(df)} observation periods across {df['instrument'].nunique()} different instruments. "
          f"The primary metric used is the standard deviation of elevation error, which quantifies the variability in "
          f"telescope pointing precision.\n")
    
    # Overall performance
    mean_stddev = df['median_stddev_arcsec'].mean()
    median_stddev = df['median_stddev_arcsec'].median()
    std_stddev = df['median_stddev_arcsec'].std()
    min_stddev = df['median_stddev_arcsec'].min()
    max_stddev = df['median_stddev_arcsec'].max()
    
    min_idx = df['median_stddev_arcsec'].idxmin()
    max_idx = df['median_stddev_arcsec'].idxmax()
    min_date = df.loc[min_idx, 'datetime'].strftime('%B %d, %Y')
    min_instrument = df.loc[min_idx, 'instrument'].upper()
    max_date = df.loc[max_idx, 'datetime'].strftime('%B %d, %Y')
    max_instrument = df.loc[max_idx, 'instrument'].upper()
    
    print(f"OVERALL POINTING STABILITY:\n")
    print(f"The telescope demonstrated a mean standard deviation of elevation error of {mean_stddev:.6f} arcseconds, "
          f"with a median of {median_stddev:.6f} arcseconds. The performance varied considerably across the observing period, "
          f"ranging from a best case of {min_stddev:.6f} arcseconds on {min_date} ({min_instrument} instrument) "
          f"to a worst case of {max_stddev:.6f} arcseconds on {max_date} ({max_instrument} instrument). "
          f"The standard deviation of these measurements was {std_stddev:.6f} arcseconds, indicating significant variability "
          f"in tracking performance across different observing conditions and time periods.\n")
    
    # Q1, Q3 analysis
    q1 = df['median_stddev_arcsec'].quantile(0.25)
    q3 = df['median_stddev_arcsec'].quantile(0.75)
    iqr = q3 - q1
    
    print(f"In terms of distribution, 25% of observations achieved pointing stability better than {q1:.6f} arcseconds, "
          f"while 75% were better than {q3:.6f} arcseconds. The interquartile range of {iqr:.6f} arcseconds reflects "
          f"the typical variability in day-to-day pointing performance.\n")
    
    # By instrument
    print(f"PERFORMANCE BY INSTRUMENT:\n")
    instruments_summary = []
    for instrument in sorted(df['instrument'].unique()):
        df_inst = df[df['instrument'] == instrument]
        mean_inst = df_inst['median_stddev_arcsec'].mean()
        count_inst = len(df_inst)
        instruments_summary.append(f"  • {instrument.upper()}: {count_inst} observations with mean stability of {mean_inst:.6f} arcseconds")
    
    for line in instruments_summary:
        print(line)
    
    print()
    
    # UP vs DOWN analysis
    df_updown = extract_up_down_data(all_data)
    
    if len(df_updown) > 0:
        print(f"UP vs DOWN DIRECTIONAL ANALYSIS:\n")
        
        mean_up = df_updown['mean_stddev_up'].mean()
        mean_down = df_updown['mean_stddev_down'].mean()
        mean_diff = mean_up - mean_down
        pct_diff = (mean_diff / mean_down * 100) if mean_down != 0 else 0
        
        print(f"The analysis reveals differences in pointing stability depending on the direction of telescope motion. "
              f"When moving in the UP direction (increasing elevation), the mean standard deviation of elevation error "
              f"was {mean_up:.6f} arcseconds. In contrast, when moving in the DOWN direction (decreasing elevation), "
              f"the mean was {mean_down:.6f} arcseconds. This represents a difference of {mean_diff:.6f} arcseconds, "
              f"or {pct_diff:+.2f}% ", end="")
        
        if pct_diff > 0:
            print(f"higher pointing variability in the UP direction.\n")
        elif pct_diff < 0:
            print(f"lower pointing variability in the UP direction.\n")
        else:
            print(f"with equivalent performance in both directions.\n")
        
        # Statistical test
        from scipy import stats as scipy_stats
        t_stat, p_value = scipy_stats.ttest_rel(df_updown['mean_stddev_up'], df_updown['mean_stddev_down'])
        
        if p_value < 0.05:
            print(f"A paired t-test yields a p-value of {p_value:.6f}, which is statistically significant at the 0.05 level. "
                  f"This indicates that the observed difference between UP and DOWN directions is unlikely to be due to random chance ")
            if pct_diff > 0:
                print(f"and suggests a systematic tendency for the telescope to experience higher pointing variability when "
                      f"moving to higher elevations.\n")
            else:
                print(f"and suggests a systematic tendency for the telescope to experience lower pointing variability when "
                      f"moving to higher elevations.\n")
        else:
            print(f"A paired t-test yields a p-value of {p_value:.6f}, which does not meet the 0.05 significance threshold. "
                  f"This suggests that while there is a numerical difference between UP and DOWN performance, it may not be "
                  f"statistically significant and could be attributable to natural variation in observing conditions.\n")
        
        # Count summary
        count_up_greater = (df_updown['mean_stddev_up'] > df_updown['mean_stddev_down']).sum()
        count_down_greater = (df_updown['mean_stddev_down'] > df_updown['mean_stddev_up']).sum()
        
        print(f"Examining individual observation periods, the UP direction exhibited higher variability in {count_up_greater} cases "
              f"({count_up_greater/len(df_updown)*100:.1f}%), while the DOWN direction showed higher variability in {count_down_greater} cases "
              f"({count_down_greater/len(df_updown)*100:.1f}%). This distribution indicates a consistent but not universal asymmetry "
              f"in pointing performance between the two elevation directions.\n")
        
        # By instrument for UP/DOWN
        print(f"By instrument, the UP vs DOWN differences are:\n")
        for instrument in sorted(df_updown['instrument'].unique()):
            df_inst = df_updown[df_updown['instrument'] == instrument]
            if len(df_inst) > 0:
                mean_up_inst = df_inst['mean_stddev_up'].mean()
                mean_down_inst = df_inst['mean_stddev_down'].mean()
                mean_diff_inst = mean_up_inst - mean_down_inst
                pct_diff_inst = (mean_diff_inst / mean_down_inst * 100) if mean_down_inst != 0 else 0
                direction = "higher in UP" if mean_diff_inst > 0 else "higher in DOWN"
                print(f"  • {instrument.upper()}: {mean_diff_inst:+.6f} arcseconds ({pct_diff_inst:+.2f}%) — variability is {direction}")
        
        print()
    
    print("="*80 + "\n")


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
    
    # Print UP vs DOWN comparison
    print_up_down_comparison_summary(all_data)
    
    # Print prose narrative summary
    print_prose_summary(df, all_data)
    
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
    
    print("\nCreating violin plot by instrument...")
    plot_violin_by_instrument(df, output_file='stddev_violin_by_instrument_2019_2026.png')
    
    print("\nCreating UP vs DOWN violin plot...")
    plot_violin_up_down(all_data, output_file='stddev_violin_up_vs_down_2019_2026.png')
    
    print("\nCreating annual violin plot by instrument...")
    plot_violin_by_year_and_instrument(df, output_file='stddev_violin_by_year_and_instrument_2019_2026.png')
    
    print("\nCreating annual UP vs DOWN violin plot...")
    plot_violin_up_down_by_year(all_data, output_file='stddev_violin_up_vs_down_by_year_2019_2026.png')
    
    print("\nCreating recent years (2024-2025) UP vs DOWN violin plot...")
    plot_violin_up_down_recent_years(all_data, output_file='stddev_violin_up_vs_down_2024_2025.png')
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print("\n📊 Generated files:")
    print("  • stddev_timeseries_2019_2026.png")
    print("  • stddev_by_instrument_2019_2026.png")
    print("  • stddev_combined_comparison_2019_2026.png")
    print("  • stddev_up_vs_down_2019_2026.png")
    print("  • stddev_violin_by_instrument_2019_2026.png")
    print("  • stddev_violin_up_vs_down_2019_2026.png")
    print("  • stddev_violin_by_year_and_instrument_2019_2026.png")
    print("  • stddev_violin_up_vs_down_by_year_2019_2026.png")
    print("  • stddev_violin_up_vs_down_2024_2025.png")
    print("\n📈 Plots displayed in the VS Code interactive window above.\n")


if __name__ == '__main__':
    main()
