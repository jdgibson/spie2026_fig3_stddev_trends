#!/usr/bin/env python3
"""
Script to execute all notebooks in the current directory and save outputs.
Shows progress during processing and handles errors gracefully.
Uses jupyter nbconvert with --execute --inplace to properly save all outputs.
Supports parallel execution for faster processing.
"""

import os
import glob
import sys
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Lock for thread-safe console output
console_lock = threading.Lock()

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def print_progress(current, total, filename):
    """Print progress information (thread-safe)."""
    with console_lock:
        status = f"[{current}/{total}] Processing: {filename}"
        print(f"\n{status}")
        print("-" * 80)

def run_notebook(notebook_path, timeout=3600, clear_outputs=False):
    """
    Execute a notebook using jupyter nbconvert and save the output to the same file.
    
    This method properly captures all output including text, figures, and display outputs.
    Supports concurrent execution with thread-safe output.
    
    Parameters:
    -----------
    notebook_path : str
        Path to the notebook file
    timeout : int
        Timeout in seconds for notebook execution
    clear_outputs : bool
        Whether to clear previous outputs before execution
        
    Returns:
    --------
    tuple: (notebook_name, success: bool, elapsed_time: float)
        Returns notebook name, success status, and execution time
    """
    notebook_name = os.path.basename(notebook_path)
    start_time = datetime.now()
    
    try:
        with console_lock:
            print(f"  Starting execution of {notebook_name}...")
            print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Optionally clear outputs first
        if clear_outputs:
            try:
                import nbformat
                with open(notebook_path, 'r') as f:
                    nb = nbformat.read(f, as_version=4)
                
                # Clear outputs from all cells
                for cell in nb.cells:
                    if cell.cell_type == 'code':
                        cell.outputs = []
                        cell.execution_count = None
                
                with open(notebook_path, 'w') as f:
                    nbformat.write(nb, f)
                print(f"  Cleared previous outputs")
            except Exception as clear_error:
                print(f"  Warning: Could not clear outputs: {str(clear_error)}")
        
        # Use jupyter nbconvert to execute notebook with --inplace
        cmd = [
            'jupyter', 'nbconvert',
            '--to', 'notebook',
            '--execute',
            '--inplace',
            f'--ExecutePreprocessor.timeout={timeout}',
            notebook_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout+60)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        with console_lock:
            if result.returncode == 0:
                print(f"  ✓ Successfully executed {notebook_name} ({elapsed:.1f}s)")
                return (notebook_name, True, elapsed)
            else:
                print(f"  ✗ Execution failed with return code {result.returncode}")
                if result.stderr:
                    print(f"  Error output: {result.stderr[:500]}")
                return (notebook_name, False, elapsed)
            
    except subprocess.TimeoutExpired:
        elapsed = (datetime.now() - start_time).total_seconds()
        with console_lock:
            print(f"  ✗ Execution timeout for {notebook_name} ({elapsed:.1f}s)")
        return (notebook_name, False, elapsed)
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        with console_lock:
            print(f"  ✗ Error processing {notebook_name}: {str(e)}")
            traceback.print_exc()
        return (notebook_name, False, elapsed)

def main():
    """Main function to find and execute all notebooks in the current directory."""
    
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print_header("NOTEBOOK RUNNER")
    print(f"Script directory: {script_dir}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find all notebooks
    pattern = os.path.join(script_dir, '*.ipynb')
    notebooks = sorted(glob.glob(pattern))
    
    if not notebooks:
        print("\n✗ No notebooks found in the directory!")
        return 1
    
    print(f"\nFound {len(notebooks)} notebook(s) to process:")
    for i, nb in enumerate(notebooks, 1):
        print(f"  {i}. {os.path.basename(nb)}")
    
    # Execute each notebook in parallel
    print_header("EXECUTING NOTEBOOKS (PARALLEL)")
    
    successful = 0
    failed = 0
    failures = []
    total_time = 0
    
    # Use ThreadPoolExecutor for parallel execution (default: 4 workers)
    # Adjust max_workers based on your system CPU count
    num_workers = min(4, len(notebooks))  # Don't spawn more threads than notebooks
    print(f"Using {num_workers} parallel worker(s)\n")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all notebooks for execution
        futures = {executor.submit(run_notebook, nb, timeout=7200, clear_outputs=False): nb 
                   for nb in notebooks}
        
        # Process results as they complete
        completed = 0
        for future in as_completed(futures):
            completed += 1
            notebook_name, success, elapsed = future.result()
            print_progress(completed, len(notebooks), notebook_name)
            
            total_time += elapsed
            if success:
                successful += 1
            else:
                failed += 1
                failures.append(notebook_name)
    
    # Print summary
    print_header("EXECUTION SUMMARY")
    print(f"Total notebooks processed: {len(notebooks)}")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    print(f"\nParallel Processing Stats:")
    print(f"  Workers used: {num_workers}")
    print(f"  Total elapsed time: {total_time:.1f}s")
    print(f"  Average per notebook: {total_time/len(notebooks):.1f}s")
    print(f"  Estimated serial time: {total_time * num_workers / len(notebooks):.1f}s")
    if total_time > 0:
        speedup = (total_time * num_workers / len(notebooks)) / total_time
        print(f"  Parallel speedup: {speedup:.1f}x")
    
    if failures:
        print("\nFailed notebooks:")
        for failure in failures:
            print(f"  - {failure}")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    return 0 if failed == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
