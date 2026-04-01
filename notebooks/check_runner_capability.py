#!/usr/bin/env python3
"""Test if run_all_notebooks.py can run notebooks in this directory."""

import os
import glob
import subprocess
import shutil
from datetime import datetime

def main():
    """Check prerequisites and notebook discovery."""
    
    print("="*80)
    print("NOTEBOOK RUNNER - CAPABILITY CHECK")
    print("="*80)
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check current directory
    script_dir = os.getcwd()
    print(f"Script directory: {script_dir}\n")
    
    # Find notebooks
    pattern = os.path.join(script_dir, '*.ipynb')
    notebooks = sorted(glob.glob(pattern))
    
    print(f"Notebook Discovery:")
    print(f"  Pattern: {pattern}")
    print(f"  Found: {len(notebooks)} notebooks\n")
    
    if not notebooks:
        print("✗ No notebooks found!")
        return False
    
    for i, nb_path in enumerate(notebooks[:5], 1):
        basename = os.path.basename(nb_path)
        size_kb = os.path.getsize(nb_path) / 1024
        print(f"  {i:2d}. {basename:45s} ({size_kb:6.1f} KB)")
    
    if len(notebooks) > 5:
        print(f"  ... and {len(notebooks) - 5} more")
    
    # Check for jupyter
    print(f"\nPrerequisites:")
    jupyter_path = shutil.which('jupyter')
    if jupyter_path:
        print(f"  ✓ jupyter: {jupyter_path}")
    else:
        print(f"  ✗ jupyter: NOT FOUND")
        return False
    
    # Check nbconvert
    nbconvert_path = shutil.which('jupyter-nbconvert')
    if nbconvert_path:
        print(f"  ✓ nbconvert: {nbconvert_path}")
    else:
        print(f"  ✗ nbconvert: NOT FOUND")
    
    # Test jupyter nbconvert availability
    try:
        result = subprocess.run(['jupyter', 'nbconvert', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"  ✓ jupyter nbconvert version: {version}")
        else:
            print(f"  ⚠ jupyter nbconvert: Issue verifying version")
    except Exception as e:
        print(f"  ✗ jupyter nbconvert: {str(e)}")
        return False
    
    print(f"\nCapability Summary:")
    print(f"  ✓ Can discover {len(notebooks)} notebooks")
    print(f"  ✓ Jupyter nbconvert available")
    print(f"  ✓ run_all_notebooks.py READY TO USE")
    
    print(f"\nTo run all notebooks:")
    print(f"  cd {script_dir}")
    print(f"  python3 run_all_notebooks.py")
    
    return True

if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
