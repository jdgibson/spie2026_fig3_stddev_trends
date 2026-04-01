#!/usr/bin/env python3
"""Fix common errors in all TelescopeElevationError notebooks."""

import json
from pathlib import Path

def fix_notebook_errors(filepath):
    """Fix import errors and __file__ references in a notebook."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    fixed = False
    
    # Fix first code cell
    if data['cells'] and data['cells'][0]['cell_type'] == 'code':
        source = data['cells'][0]['source']
        new_source = []
        
        for line in source:
            # Fix matplotlib typo
            if 'matplotlib.pypalot' in line:
                new_source.append(line.replace('matplotlib.pypalot', 'matplotlib.pyplot'))
                fixed = True
            # Fix __file__ reference
            elif 'os.path.dirname(os.path.abspath(__file__))' in line:
                new_source.append(line.replace(
                    'os.path.dirname(os.path.abspath(__file__))',
                    'os.getcwd()'
                ))
                fixed = True
            else:
                new_source.append(line)
        
        data['cells'][0]['source'] = new_source
    
    return data, fixed

notebooks = sorted(Path('.').glob('TelescopeElevationError_*.ipynb'))
fixed_count = 0

for nb_path in notebooks:
    try:
        data, fixed = fix_notebook_errors(nb_path)
        
        if fixed:
            with open(nb_path, 'w') as f:
                json.dump(data, f, indent=1)
            print(f"✓ {nb_path.name}: Fixed errors")
            fixed_count += 1
        else:
            print(f"- {nb_path.name}: No errors found")
    except Exception as e:
        print(f"✗ {nb_path.name}: Error - {e}")

print(f"\n✅ Total fixed: {fixed_count}/{len(notebooks)}")
