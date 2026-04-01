#!/usr/bin/env python3
"""Fix __file__ references in all notebooks to use os.getcwd() instead."""

import json
from pathlib import Path

notebooks = sorted(Path('.').glob('TelescopeElevationError*.ipynb'))
fixed_count = 0
error_count = 0
already_fixed = 0

for nb_path in notebooks:
    try:
        with open(nb_path, 'r') as f:
            data = json.load(f)
        
        # Check if first cell has the problematic __file__ code
        if data['cells'] and data['cells'][0]['cell_type'] == 'code':
            source = ''.join(data['cells'][0]['source'])
            if '__file__' in source:
                # Replace __file__ with os.getcwd()
                new_source = source.replace(
                    'notebook_dir = os.path.dirname(os.path.abspath(__file__))',
                    'notebook_dir = os.getcwd()'
                )
                data['cells'][0]['source'] = new_source.split('\n')
                
                # Write back
                with open(nb_path, 'w') as f:
                    json.dump(data, f, indent=1)
                print(f"✓ {nb_path.name}: Fixed __file__ reference")
                fixed_count += 1
            elif 'os.getcwd()' in source:
                print(f"✓ {nb_path.name}: Already using os.getcwd()")
                already_fixed += 1
            else:
                print(f"- {nb_path.name}: No path setup found")
        else:
            print(f"- {nb_path.name}: First cell is not code")
            
    except Exception as e:
        print(f"✗ {nb_path.name}: Error - {e}")
        error_count += 1

print(f"\nSummary: {fixed_count} fixed, {already_fixed} already fixed, {error_count} errors")
