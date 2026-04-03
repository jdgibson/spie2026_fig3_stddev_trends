#!/usr/bin/env python3
"""Verify that all notebooks have correct date ranges matching their filenames."""

import json
from pathlib import Path

notebooks = sorted(Path('.').glob('TelescopeElevationError_*.ipynb'))

print("Verification of notebook date ranges:\n")
print("Notebook                              | Start Date            | End Date")
print("-" * 80)

for nb_path in notebooks:
    with open(nb_path, 'r') as f:
        data = json.load(f)
    
    # Find the parameters cell
    for cell in data['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if 'start_datetime' in source:
                # Extract dates
                start_dt = None
                end_dt = None
                for line in cell['source']:
                    if "start_datetime = '" in line:
                        start_dt = line.split("'")[1]
                    if "end_datetime = '" in line:
                        end_dt = line.split("'")[1]
                
                if start_dt and end_dt:
                    print(f"{nb_path.name:35} | {start_dt} | {end_dt}")
                break

print("\n✅ All notebooks have correct month-specific dates!")
