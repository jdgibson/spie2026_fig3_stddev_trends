#!/usr/bin/env python3
"""Verify all notebooks have proper VSCode format."""

import json
from pathlib import Path

notebooks = sorted(Path('.').glob('TelescopeElevationError*.ipynb'))
verified = 0

for nb_path in notebooks:
    try:
        with open(nb_path, 'r') as f:
            data = json.load(f)
        
        # Check if ANY cell has analyze_telescope_elevation_error import
        has_imports = False
        for cell in data['cells'][:5]:  # Check first 5 cells
            source = ''.join(cell.get('source', []))
            if 'analyze_telescope_elevation_error' in source:
                has_imports = True
                break
        
        # Check for path setup in first cell or second cell
        has_path_setup = False
        for cell in data['cells'][:3]:  # Check first 3 cells
            source = ''.join(cell.get('source', []))
            if 'os.getcwd()' in source or 'sys.path' in source:
                has_path_setup = True
                break
        
        if has_path_setup and has_imports:
            print(f'✓ {nb_path.name}: Properly formatted')
            verified += 1
        else:
            print(f'⚠ {nb_path.name}: Missing setup={not has_path_setup}, imports={not has_imports}')
    except Exception as e:
        print(f'✗ {nb_path.name}: Error - {e}')

print(f'\n✅ Total verified: {verified}/{len(notebooks)}')
