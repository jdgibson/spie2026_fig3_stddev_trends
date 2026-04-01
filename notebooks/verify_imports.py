#!/usr/bin/env python3
"""Verify all notebooks have correct imports and setup."""

import json
from pathlib import Path

notebooks = sorted(Path('.').glob('TelescopeElevationError_*.ipynb'))
all_ok = True

print("Verifying all notebooks have correct imports:\n")

for nb_path in notebooks:
    with open(nb_path, 'r') as f:
        data = json.load(f)
    
    if data['cells'] and data['cells'][0]['cell_type'] == 'code':
        source_text = ''.join(data['cells'][0]['source'])
        
        issues = []
        
        if 'matplotlib.pypalot' in source_text:
            issues.append("pypalot typo")
        elif 'matplotlib.pyplot' not in source_text:
            issues.append("missing pyplot import")
        
        if '__file__' in source_text:
            issues.append("__file__ reference")
        elif 'os.getcwd()' not in source_text:
            issues.append("missing os.getcwd()")
        
        if issues:
            print(f"✗ {nb_path.name}: {', '.join(issues)}")
            all_ok = False
        else:
            print(f"✓ {nb_path.name}: OK")

if all_ok:
    print("\n✅ All 24 notebooks have correct imports and setup!")
else:
    print("\n⚠️ Some notebooks need fixing")
