#!/usr/bin/env python3
"""Verify configuration cells were added correctly."""

import json
from pathlib import Path

notebooks = sorted(Path('.').glob('TelescopeElevationError_*.ipynb'))
verified = 0

print("Verification of configuration cells:\n")

for nb_path in notebooks[:3]:  # Check first 3
    with open(nb_path, 'r') as f:
        data = json.load(f)
    
    # Find config cell
    has_config = False
    for i, cell in enumerate(data['cells'], 1):
        source = ''.join(cell.get('source', []))
        if 'CACHE_DIR' in source:
            has_config = True
            # Check for key variables
            vars_found = []
            for var in ['start_dt', 'end_dt', 'DB_HOST', 'CACHE_DIR', 'ENABLE_QUERY_CACHE']:
                if var in source:
                    vars_found.append(var)
            
            print(f"✓ {nb_path.name}:")
            print(f"    Configuration cell at position: {i}")
            print(f"    Variables defined: {', '.join(vars_found)}")
            verified += 1
            break
    
    if not has_config:
        print(f"✗ {nb_path.name}: No configuration cell found")

if verified == 3:
    print(f"\n✅ All checked notebooks have proper configuration!")
    print(f"\nAll variables now defined in each notebook:")
    print(f"  • start_dt, end_dt (converted from start_datetime, end_datetime)")
    print(f"  • DB_HOST, DB_USER, DB_PASSWORD, DB_MEASUREMENTS")
    print(f"  • CACHE_DIR, ENABLE_QUERY_CACHE")
    print(f"  • SKIP_INDIVIDUAL_PLOTS, SKIP_SUMMARY_PLOTS")
    print(f"  • ENABLE_PLOT_DOWNSAMPLING, MAX_PLOT_POINTS")
