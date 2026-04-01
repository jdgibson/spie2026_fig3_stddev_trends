#!/usr/bin/env python3
"""Populate empty notebooks with proper template structure."""

import json
from pathlib import Path

# Load the template from 201901
with open('TelescopeElevationError_201901.ipynb', 'r') as f:
    template = json.load(f)

# Define notebooks to create with their date ranges
notebooks_to_create = [
    ('TelescopeElevationError_201903.ipynb', '2019-03-01 00:00:00', '2019-03-31 23:59:59'),
    ('TelescopeElevationError_201904.ipynb', '2019-04-01 00:00:00', '2019-04-30 23:59:59'),
    ('TelescopeElevationError_201905.ipynb', '2019-05-01 00:00:00', '2019-05-31 23:59:59'),
    ('TelescopeElevationError_201906.ipynb', '2019-06-01 00:00:00', '2019-06-30 23:59:59'),
    ('TelescopeElevationError_201907.ipynb', '2019-07-01 00:00:00', '2019-07-31 23:59:59'),
    ('TelescopeElevationError_201908.ipynb', '2019-08-01 00:00:00', '2019-08-31 23:59:59'),
    ('TelescopeElevationError_201909.ipynb', '2019-09-01 00:00:00', '2019-09-30 23:59:59'),
    ('TelescopeElevationError_201910.ipynb', '2019-10-01 00:00:00', '2019-10-31 23:59:59'),
]

for filename, start_date, end_date in notebooks_to_create:
    # Create a copy of the template
    nb_copy = json.loads(json.dumps(template))
    
    # Update the date range in the parameters cell (should be index 3)
    if len(nb_copy['cells']) > 3 and nb_copy['cells'][3]['cell_type'] == 'code':
        source_lines = nb_copy['cells'][3]['source']
        new_lines = []
        for line in source_lines:
            if "start_datetime = '2019-02-01" in line:
                new_lines.append(f"start_datetime = '{start_date}'\n")
            elif "end_datetime = '2019-02-28" in line:
                new_lines.append(f"end_datetime = '{end_date}'\n")
            else:
                new_lines.append(line)
        nb_copy['cells'][3]['source'] = new_lines
    
    # Write the notebook
    with open(filename, 'w') as f:
        json.dump(nb_copy, f, indent=1)
    
    print(f"✓ Created {filename}")

print("\n✓ All 8 empty notebooks populated with proper structure")
