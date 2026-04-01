#!/usr/bin/env python3
"""Update all notebooks to have correct start_datetime and end_datetime based on filename month."""

import json
from pathlib import Path
from datetime import datetime, timedelta
import calendar

def get_month_dates(year, month):
    """Get start and end dates for a given month."""
    start_date = datetime(year, month, 1)
    # Get the last day of the month
    last_day = calendar.monthrange(year, month)[1]
    end_date = datetime(year, month, last_day, 23, 59, 59)
    
    return (
        start_date.strftime('%Y-%m-%d 00:00:00'),
        end_date.strftime('%Y-%m-%d 23:59:59')
    )

def extract_month_from_filename(filename):
    """Extract year and month from filename like TelescopeElevationError_201901.ipynb"""
    # Extract the date part: 201901
    date_str = filename.replace('TelescopeElevationError_', '').replace('.ipynb', '')
    if len(date_str) == 6:
        year = int(date_str[:4])
        month = int(date_str[4:6])
        return year, month
    return None, None

notebooks = sorted(Path('.').glob('TelescopeElevationError_*.ipynb'))
updated_count = 0

for nb_path in notebooks:
    year, month = extract_month_from_filename(nb_path.name)
    
    if year is None or month is None:
        print(f"✗ {nb_path.name}: Could not extract date from filename")
        continue
    
    try:
        with open(nb_path, 'r') as f:
            data = json.load(f)
        
        # Get the correct dates
        start_dt, end_dt = get_month_dates(year, month)
        
        # Find and update the parameters cell (usually cell 3 in our notebooks)
        updated = False
        for cell in data['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell.get('source', []))
                
                # Check if this is the parameters cell with start_datetime
                if 'start_datetime' in source and 'end_datetime' in source:
                    # Update the source lines
                    new_source = []
                    for line in cell['source']:
                        if line.startswith("start_datetime = "):
                            new_source.append(f"start_datetime = '{start_dt}'\n")
                        elif line.startswith("end_datetime = "):
                            new_source.append(f"end_datetime = '{end_dt}'\n")
                        else:
                            new_source.append(line)
                    
                    cell['source'] = new_source
                    updated = True
                    break
        
        if updated:
            with open(nb_path, 'w') as f:
                json.dump(data, f, indent=1)
            
            print(f"✓ {nb_path.name}: Updated to {year}-{month:02d} ({start_dt} to {end_dt})")
            updated_count += 1
        else:
            print(f"⚠ {nb_path.name}: Could not find parameters cell to update")
            
    except Exception as e:
        print(f"✗ {nb_path.name}: Error - {e}")

print(f"\n✅ Total updated: {updated_count}/{len(notebooks)}")
