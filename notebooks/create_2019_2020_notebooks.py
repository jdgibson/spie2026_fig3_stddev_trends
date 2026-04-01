#!/usr/bin/env python3
"""
Create monthly Telescope Elevation Error analysis notebooks for remainder of 2019 and all of 2020.
Each notebook analyzes data for one month.
"""

import json
from pathlib import Path
from datetime import datetime
import calendar

def get_month_dates(year, month):
    """Get start and end dates for a given month."""
    start_date = datetime(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    end_date = datetime(year, month, last_day, 23, 59, 59)
    
    return (
        start_date.strftime('%Y-%m-%d 00:00:00'),
        end_date.strftime('%Y-%m-%d 23:59:59')
    )

def create_monthly_notebooks():
    """Create notebooks for all months from Nov 2019 through Dec 2020."""
    
    # Load template from existing notebook
    template_path = Path('TelescopeElevationError_201901.ipynb')
    if not template_path.exists():
        print("✗ Template notebook not found: TelescopeElevationError_201901.ipynb")
        return False
    
    with open(template_path, 'r') as f:
        template = json.load(f)
    
    # Define all months to create: Nov-Dec 2019 + all of 2020
    months_to_create = []
    
    # Add remaining months of 2019
    for month in range(11, 13):  # Nov, Dec
        months_to_create.append((2019, month))
    
    # Add all months of 2020
    for month in range(1, 13):  # Jan - Dec
        months_to_create.append((2020, month))
    
    print(f"Creating {len(months_to_create)} new notebooks...\n")
    
    created_count = 0
    skipped_count = 0
    error_count = 0
    
    for year, month in months_to_create:
        # Create filename
        filename = f"TelescopeElevationError_{year}{month:02d}.ipynb"
        filepath = Path(filename)
        
        # Check if notebook already exists and has content
        if filepath.exists():
            file_size = filepath.stat().st_size
            if file_size > 100:  # Has meaningful content
                print(f"⊘ {filename}: Already exists (skipped)")
                skipped_count += 1
                continue
            # File exists but is empty, so we'll populate it
        
        try:
            # Create a deep copy of the template
            nb_copy = json.loads(json.dumps(template))
            
            # Get correct date range for this month
            start_dt, end_dt = get_month_dates(year, month)
            
            # Find and update the parameters cell
            updated = False
            for cell in nb_copy['cells']:
                if cell['cell_type'] == 'code':
                    source = ''.join(cell.get('source', []))
                    
                    # Check if this is the parameters cell
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
            
            if not updated:
                print(f"⚠ {filename}: Could not find parameters cell to update")
                error_count += 1
                continue
            
            # Write the new notebook
            with open(filepath, 'w') as f:
                json.dump(nb_copy, f, indent=1)
            
            action = "Updated" if filepath.exists() and filepath.stat().st_size < 200 else "Created"
            print(f"✓ {action} {filename} ({year}-{month:02d}: {start_dt[:10]} to {end_dt[:10]})")
            created_count += 1
            
        except Exception as e:
            print(f"✗ {filename}: Error - {e}")
            error_count += 1
    
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  ✓ Created: {created_count}")
    print(f"  ⊘ Skipped: {skipped_count}")
    print(f"  ✗ Errors:  {error_count}")
    print(f"  Total:     {len(months_to_create)}")
    print(f"{'='*80}")
    
    return created_count > 0

if __name__ == '__main__':
    import sys
    success = create_monthly_notebooks()
    sys.exit(0 if success else 1)
