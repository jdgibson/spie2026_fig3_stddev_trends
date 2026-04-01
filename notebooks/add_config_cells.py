#!/usr/bin/env python3
"""Add missing configuration variables to all notebooks."""

import json
from pathlib import Path

def add_config_cell(data):
    """Add a configuration cell after the parameters cell."""
    
    # Configuration cell content
    config_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Configuration Variables\n",
            "import os\n",
            "\n",
            "# Convert start_datetime and end_datetime to start_dt and end_dt for compatibility\n",
            "start_dt = start_datetime\n",
            "end_dt = end_datetime\n",
            "\n",
            "# Database configuration\n",
            "DB_HOST = 'mariadb.mmto.arizona.edu'\n",
            "DB_USER = 'mmtstaff'\n",
            "DB_PASSWORD = 'multiple'\n",
            "DB_MEASUREMENTS = 'measurements'\n",
            "\n",
            "# Cache configuration\n",
            "CACHE_DIR = os.path.join(os.getcwd(), '.notebook_cache')\n",
            "ENABLE_QUERY_CACHE = False  # Set to True to use cached results\n",
            "\n",
            "# Performance settings\n",
            "SKIP_INDIVIDUAL_PLOTS = False\n",
            "SKIP_SUMMARY_PLOTS = False\n",
            "ENABLE_PLOT_DOWNSAMPLING = True\n",
            "MAX_PLOT_POINTS = 10000\n",
            "\n",
            "# Create cache directory if needed\n",
            "os.makedirs(CACHE_DIR, exist_ok=True)\n",
            "\n",
            "print(f\"✓ Configuration loaded\")\n",
            "print(f\"  Analysis period: {start_dt} to {end_dt}\")\n",
            "print(f\"  Database: {DB_HOST}\")\n",
            "print(f\"  Cache directory: {CACHE_DIR}\")"
        ]
    }
    
    # Find where to insert the config cell
    # It should go after the parameters cell (usually index 3)
    # Look for the cell that has start_datetime definition
    
    insert_index = None
    for i, cell in enumerate(data['cells']):
        if cell['cell_type'] == 'code':
            source_text = ''.join(cell.get('source', []))
            if 'start_datetime = ' in source_text and insert_index is None:
                # This is the parameters cell, insert after it
                insert_index = i + 1
                break
    
    if insert_index is None:
        # If we can't find parameters cell, insert after first markdown cell
        for i, cell in enumerate(data['cells']):
            if cell['cell_type'] == 'markdown':
                insert_index = i + 1
                break
    
    if insert_index is None:
        insert_index = 1
    
    # Check if config cell already exists
    for i, cell in enumerate(data['cells'][insert_index:insert_index+2], insert_index):
        source_text = ''.join(cell.get('source', []))
        if 'CACHE_DIR' in source_text and 'DB_HOST' in source_text:
            # Config already exists
            return data, False
    
    # Insert the config cell
    data['cells'].insert(insert_index, config_cell)
    return data, True

def main():
    notebooks = sorted(Path('.').glob('TelescopeElevationError_*.ipynb'))
    added_count = 0
    skipped_count = 0
    error_count = 0
    
    print("Adding configuration cells to all notebooks...\n")
    
    for nb_path in notebooks:
        try:
            with open(nb_path, 'r') as f:
                data = json.load(f)
            
            data, added = add_config_cell(data)
            
            if added:
                with open(nb_path, 'w') as f:
                    json.dump(data, f, indent=1)
                print(f"✓ {nb_path.name}: Added configuration cell")
                added_count += 1
            else:
                print(f"⊘ {nb_path.name}: Configuration already exists")
                skipped_count += 1
                
        except Exception as e:
            print(f"✗ {nb_path.name}: Error - {e}")
            error_count += 1
    
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  ✓ Added:   {added_count}")
    print(f"  ⊘ Skipped: {skipped_count}")
    print(f"  ✗ Errors:  {error_count}")
    print(f"  Total:     {len(notebooks)}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
