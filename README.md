# Telescope Elevation Error Analysis

This project contains a series of Jupyter notebooks designed to analyze telescope elevation error data over a range of months from February 2019 through February 2026. Each notebook focuses on a specific month and includes parameters for data filtering and performance optimization.

## Project Structure

- **notebooks/**: Contains individual notebooks for each month of analysis.
  - `TelescopeElevationError_YYYYMM.ipynb`: Each notebook analyzes telescope elevation error data for the specified month and year.
  
- **scripts/**: Contains Python scripts for automating tasks.
  - `generate_monthly_notebooks.py`: Script to automate the creation of monthly notebooks.
  - `batch_execute_notebooks.py`: Script to execute all notebooks in batch mode.

- **templates/**: Contains template files for creating new notebooks.
  - `telescope_elevation_template.ipynb`: Template notebook used for creating monthly notebooks.

- **results/**: Stores output files generated from the analysis.
  - **csv/**: Directory for CSV files.
  - **json/**: Directory for JSON files.

- **figures/**: Contains figures generated from the analysis.
  - **monthly/**: Directory for monthly figures.

- **analysis_cache/**: Directory for caching analysis results to speed up future runs.

- **requirements.txt**: Lists the dependencies required for the project.

## Usage

1. **Setup**: Ensure all dependencies are installed as listed in `requirements.txt`.
2. **Notebook Analysis**: Open any of the notebooks in the `notebooks/` directory to analyze telescope elevation error data for the corresponding month.
3. **Automation**: Use the scripts in the `scripts/` directory to generate new monthly notebooks or execute all notebooks in batch mode.

## Contribution

Contributions to improve the analysis or add new features are welcome. Please submit a pull request or open an issue for discussion.

## License

This project is licensed under the MIT License. See the LICENSE file for details.