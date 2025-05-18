# Voice Source Analysis Project

This project is focused on analyzing vocal data of entrepreneurial pitches. The analysis includes a main analysis, as well as specific analyses for both male and female voice data. Additionally, it includes a robustness analysis.

## Project Structure

### Main Analysis Script
- `final_run_voice_analysis.py` - Main script for voice analysis

### Gender-Specific Analysis
- `run_voice_analysis_male.py` - Analysis for male voice data
- `run_voice_analysis_female.py` - Analysis for female voice data

### Robustness Checks
- `analyze_robustness.py` - Robustness analysis for main script

## Dependencies

The project requires the following main dependencies:
- PyQt5 and PyQtWebEngine for GUI components
- matplotlib for visualization
- torch for machine learning capabilities
- Various utility packages (setuptools, wheel, etc.)

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main analysis can be run using:
```bash
python final_run_voice_analysis.py
```

For specific analyses:
- Gender-specific analysis: Use the respective male/female analysis scripts
- Robustness analysis: Use `analyze_robustness.py`

## Results

The analysis results are stored in various CSV files, with the main results in:
- `FINAL_significant_results.csv`
- `FINAL_significant_results_henry.csv`

