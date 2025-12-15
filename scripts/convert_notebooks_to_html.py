#!/usr/bin/env python3
"""
Converts all Jupyter Notebooks to HTML for GitHub Pages.

This script converts all notebooks in the data_analysis/ folder
to HTML files and saves them in the corresponding structure
in the docs/ folder.
"""

import subprocess
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ANALYSIS_DIR = PROJECT_ROOT / "data_analysis"
DOCS_DIR = PROJECT_ROOT / "docs"

# Mapping from analysis folders to docs subfolders
NOTEBOOK_MAPPINGS = {
    "eda": "eda",
    "party": "party",
    "geographical": "geographical",
    "demographical": "demographical",
    "temporal": "temporal",
    "conclusion": "conclusion",
}


def convert_notebook(notebook_path: Path, output_dir: Path):
    """Converts a single notebook to HTML."""
    print(f"Converting: {notebook_path.name}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert notebook to HTML
    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "html",
        "--output-dir",
        str(output_dir),
        str(notebook_path),
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✓ Successfully converted to {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print("  ✗ Error: jupyter nbconvert not found. Install it with: pip install nbconvert")
        return False


def main():
    """Main function to convert all notebooks."""
    print("=" * 60)
    print("Converting Jupyter Notebooks to HTML")
    print("=" * 60)
    print()
    
    # Check if data_analysis directory exists
    if not DATA_ANALYSIS_DIR.exists():
        print(f"Error: {DATA_ANALYSIS_DIR} does not exist!")
        return
    
    # Create docs directory if it doesn't exist
    DOCS_DIR.mkdir(exist_ok=True)
    
    converted = 0
    failed = 0
    
    # Search through all analysis folders
    for analysis_dir, docs_subdir in NOTEBOOK_MAPPINGS.items():
        analysis_path = DATA_ANALYSIS_DIR / analysis_dir
        
        if not analysis_path.exists():
            print(f"Warning: {analysis_path} does not exist, skipping...")
            continue
        
        # Find all .ipynb files
        notebooks = list(analysis_path.glob("*.ipynb"))
        
        if not notebooks:
            print(f"No notebooks found in {analysis_dir}/")
            continue
        
        print(f"\n📁 {analysis_dir}/")
        output_dir = DOCS_DIR / docs_subdir
        
        for notebook in notebooks:
            if convert_notebook(notebook, output_dir):
                converted += 1
            else:
                failed += 1
    
    print()
    print("=" * 60)
    print(f"Done! {converted} notebooks converted, {failed} errors")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Review the HTML files in the docs/ folder")
    print("2. Commit and push the changes")
    print("3. Enable GitHub Pages in the repository settings")


if __name__ == "__main__":
    main()







