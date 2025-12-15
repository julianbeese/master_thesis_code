# GitHub Pages Setup

This folder contains the static files for GitHub Pages publication.

## Structure

```
docs/
├── index.md              # Main page
├── eda/                  # Exploratory Data Analysis
├── party/                # Party analyses
├── geographical/         # Geographical analyses
├── demographical/        # Demographical analyses
└── temporal/             # Temporal analyses
```

## Converting Notebooks to HTML

To convert the Jupyter Notebooks to HTML, use the `convert_notebooks_to_html.py` script in the main directory:

```bash
python scripts/convert_notebooks_to_html.py
```

Or manually with jupyter nbconvert:

```bash
# Single notebook
jupyter nbconvert --to html --output-dir docs/temporal data_analysis/temporal/distribution_of_frame_observations.ipynb

# All notebooks
jupyter nbconvert --to html --output-dir docs/eda data_analysis/eda/*.ipynb
jupyter nbconvert --to html --output-dir docs/party data_analysis/party/*.ipynb
jupyter nbconvert --to html --output-dir docs/geographical data_analysis/geographical/*.ipynb
jupyter nbconvert --to html --output-dir docs/demographical data_analysis/demographical/*.ipynb
jupyter nbconvert --to html --output-dir docs/temporal data_analysis/temporal/*.ipynb
```

## Enabling GitHub Pages

1. Go to your GitHub repository
2. Settings → Pages
3. Source: `main` branch, `/docs` folder
4. Click "Save"
5. The page will be available after a few minutes at: `https://[username].github.io/master_thesis_code`

## Notes

- Make sure all HTML files and assets (graphics) are in the `docs/` folder
- Relative paths work best
- The page is automatically updated on every push
