# Data Directory

This project expects data files under these folders:

- `raw/`: original input CSV files
- `processed/`: generated intermediate CSV files
- `database/`: local SQLite databases

These files are ignored in git because the full datasets and generated artifacts are too large for a normal GitHub repository.

To reproduce the project locally, place the JD source CSVs from https://huggingface.co/datasets/a6687543/MSOM_Data_Driven_Challenge_2020/tree/main into `data/raw/` and then run the scripts documented in the top-level `README.md`.
