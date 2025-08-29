**Package Size Prediction (PyTorch)**

- **Goal**: Train a simple linear regression model with PyTorch to predict a program's package size over time using daily data from an Excel file.
- **Data**: If `data/sample.xlsx` does not exist, a sample dataset is generated automatically.
- **Logging**: Logs print to console and are saved under `logs/` as `training_YYYYMMDD_HHMMSS.log`.

**Run**
- `python run_training.py`
- Optional args:
  - `--data-dir data` directory to find/create the Excel file
  - `--excel-name sample.xlsx` Excel filename
  - `--epochs 300` number of epochs
  - `--batch-size 32` mini-batch size
  - `--lr 0.01` learning rate
  - `--val-split 0.2` validation fraction
  - `--logs logs` log directory

**Dependencies**
- Python 3.9+
- `torch` (PyTorch)
- `pandas`
- `openpyxl` (Excel IO engine)

Install (example):
- `pip install torch pandas openpyxl`

**Notes**
- Dates are encoded as an ordinal day index starting at 0; the model learns `package_size ≈ a * day_index + b`.
- The sample data includes a slight trend and weekly seasonality.

