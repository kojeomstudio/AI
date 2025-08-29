"""
Generate sample Excel dataset without running training.

Usage:
  python generate_sample_data.py --data-dir data --excel-name sample.xlsx
"""
from pathlib import Path
from src.package_size_predict.data import DatasetPaths, ensure_sample_excel


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Generate sample Excel dataset")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--excel-name", type=str, default="sample.xlsx")
    args = p.parse_args()

    paths = DatasetPaths(root=args.data_dir, excel=args.data_dir / args.excel_name)
    ensure_sample_excel(paths)
    print(f"Sample data generated at: {paths.excel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

