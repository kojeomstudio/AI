#!/usr/bin/env python3
"""
Replicates the DataGenerator WPF tool logic for CLI/headless use.
Reads Template/*.md → writes GameData/*.json + Source/.../Generated/*Generated.h
"""
import os, re, json, glob

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATE_DIR = os.path.join(ROOT, "Template")
GAMEDATA_DIR = os.path.join(ROOT, "GameData")
CODEGEN_DIR  = os.path.join(ROOT, "Source", "UnrealWorld", "Public", "Data", "Generated")

os.makedirs(GAMEDATA_DIR, exist_ok=True)
os.makedirs(CODEGEN_DIR, exist_ok=True)

def parse_value(type_str, value):
    t = type_str.lower()
    try:
        if t == "int":   return int(value)
        if t == "float": return float(value)
        if t == "bool":  return value.lower() == "true"
    except Exception:
        pass
    return value  # fallback: string

def to_unreal_type(type_str):
    t = type_str.lower()
    if t == "int":   return "int32"
    if t == "float": return "float"
    if t == "bool":  return "bool"
    if t == "fname": return "FName"
    return "FString"

def process_file(md_path):
    filename = os.path.splitext(os.path.basename(md_path))[0]
    print(f"Processing: {filename}.md")

    with open(md_path, encoding="utf-8") as f:
        lines = f.readlines()

    # Extract table rows (lines starting and ending with |)
    table_lines = [l.strip() for l in lines if l.strip().startswith("|") and l.strip().endswith("|")]

    if len(table_lines) < 3:
        print(f"  Skip: no valid table (need >=3 rows, got {len(table_lines)})")
        return

    # Row 0 = headers, Row 1 = separator (skip), Row 2+ = data
    header_cells = [h.strip() for h in table_lines[0].split("|") if h.strip()]

    # Parse type:Name pairs
    headers = []
    for cell in header_cells:
        parts = cell.split(":", 1)
        if len(parts) == 2:
            headers.append((parts[0].strip(), parts[1].strip()))
        else:
            headers.append(("string", cell))

    # Parse data rows
    entries = []
    for i in range(2, len(table_lines)):
        cells = [c.strip() for c in table_lines[i].split("|") if True]
        # Remove empty strings from leading/trailing |
        cells = [c for c in cells if c != ""]
        # The split by | includes the empty strings at start/end from the leading/trailing |
        # Re-split more carefully:
        raw = table_lines[i]
        inner = raw[1:-1]  # strip leading and trailing |
        cells = [c.strip() for c in inner.split("|")]

        if len(cells) != len(headers):
            print(f"  Warning: row {i} has {len(cells)} cells, expected {len(headers)} — skipping")
            continue

        row = {}
        for j, (typ, name) in enumerate(headers):
            row[name] = parse_value(typ, cells[j])
        entries.append(row)

    # Write JSON
    json_path = os.path.join(GAMEDATA_DIR, filename + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"  Saved JSON:    {filename}.json  ({len(entries)} rows)")

    # Write C++ header
    lines_cpp = []
    lines_cpp.append("#pragma once")
    lines_cpp.append("")
    lines_cpp.append('#include "CoreMinimal.h"')
    lines_cpp.append('#include "Engine/DataTable.h"')
    lines_cpp.append(f'#include "{filename}Generated.generated.h"')
    lines_cpp.append("")
    lines_cpp.append("USTRUCT(BlueprintType)")
    lines_cpp.append(f"struct F{filename} : public FTableRowBase")
    lines_cpp.append("{")
    lines_cpp.append("\tGENERATED_BODY()")
    lines_cpp.append("")
    for typ, name in headers:
        ue_type = to_unreal_type(typ)
        lines_cpp.append('\tUPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Data")')
        lines_cpp.append(f"\t{ue_type} {name};")
        lines_cpp.append("")
    lines_cpp.append("};")
    lines_cpp.append("")

    header_path = os.path.join(CODEGEN_DIR, filename + "Generated.h")
    with open(header_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_cpp))
    print(f"  Generated C++: {filename}Generated.h")

def main():
    md_files = sorted(glob.glob(os.path.join(TEMPLATE_DIR, "*.md")))
    if not md_files:
        print(f"No .md files found in {TEMPLATE_DIR}")
        return
    print(f"Root: {ROOT}")
    print(f"Found {len(md_files)} template(s)\n")
    for path in md_files:
        process_file(path)
    print("\nDone.")

if __name__ == "__main__":
    main()
