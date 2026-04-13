"""
Backfill missing country metrics using an external public dataset.

Source dataset:
- URL: https://raw.githubusercontent.com/kautzarichramsyah/medium/main/numbeo_quality_of_life_index_by_country_2023_12_28.csv
- Fields used: safety_index, climate_index, cost_of_living_index
- Retrieved: 2026-04-13

This script only fills empty values in data/country_data.csv and does not overwrite
existing non-empty values.
"""

import csv
import io
import os
import re
import urllib.request

from country_profiles import build_country_profiles


EXTERNAL_URL = (
    "https://raw.githubusercontent.com/kautzarichramsyah/medium/main/"
    "numbeo_quality_of_life_index_by_country_2023_12_28.csv"
)


def normalize_country_name(name):
    """Normalize country names for robust matching across datasets."""
    if not name:
        return ""
    text = name.strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"[()'.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    aliases = {
        "usa": "united states",
        "uk": "united kingdom",
        "czechia": "czech republic",
        "bosnia and herzegovina": "bosnia and herzegovina",
        "hong kong": "hong kong china",
        "south korea": "korea south",
    }
    return aliases.get(text, text)


def load_external_metrics():
    """Download and parse external CSV into normalized country metric map."""
    with urllib.request.urlopen(EXTERNAL_URL, timeout=30) as resp:
        content = resp.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(content))
    metrics = {}
    for row in reader:
        country = normalize_country_name(row.get("country", ""))
        if not country:
            continue
        metrics[country] = {
            "quality_of_life_index": (row.get("quality_of_life_index") or "").strip(),
            "safety_index": (row.get("safety_index") or "").strip(),
            "health_care_index": (row.get("health_care_index") or "").strip(),
            "climate_index": (row.get("climate_index") or "").strip(),
            "cost_of_living_index": (row.get("cost_of_living_index") or "").strip(),
        }
    return metrics


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "data", "country_data.csv")

    external = load_external_metrics()

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    fill_fields = [
        "quality_of_life_index",
        "health_care_index",
        "safety_index",
        "climate_index",
        "cost_of_living_index",
    ]
    fill_count = {k: 0 for k in fill_fields}
    unresolved = []

    for row in rows:
        raw_country = row.get("country", "")
        key = normalize_country_name(raw_country)
        ext = external.get(key)
        if ext is None:
            unresolved.append(raw_country)
            continue

        for field in fill_fields:
            if (row.get(field) or "").strip() == "" and ext.get(field, "") != "":
                row[field] = ext[field]
                fill_count[field] += 1

    # Add new rows for countries that appear in Reddit profiles but are missing from CSV.
    profile_countries = set(build_country_profiles().keys())
    existing_countries = {row.get("country", "").strip() for row in rows}
    added_rows = 0
    for country in sorted(profile_countries - existing_countries):
        key = normalize_country_name(country)
        ext = external.get(key)
        if ext is None:
            unresolved.append(country)
            continue
        new_row = {field: "" for field in fieldnames}
        new_row["country"] = country
        for field in fill_fields:
            if field in new_row:
                new_row[field] = ext.get(field, "")
                if new_row[field]:
                    fill_count[field] += 1
        rows.append(new_row)
        added_rows += 1

    rows.sort(key=lambda r: (r.get("country", "") or "").lower())

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Backfill complete.")
    for field in fill_fields:
        print(f"Filled {field}: {fill_count[field]}")
    print(f"Rows added from external dataset: {added_rows}")
    print(f"Countries without external match: {len(set(unresolved))}")


if __name__ == "__main__":
    main()
