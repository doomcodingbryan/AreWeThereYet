"""
Generate a schema-aligned country_data CSV using Numbeo-derived data only.

Output:
- data/country_data_numbeo_only.csv
"""

import csv
import io
import os
import urllib.request


SOURCE_URL = (
    "https://raw.githubusercontent.com/kautzarichramsyah/medium/main/"
    "numbeo_quality_of_life_index_by_country_2023_12_28.csv"
)

TARGET_FIELDS = [
    "country",
    "ef_epi_score",
    "ef_epi_band",
    "quality_of_life_index",
    "purchasing_power_index",
    "safety_index",
    "health_care_index",
    "cost_of_living_index",
    "pollution_index",
    "climate_index",
    "official_languages",
    "english_official",
    "region",
    "population",
    "currency",
    "gdp_per_capita_usd",
    "skilled_worker_visa",
    "visa_name",
    "passport_visa_free_count",
]


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, "data", "country_data_numbeo_only.csv")

    with urllib.request.urlopen(SOURCE_URL, timeout=30) as resp:
        raw = resp.read().decode("utf-8")

    source_rows = list(csv.DictReader(io.StringIO(raw)))
    out_rows = []

    for row in source_rows:
        out = {f: "" for f in TARGET_FIELDS}
        out["country"] = (row.get("country") or "").strip()
        out["quality_of_life_index"] = (row.get("quality_of_life_index") or "").strip()
        out["purchasing_power_index"] = (row.get("purchasing_power_index") or "").strip()
        out["safety_index"] = (row.get("safety_index") or "").strip()
        out["health_care_index"] = (row.get("health_care_index") or "").strip()
        out["cost_of_living_index"] = (row.get("cost_of_living_index") or "").strip()
        out["pollution_index"] = (row.get("pollution_index") or "").strip()
        out["climate_index"] = (row.get("climate_index") or "").strip()
        out_rows.append(out)

    out_rows.sort(key=lambda r: r["country"].lower())

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TARGET_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
