"""
Build country language metadata CSV from Wikidata with ISO country codes.

Output:
- data/country_languages.csv

Schema:
- country, iso2, iso3, official_languages, major_languages, english_official
"""

import csv
import os
import urllib.parse
import urllib.request
import json


WIKIDATA_SPARQL = """
SELECT ?countryLabel ?iso2 ?iso3 (GROUP_CONCAT(DISTINCT ?officialLangLabel; separator=", ") AS ?official_languages)
WHERE {
  ?country wdt:P31 wd:Q3624078 .
  ?country wdt:P297 ?iso2 .
  ?country wdt:P298 ?iso3 .
  OPTIONAL {
    ?country wdt:P37 ?officialLang .
    ?officialLang rdfs:label ?officialLangLabel .
    FILTER (lang(?officialLangLabel) = "en")
  }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
GROUP BY ?countryLabel ?iso2 ?iso3
ORDER BY ?countryLabel
"""


def fetch_wikidata_rows():
    """Query Wikidata SPARQL endpoint and return parsed rows."""
    query = urllib.parse.quote(WIKIDATA_SPARQL)
    url = f"https://query.wikidata.org/sparql?format=json&query={query}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "AreWeThereYet-country-language-builder/1.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload.get("results", {}).get("bindings", [])


def value(row, key):
    """Safe extractor from SPARQL JSON bindings."""
    return row.get(key, {}).get("value", "").strip()


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, "data", "country_languages.csv")

    rows = fetch_wikidata_rows()
    out = []
    for row in rows:
        country = value(row, "countryLabel")
        iso2 = value(row, "iso2")
        iso3 = value(row, "iso3")
        official_languages = value(row, "official_languages")
        major_languages = official_languages
        english_official = "true" if "english" in official_languages.lower() else "false"
        out.append(
            {
                "country": country,
                "iso2": iso2,
                "iso3": iso3,
                "official_languages": official_languages,
                "major_languages": major_languages,
                "english_official": english_official,
            }
        )

    out.sort(key=lambda r: r["country"].lower())

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "country",
                "iso2",
                "iso3",
                "official_languages",
                "major_languages",
                "english_official",
            ],
        )
        writer.writeheader()
        writer.writerows(out)

    print(f"Wrote {len(out)} rows to {output_path}")


if __name__ == "__main__":
    main()
