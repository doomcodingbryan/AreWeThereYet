"""
Search Engine using TF-IDF vectorization and cosine-similarity ranking.

Given country profiles and structured country metadata, this module builds 
a TF-IDF index and ranks countries with a user's "vibe"  query.
"""

import csv
import os

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer

from country_profiles import build_country_profiles

# Synonym map: query word → extra terms to append
SYNONYMS = {
    # Cost
    "cheap":       ["affordable", "inexpensive", "low cost", "budget"],
    "affordable":  ["cheap", "inexpensive", "low cost", "budget"],
    "expensive":   ["high cost", "pricey", "costly"],
    "budget":      ["cheap", "affordable", "low cost"],

    # Safety (non-sentiment)
    "safe":        ["low crime", "peaceful", "secure", "safety"],
    "unsafe":      ["high crime", "dangerous", "crime rate"],
    "dangerous":   ["unsafe", "high crime", "crime"],

    # Weather / climate
    "warm":        ["tropical", "hot", "sunny", "mild", "humid"],
    "hot":         ["warm", "tropical", "sunny", "heat"],
    "sunny":       ["warm", "tropical", "sunshine", "clear skies"],
    "tropical":    ["warm", "humid", "hot", "beach", "rainforest"],
    "cold":        ["cool", "nordic", "winter", "snow", "freezing"],
    "cool":        ["mild", "temperate", "cold"],
    "mild":        ["temperate", "moderate", "pleasant"],
    "beach":       ["coastal", "ocean", "tropical", "seaside"],
    "snow":        ["cold", "winter", "skiing", "alpine"],

    # Nature
    "nature":      ["outdoors", "hiking", "mountains", "forests", "wildlife"],
    "hiking":      ["mountains", "trails", "outdoors", "nature", "trekking"],
    "mountains":   ["hiking", "alps", "alpine", "elevation", "skiing"],
    "ocean":       ["sea", "beach", "coastal", "diving", "surfing"],

    # Lifestyle
    "nightlife":   ["bars", "clubs", "social", "entertainment", "vibrant"],
    "food":        ["cuisine", "restaurants", "culinary", "gastronomy"],
    "culture":     ["arts", "history", "museums", "heritage", "traditions"],
    "expat":       ["expatriate", "foreigner", "immigrant", "relocation", "abroad"],
    "remote work": ["digital nomad", "wifi", "coworking", "internet", "laptop"],
    "nomad":       ["remote work", "digital nomad", "freelance", "coworking"],

    # English
    "english":     ["english speaking", "anglophone", "english language"],

    # Healthcare
    "healthcare":  ["medical", "hospitals", "health system", "doctors"],
    "medical":     ["healthcare", "hospitals", "health care"],

    # Visa / immigration
    "visa":        ["immigration", "residency", "work permit", "permit"],
    "retire":      ["retirement", "pension", "retiree", "expat"],
    "immigrate":   ["visa", "residency", "immigration", "move abroad"],
}


def expand_query(query: str) -> str:
    """Append synonyms for recognized terms in the query."""
    lower = query.lower()
    extras = []
    for term, synonyms in SYNONYMS.items():
        if term in lower:
            extras.extend(synonyms)
    if extras:
        return query + " " + " ".join(extras)
    return query


def _load_country_metadata(csv_path=None):
    """Load country_data.csv into a dict keyed by country name."""
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        csv_path = os.path.join(project_root, "data", "country_data.csv")

    metadata = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("country", "").strip()
            if name:
                metadata[name] = row
    return metadata


class CountrySearchEngine:
    """TF-IDF search engine over country profiles."""

    def __init__(self):
        self.country_names = []
        self.tfidf_matrix = None
        self.vectorizer = None
        self.svd = None
        self.lsa_matrix = None
        self.metadata = {}

    def build_index(self, json_path=None, csv_path=None):
        """
        Build the TF-IDF index from Reddit data and load metadata.
        """
        # 1. Build country text profiles
        profiles = build_country_profiles(json_path)

        # 2. Prepare ordered lists
        self.country_names = sorted(profiles.keys())
        documents = [profiles[c] for c in self.country_names]

        # 3. Fit TF-IDF
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_df=0.85,
            min_df=1,
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)

        # 4. Apply SVD (LSA) — reduce to 30 latent dimensions
        # n_components capped at n_countries - 1
        n_components = min(30, len(self.country_names) - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        lsa_raw = self.svd.fit_transform(self.tfidf_matrix)
        # Normalize rows so cosine similarity is just a dot product
        self.lsa_matrix = Normalizer(copy=False).fit_transform(lsa_raw)

        # 5. Load structured metadata
        self.metadata = _load_country_metadata(csv_path)

        print(f"TF-IDF index built: {self.tfidf_matrix.shape[0]} countries, "
              f"{self.tfidf_matrix.shape[1]} terms")
        print(f"LSA applied: {n_components} latent dimensions")

    def search(self, query, top_k=10):
        """
        Rank countries by cosine similarity to the query.
        """
        if self.vectorizer is None or self.lsa_matrix is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        if not query or not query.strip():
            return []

        # Expand query with synonyms before vectorizing
        query = expand_query(query)

        # Transform query into TF-IDF then LSA space
        query_tfidf = self.vectorizer.transform([query])
        query_lsa = self.svd.transform(query_tfidf)
        query_lsa = Normalizer().fit_transform(query_lsa)

        # Compute cosine similarity against all countries in LSA space
        similarities = cosine_similarity(query_lsa, self.lsa_matrix).flatten()

        # Rank by descending similarity
        ranked_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            country = self.country_names[idx]
            score = float(similarities[idx])

            # Skip countries with zero similarity
            if score <= 0:
                continue

            # Attach metadata if available
            meta = self.metadata.get(country, {})

            results.append({
                "country": country,
                "score": score,
                "metadata": {
                    "region": meta.get("region", ""),
                    "quality_of_life_index": meta.get("quality_of_life_index", ""),
                    "cost_of_living_index": meta.get("cost_of_living_index", ""),
                    "safety_index": meta.get("safety_index", ""),
                    "health_care_index": meta.get("health_care_index", ""),
                    "climate_index": meta.get("climate_index", ""),
                    "official_languages": meta.get("official_languages", ""),
                    "english_official": meta.get("english_official", ""),
                    "gdp_per_capita_usd": meta.get("gdp_per_capita_usd", ""),
                    "skilled_worker_visa": meta.get("skilled_worker_visa", ""),
                    "visa_name": meta.get("visa_name", ""),
                }
            })

        # Normalize scores relative to the best match so percentages are meaningful
        if results:
            max_score = results[0]["score"]
            if max_score > 0:
                for r in results:
                    r["score"] = round(r["score"] / max_score, 4)

        return results
