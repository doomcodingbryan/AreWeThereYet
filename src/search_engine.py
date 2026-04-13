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
from sklearn.preprocessing import normalize

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


def _safe_float(value):
    """Parse numeric metadata safely; return None if unavailable."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _bucket_numeric(value):
    """
    Convert a numeric value to coarse labels used in metadata text docs.
    This keeps ranking semantic (TF-IDF/SVD) without hard-coded query logic.
    """
    if value is None:
        return "unknown"
    if value >= 75:
        return "very_high"
    if value >= 60:
        return "high"
    if value >= 45:
        return "medium"
    return "low"


def _load_country_metadata(csv_path=None):
    """Load country_data.csv into a dict keyed by country name."""
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        csv_path = os.path.join(project_root, "data", "country_data_numbeo_only.csv")

    # Fallback to legacy file if numbeo-only CSV is unavailable.
    if not os.path.exists(csv_path):
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


def _normalize_country_name(name):
    """Normalize country names for cross-dataset joining."""
    if not name:
        return ""
    text = str(name).strip().lower()
    aliases = {
        "usa": "united states",
        "united states of america": "united states",
        "uk": "united kingdom",
        "uae": "united arab emirates",
        "czechia": "czech republic",
        "south korea": "korea south",
        "north macedonia": "north macedonia",
        "hong kong": "hong kong",
        "hong kong (china)": "hong kong",
        "kingdom of the netherlands": "netherlands",
        "korea, south": "korea south",
        "bosnia and herzegovina": "bosnia and herzegovina",
    }
    return aliases.get(text, text)


def _load_language_metadata(csv_path=None):
    """Load country_languages.csv into a normalized dict keyed by country name."""
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        csv_path = os.path.join(project_root, "data", "country_languages.csv")

    if not os.path.exists(csv_path):
        return {}

    language_meta = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = _normalize_country_name(row.get("country", ""))
            if key:
                language_meta[key] = row
    return language_meta


def _extract_language_constraints(query):
    """
    Extract language constraints from query text.
    Returns a set of normalized language names requested by the user.
    """
    if not query:
        return set()
    q = query.lower()
    aliases = {
        "english": ["english speaking", "speak english", "english"],
        "spanish": ["spanish speaking", "speak spanish", "spanish"],
        "french": ["french speaking", "speak french", "french"],
        "german": ["german speaking", "speak german", "german"],
        "italian": ["italian speaking", "speak italian", "italian"],
        "portuguese": ["portuguese speaking", "speak portuguese", "portuguese"],
        "arabic": ["arabic speaking", "speak arabic", "arabic"],
        "japanese": ["japanese speaking", "speak japanese", "japanese"],
        "korean": ["korean speaking", "speak korean", "korean"],
        "mandarin": ["mandarin speaking", "speak mandarin", "mandarin", "chinese speaking"],
    }
    requested = set()
    for canonical, patterns in aliases.items():
        if any(p in q for p in patterns):
            requested.add(canonical)
    return requested


class CountrySearchEngine:
    """TF-IDF search engine over country profiles."""

    def __init__(self):
        self.country_names = []
        self.tfidf_matrix = None
        self.vectorizer = None
        self.svd = None
        self.lsa_matrix = None
        self.metadata = {}
        self.n_components = 100
        # Blend lexical match (TF-IDF) with semantic match (SVD/LSA).
        self.hybrid_alpha = 0.35
        self.metadata_blend = 0.40
        self.stage2_candidate_k = 30
        self.metadata_vectorizer = None
        self.metadata_tfidf_matrix = None
        self.metadata_svd = None
        self.metadata_lsa_matrix = None
        self.language_metadata = {}
        self.intent_anchors = {
            "safety": "safe secure safety security low_crime safety_index",
            "climate": "weather climate warm sunny mild pleasant climate_index",
            "cost": "affordable budget low_cost cheap inexpensive affordability cost_of_living_index",
        }
        self.anchor_tfidf_vectors = {}
        self.anchor_vectors = {}
        self.qol_weight = 0.20

    def _build_metadata_document(self, country, meta):
        """Build a text document from structured metadata for semantic retrieval."""
        safety = _safe_float(meta.get("safety_index"))
        climate = _safe_float(meta.get("climate_index"))
        cost = _safe_float(meta.get("cost_of_living_index"))
        quality = _safe_float(meta.get("quality_of_life_index"))
        healthcare = _safe_float(meta.get("health_care_index"))
        region = meta.get("region", "")
        languages = meta.get("official_languages", "")

        parts = [
            country,
            region,
            languages,
            f"safety_index {meta.get('safety_index', '')} safety safe secure security {_bucket_numeric(safety)}",
            f"climate_index {meta.get('climate_index', '')} climate weather warm sunny {_bucket_numeric(climate)}",
            f"cost_of_living_index {meta.get('cost_of_living_index', '')} affordability affordable budget cost {_bucket_numeric(None if cost is None else 100 - cost)}",
            f"quality_of_life_index {meta.get('quality_of_life_index', '')} quality_of_life {_bucket_numeric(quality)}",
            f"health_care_index {meta.get('health_care_index', '')} healthcare {_bucket_numeric(healthcare)}",
            f"skilled_worker_visa {meta.get('skilled_worker_visa', '')} visa_name {meta.get('visa_name', '')}",
        ]
        return " ".join(p for p in parts if p.strip())

    def _normalize_feature(self, value):
        """Normalize 0-100 index values to [0, 1], with neutral fallback."""
        if value is None:
            return 0.5
        return max(0.0, min(1.0, value / 100.0))

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
        n_samples, n_features = self.tfidf_matrix.shape

        # 4. Fit SVD/LSA projection when enough data exists.
        max_components = max(1, min(n_samples - 1, n_features - 1))
        component_count = min(self.n_components, max_components)
        self.svd = TruncatedSVD(n_components=component_count, random_state=42)
        self.lsa_matrix = normalize(self.svd.fit_transform(self.tfidf_matrix))

        # 5. Load structured metadata
        self.metadata = _load_country_metadata(csv_path)
        self.language_metadata = _load_language_metadata()
        metadata_documents = [
            self._build_metadata_document(c, self.metadata.get(c, {}))
            for c in self.country_names
        ]

        # 6. Build a second TF-IDF + SVD index over metadata text.
        self.metadata_vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=1,
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.metadata_tfidf_matrix = self.metadata_vectorizer.fit_transform(metadata_documents)
        meta_samples, meta_features = self.metadata_tfidf_matrix.shape
        meta_max_components = max(1, min(meta_samples - 1, meta_features - 1))
        meta_component_count = min(self.n_components, meta_max_components)
        self.metadata_svd = TruncatedSVD(
            n_components=meta_component_count,
            random_state=42
        )
        self.metadata_lsa_matrix = normalize(
            self.metadata_svd.fit_transform(self.metadata_tfidf_matrix)
        )
        self.anchor_vectors = {}
        self.anchor_tfidf_vectors = {}
        for name, text in self.intent_anchors.items():
            anchor_tfidf = self.metadata_vectorizer.transform([text])
            anchor_lsa = normalize(self.metadata_svd.transform(anchor_tfidf))
            self.anchor_tfidf_vectors[name] = anchor_tfidf
            self.anchor_vectors[name] = anchor_lsa

        print(f"TF-IDF index built: {self.tfidf_matrix.shape[0]} countries, "
              f"{self.tfidf_matrix.shape[1]} terms")
        print(f"SVD/LSA projection built: {component_count} latent dimensions")
        print(
            f"Metadata TF-IDF/SVD built: {self.metadata_tfidf_matrix.shape[1]} terms, "
            f"{meta_component_count} latent dimensions"
        )

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

        # Compute lexical similarity in raw TF-IDF space.
        query_vec = self.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Compute semantic similarity in reduced LSA space.
        similarities = tfidf_scores
        if self.svd is not None and self.lsa_matrix is not None:
            query_lsa = normalize(self.svd.transform(query_vec))
            lsa_scores = cosine_similarity(query_lsa, self.lsa_matrix).flatten()
            similarities = (
                self.hybrid_alpha * tfidf_scores
                + (1 - self.hybrid_alpha) * lsa_scores
            )

        # Query the structured-metadata semantic index as well.
        metadata_scores = similarities * 0.0
        if self.metadata_vectorizer is not None and self.metadata_tfidf_matrix is not None:
            meta_query_vec = self.metadata_vectorizer.transform([query])
            meta_tfidf_scores = cosine_similarity(
                meta_query_vec, self.metadata_tfidf_matrix
            ).flatten()
            metadata_scores = meta_tfidf_scores
            if self.metadata_svd is not None and self.metadata_lsa_matrix is not None:
                meta_query_lsa = normalize(self.metadata_svd.transform(meta_query_vec))
                meta_lsa_scores = cosine_similarity(
                    meta_query_lsa, self.metadata_lsa_matrix
                ).flatten()
                metadata_scores = (
                    self.hybrid_alpha * meta_tfidf_scores
                    + (1 - self.hybrid_alpha) * meta_lsa_scores
                )

        # Stage 1 base semantic score combines profile and metadata retrieval.
        stage1_scores = (
            (1 - self.metadata_blend) * similarities
            + self.metadata_blend * metadata_scores
        )

        # Stage 2: rerank top semantic candidates using learned-style weighted signals.
        all_indices = list(range(len(stage1_scores)))
        language_constraints = _extract_language_constraints(query)
        if language_constraints:
            filtered = []
            for idx in all_indices:
                country = self.country_names[idx]
                meta = self.metadata.get(country, {})
                language_row = self.language_metadata.get(_normalize_country_name(country), {})
                language_source = (
                    language_row.get("official_languages")
                    or language_row.get("major_languages")
                    or meta.get("official_languages", "")
                )
                country_languages = (language_source or "").lower()
                if all(lang in country_languages for lang in language_constraints):
                    filtered.append(idx)
            if filtered:
                all_indices = filtered

        candidate_k = min(self.stage2_candidate_k, len(all_indices))
        candidate_indices = sorted(
            all_indices,
            key=lambda i: stage1_scores[i],
            reverse=True
        )[:candidate_k]

        # Compute soft intent weights from semantic similarity to anchor concepts.
        anchor_scores = {"safety": 0.0, "climate": 0.0, "cost": 0.0}
        if self.metadata_vectorizer is not None and self.metadata_svd is not None:
            meta_query_vec = self.metadata_vectorizer.transform([query])
            meta_query_lsa = normalize(self.metadata_svd.transform(meta_query_vec))
            for name, anchor_vec in self.anchor_vectors.items():
                tfidf_sim = cosine_similarity(
                    meta_query_vec, self.anchor_tfidf_vectors[name]
                ).flatten()[0]
                lsa_sim = cosine_similarity(meta_query_lsa, anchor_vec).flatten()[0]
                # Use mostly TF-IDF cosine (non-negative), with LSA as soft semantic signal.
                lsa_scaled = (float(lsa_sim) + 1.0) / 2.0
                anchor_scores[name] = (
                    0.7 * max(0.0, float(tfidf_sim))
                    + 0.3 * max(0.0, lsa_scaled)
                )

        anchor_total = sum(anchor_scores.values())
        if anchor_total > 0:
            safety_weight = anchor_scores["safety"] / anchor_total
            climate_weight = anchor_scores["climate"] / anchor_total
            cost_weight = anchor_scores["cost"] / anchor_total
        else:
            safety_weight = climate_weight = cost_weight = 1 / 3

        # Intent confidence from semantic anchor alignment (not keyword logic).
        safety_intent_conf = anchor_scores["safety"]
        intent_sum = anchor_total if anchor_total > 0 else 1.0
        safety_intent_share = anchor_scores["safety"] / intent_sum

        # Keep semantic signals dominant by default; strengthen structured reranking
        # only when safety intent is semantically strong.
        stage2_profile_w = 0.45
        stage2_meta_w = 0.30
        stage2_feature_total = 0.25
        safety_floor = None
        if safety_intent_conf >= 0.22 and safety_intent_share >= 0.45:
            # For "safe / secure" style queries, lift safety as a stronger rerank signal.
            stage2_profile_w = 0.35
            stage2_meta_w = 0.20
            stage2_feature_total = 0.45
            safety_floor = 60.0
            safety_weight = max(safety_weight, 0.65)
            remaining = max(0.0, 1.0 - safety_weight)
            if climate_weight + cost_weight > 0:
                norm = climate_weight + cost_weight
                climate_weight = remaining * (climate_weight / norm)
                cost_weight = remaining * (cost_weight / norm)
            else:
                climate_weight = remaining * 0.5
                cost_weight = remaining * 0.5
        elif safety_weight >= 0.38:
            # Mixed intents like "good weather and safe": remove only very unsafe options.
            safety_floor = 45.0

        reranked_scores = {}
        for idx in candidate_indices:
            country = self.country_names[idx]
            meta = self.metadata.get(country, {})
            language_row = self.language_metadata.get(_normalize_country_name(country), {})
            language_source = (
                language_row.get("official_languages")
                or language_row.get("major_languages")
                or meta.get("official_languages", "")
            )
            country_languages = (language_source or "").lower()

            if language_constraints:
                if not all(lang in country_languages for lang in language_constraints):
                    continue

            safety_raw = _safe_float(meta.get("safety_index"))
            if safety_floor is not None and (safety_raw is None or safety_raw < safety_floor):
                continue
            safety = self._normalize_feature(safety_raw)
            climate = self._normalize_feature(_safe_float(meta.get("climate_index")))
            cost_inverted = 1.0 - self._normalize_feature(_safe_float(meta.get("cost_of_living_index")))
            qol = self._normalize_feature(_safe_float(meta.get("quality_of_life_index")))
            base_structured = (
                safety_weight * safety
                + climate_weight * climate
                + cost_weight * cost_inverted
            )
            structured_score = (
                (1.0 - self.qol_weight) * base_structured
                + self.qol_weight * qol
            )
            stage2_score = (
                stage2_profile_w * float(similarities[idx])
                + stage2_meta_w * float(metadata_scores[idx])
                + stage2_feature_total * structured_score
            )
            # Safety-aware penalty shaped by semantic safety weight.
            # Keeps NLP central while demoting low-safety countries when safety matters.
            if safety_weight > 0:
                safety_gap = max(0.0, (0.65 - safety) / 0.65)
                penalty = 1.0 - min(0.65, 1.6 * safety_weight * safety_gap)
                stage2_score *= max(0.35, penalty)
            reranked_scores[idx] = stage2_score

        if not reranked_scores:
            reranked_scores = {i: float(stage1_scores[i]) for i in candidate_indices}

        # Rank by stage-2 score for candidates, then fill remainder by stage-1.
        ranked_stage2 = sorted(
            reranked_scores.keys(),
            key=lambda i: reranked_scores[i],
            reverse=True
        )
        remaining = [i for i in all_indices if i not in set(candidate_indices)]
        ranked_remaining = sorted(remaining, key=lambda i: stage1_scores[i], reverse=True)
        final_ranked = (ranked_stage2 + ranked_remaining)[:top_k]

        # Final ranking and score output.
        ranked_indices = sorted(
            final_ranked,
            key=lambda i: reranked_scores.get(i, stage1_scores[i]),
            reverse=True
        )[:top_k]

        results = []
        for idx in ranked_indices:
            country = self.country_names[idx]
            score = float(reranked_scores.get(idx, stage1_scores[idx]))

            # Skip countries with zero similarity
            if score <= 0:
                continue

            # Attach metadata if available
            meta = self.metadata.get(country, {})
            language_row = self.language_metadata.get(_normalize_country_name(country), {})
            language_value = (
                language_row.get("official_languages")
                or language_row.get("major_languages")
                or meta.get("official_languages", "")
            )
            english_official = (
                language_row.get("english_official")
                or meta.get("english_official", "")
            )

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
                    "official_languages": language_value,
                    "english_official": english_official,
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
