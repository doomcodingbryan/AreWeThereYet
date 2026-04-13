"""
SVD Explainability Analysis for AreWeThereYet
----------------------------------------------
Shows:
  1. The top latent dimensions discovered by SVD (top terms per component)
  2. Side-by-side results: TF-IDF only vs. TF-IDF + SVD hybrid
  3. Per-dimension activation breakdown for a query: which latent concepts
     drive the match (positive) and which suppress it (negative)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from search_engine import CountrySearchEngine, expand_query, SYNONYMS

SEP = "=" * 70


def top_terms_per_component(svd, vectorizer, n_terms=8, n_components=10):
    """Return the top positive and negative terms for each SVD component."""
    terms = vectorizer.get_feature_names_out()
    results = []
    for i in range(min(n_components, svd.components_.shape[0])):
        comp = svd.components_[i]
        top_pos_idx = np.argsort(comp)[::-1][:n_terms]
        top_neg_idx = np.argsort(comp)[:n_terms]
        pos_terms = [(terms[j], round(float(comp[j]), 4)) for j in top_pos_idx]
        neg_terms = [(terms[j], round(float(comp[j]), 4)) for j in top_neg_idx]
        results.append({
            "component": i,
            "variance_ratio": round(float(svd.explained_variance_ratio_[i]) * 100, 2),
            "positive": pos_terms,
            "negative": neg_terms,
        })
    return results


def search_tfidf_only(engine, query, top_k=10):
    """Search using only raw TF-IDF cosine similarity (no LSA)."""
    expanded = expand_query(query)
    query_vec = engine.vectorizer.transform([expanded])
    tfidf_scores = cosine_similarity(query_vec, engine.tfidf_matrix).flatten()
    ranked = np.argsort(tfidf_scores)[::-1][:top_k]
    results = []
    for idx in ranked:
        if tfidf_scores[idx] > 0:
            results.append({
                "country": engine.country_names[idx],
                "score": round(float(tfidf_scores[idx]), 5),
            })
    # Normalize
    if results:
        best = results[0]["score"]
        if best > 0:
            for r in results:
                r["score"] = round(r["score"] / best, 4)
    return results


def dimension_breakdown(engine, query, country_name, top_n=8):
    """
    For a (query, country) pair, return the SVD dimensions that most
    contribute to their cosine similarity in LSA space, with sign.

    The LSA cosine between query vector q and country vector d equals:
        sum_k  q_k * d_k   (since both are L2-normalised)
    Each term q_k * d_k is the signed contribution of dimension k.
    """
    expanded = expand_query(query)
    query_vec = engine.vectorizer.transform([expanded])

    # Unnormalized LSA coordinates
    query_lsa_raw = engine.svd.transform(query_vec)[0]       # shape: (n_components,)
    query_lsa = query_lsa_raw / (np.linalg.norm(query_lsa_raw) + 1e-10)

    idx = engine.country_names.index(country_name)
    # Already normalized in engine.lsa_matrix
    country_lsa = engine.lsa_matrix[idx]                     # shape: (n_components,)

    contributions = query_lsa * country_lsa                  # element-wise product

    # Also describe each dimension with its top term
    terms = engine.vectorizer.get_feature_names_out()
    top_terms_by_dim = []
    for k in range(engine.svd.components_.shape[0]):
        comp = engine.svd.components_[k]
        top_idx = np.argmax(comp)
        top_terms_by_dim.append(terms[top_idx])

    breakdown = []
    for k in range(len(contributions)):
        breakdown.append({
            "dim": k,
            "contribution": float(contributions[k]),
            "query_coord": float(query_lsa[k]),
            "doc_coord": float(country_lsa[k]),
            "top_term": top_terms_by_dim[k],
            "var_pct": round(float(engine.svd.explained_variance_ratio_[k]) * 100, 2),
        })

    pos = sorted([b for b in breakdown if b["contribution"] > 0],
                 key=lambda x: x["contribution"], reverse=True)[:top_n]
    neg = sorted([b for b in breakdown if b["contribution"] < 0],
                 key=lambda x: x["contribution"])[:top_n]
    return pos, neg, float(np.dot(query_lsa, country_lsa))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  AreWeThereYet — SVD / LSA Explainability Report")
    print(SEP)

    engine = CountrySearchEngine()
    engine.build_index()
    print()

    # ── Section 1: Latent dimensions ──────────────────────────────────────────
    print(SEP)
    print("  SECTION 1 — Top latent dimensions (SVD components on Reddit corpus)")
    print(SEP)
    components = top_terms_per_component(engine.svd, engine.vectorizer,
                                         n_terms=8, n_components=15)
    total_var = sum(c["variance_ratio"] for c in components)
    print(f"  Dimensions shown: {len(components)} of {engine.svd.n_components}")
    print(f"  Variance explained by shown dims: {round(total_var,1)}%")
    print()
    for c in components:
        pos_str = ", ".join(f"{t}({w:+.3f})" for t, w in c["positive"])
        neg_str = ", ".join(f"{t}({w:+.3f})" for t, w in c["negative"])
        print(f"  Dim {c['component']:3d}  [{c['variance_ratio']:5.2f}% var]")
        print(f"    + {pos_str}")
        print(f"    - {neg_str}")
        print()

    # ── Section 2: Side-by-side: TF-IDF only vs Hybrid ───────────────────────
    queries = [
        "good air quality",
        "cheap cost of living",
        "safe country to raise a family",
        "warm weather beach lifestyle",
        "excellent healthcare system",
    ]

    print(SEP)
    print("  SECTION 2 — Search results: TF-IDF only vs. Hybrid (TF-IDF + SVD)")
    print(SEP)
    for q in queries:
        tfidf_results = search_tfidf_only(engine, q, top_k=8)
        hybrid_results = engine.search(q, top_k=8)
        print(f"\n  Query: \"{q}\"")
        print(f"  {'Rank':<5} {'TF-IDF only':<28} {'score':>7}   {'TF-IDF + SVD Hybrid':<28} {'score':>7}")
        print("  " + "-"*62)
        for i in range(max(len(tfidf_results), len(hybrid_results))):
            t_entry = tfidf_results[i] if i < len(tfidf_results) else {"country": "—", "score": 0.0}
            h_entry = hybrid_results[i] if i < len(hybrid_results) else {"country": "—", "score": 0.0}
            print(f"  {i+1:<5} {t_entry['country']:<28} {t_entry['score']:>7.4f}   "
                  f"{h_entry['country']:<28} {h_entry['score']:>7.4f}")

    # ── Section 3: Dimension activation for specific results ──────────────────
    analysis_cases = [
        ("good air quality",          "New Zealand"),
        ("good air quality",          "India"),
        ("warm weather beach lifestyle", "Thailand"),
        ("safe country to raise a family", "Iceland"),
        ("cheap cost of living",      "Vietnam"),
    ]

    print()
    print(SEP)
    print("  SECTION 3 — Per-dimension activation: which latent concepts fire?")
    print(SEP)
    for query, country in analysis_cases:
        try:
            pos, neg, lsa_sim = dimension_breakdown(engine, query, country, top_n=5)
        except ValueError as e:
            print(f"\n  [{query}] × [{country}]: {e}")
            continue

        print(f"\n  Query: \"{query}\"   Country: {country}   (LSA cosine = {lsa_sim:.4f})")
        print(f"  {'Dim':>4}  {'Var%':>5}  {'Top term':<22}  {'Q-coord':>8}  {'D-coord':>8}  {'Contribution':>12}")
        print("  " + "-"*68)
        print("  [POSITIVELY activated dimensions — matching concepts]")
        for b in pos:
            print(f"  {b['dim']:4d}  {b['var_pct']:5.2f}  {b['top_term']:<22}  "
                  f"{b['query_coord']:+8.4f}  {b['doc_coord']:+8.4f}  {b['contribution']:+12.5f}")
        print("  [NEGATIVELY activated dimensions — opposing concepts]")
        for b in neg:
            print(f"  {b['dim']:4d}  {b['var_pct']:5.2f}  {b['top_term']:<22}  "
                  f"{b['query_coord']:+8.4f}  {b['doc_coord']:+8.4f}  {b['contribution']:+12.5f}")


if __name__ == "__main__":
    main()
