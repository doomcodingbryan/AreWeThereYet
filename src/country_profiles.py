"""
Country Profiles where we aggregate Reddit posts into per-country text documents.

Each country gets a single text profile built from all posts that mention it.
To better reflect the reliability and visibility of different experiences, posts with higher social engagement
(score, upvotes, comments) are repeated more times in the profile, so they have a stronger influence on the TF-IDF results.
"""

import ast
import json
import math
import os


def _parse_countries(raw):
    """Parse the countries field, because reddit json data is messy"""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if raw == "" or raw == "[]":
            return []
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return []


def _social_weight(post):
    """
    Compute a multiplier based on social signals.
    Higher scored posts get repeated so they weigh more in TF-IDF.
    """
    score = max(post.get("score", 0) or 0, 0)
    ratio = post.get("upvote_ratio", 0.5) or 0.5
    comments = max(post.get("num_comments", 0) or 0, 0)

    engagement = score * ratio + math.log1p(comments)

    if engagement >= 50:
        return 3
    elif engagement >= 10:
        return 2
    return 1


def build_country_profiles(json_path=None):
    """
    Load reddit_sample.json and aggregate posts into per-country text profiles.
    """
    if json_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        json_path = os.path.join(project_root, "data", "reddit_sample.json")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    profiles = {}

    for post_id, post in data.items():
        if post.get("has_country") != "True":
            continue

        countries = _parse_countries(post.get("countries", []))
        if not countries:
            continue

        text = post.get("full_text") or post.get("body") or post.get("title", "")
        if not text or len(text.strip()) < 50:
            continue

        weight = _social_weight(post)

        for country in countries:
            country = country.strip()
            if not country:
                continue
            if country not in profiles:
                profiles[country] = []
            for _ in range(weight):
                profiles[country].append(text)

    country_documents = {}
    for country, texts in profiles.items():
        country_documents[country] = "\n\n".join(texts)

    print(f"Built profiles for {len(country_documents)} countries "
          f"from {len(data)} posts")

    return country_documents
