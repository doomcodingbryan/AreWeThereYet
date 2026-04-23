"""
Centralized search configuration for the recommendation engine.

Keeping these values in one place makes it easier to tune behavior
without editing core ranking logic.
"""

SYNONYMS = {
    # Cost
    "cheap": ["affordable", "inexpensive", "low cost", "budget"],
    "affordable": ["cheap", "inexpensive", "low cost", "budget"],
    "expensive": ["high cost", "pricey", "costly"],
    "budget": ["cheap", "affordable", "low cost"],
    # Safety
    "safe": ["low crime", "peaceful", "secure", "safety"],
    "unsafe": ["high crime", "dangerous", "crime rate"],
    "dangerous": ["unsafe", "high crime", "crime"],
    # Weather / climate
    "warm": ["tropical", "hot", "sunny", "mild", "humid"],
    "hot": ["warm", "tropical", "sunny", "heat"],
    "sunny": ["warm", "tropical", "sunshine", "clear skies"],
    "tropical": ["warm", "humid", "hot", "beach", "rainforest"],
    "cold": ["cool", "nordic", "winter", "snow", "freezing"],
    "cool": ["mild", "temperate", "cold"],
    "mild": ["temperate", "moderate", "pleasant"],
    "beach": ["coastal", "ocean", "tropical", "seaside"],
    "snow": ["cold", "winter", "skiing", "alpine"],
    # Nature
    "nature": ["outdoors", "hiking", "mountains", "forests", "wildlife"],
    "hiking": ["mountains", "trails", "outdoors", "nature", "trekking"],
    "mountains": ["hiking", "alps", "alpine", "elevation", "skiing"],
    "ocean": ["sea", "beach", "coastal", "diving", "surfing"],
    # Lifestyle
    "nightlife": ["bars", "clubs", "social", "entertainment", "vibrant"],
    "food": ["cuisine", "restaurants", "culinary", "gastronomy"],
    "culture": ["arts", "history", "museums", "heritage", "traditions"],
    "expat": ["expatriate", "foreigner", "immigrant", "relocation", "abroad"],
    "remote work": ["digital nomad", "wifi", "coworking", "internet", "laptop"],
    "nomad": ["remote work", "digital nomad", "freelance", "coworking"],
    # English
    "english": ["english speaking", "anglophone", "english language"],
    # Healthcare
    "healthcare": ["medical", "hospitals", "health system", "doctors"],
    "medical": ["healthcare", "hospitals", "health care"],
    # Visa / immigration
    "visa": ["immigration", "residency", "work permit", "permit"],
    "retire": ["retirement", "pension", "retiree", "expat"],
    "immigrate": ["visa", "residency", "immigration", "move abroad"],
}

NOISY_TERMS = {
    "really", "thing", "things", "want", "wanted", "need", "needs", "good",
    "better", "best", "like", "just", "maybe", "term", "long", "plan", "plans",
    "country", "countries", "city", "lifestyle", "people", "family", "friendly",
}

COUNTRY_NAME_ALIASES = {
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

LANGUAGE_QUERY_ALIASES = {
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

INTENT_ANCHORS = {
    "safety": "safe secure safety security low_crime safety_index",
    "climate": "weather climate warm sunny mild pleasant climate_index",
    "cost": "affordable budget low_cost cheap inexpensive affordability cost_of_living_index",
}

DEFAULT_RANKING = {
    "n_components": 100,
    "hybrid_alpha": 0.35,
    "metadata_blend": 0.40,
    "stage2_candidate_k": 30,
    "qol_weight": 0.20,
    "score_ceiling": 0.38,
    "safety_intent_conf_threshold": 0.22,
    "safety_intent_share_threshold": 0.45,
    "safety_weight_partial_threshold": 0.38,
    "safety_floor_strict": 60.0,
    "safety_floor_partial": 45.0,
    "stage2_profile_w_default": 0.45,
    "stage2_meta_w_default": 0.30,
    "stage2_feature_total_default": 0.25,
    "stage2_profile_w_safety_boost": 0.35,
    "stage2_meta_w_safety_boost": 0.20,
    "stage2_feature_total_safety_boost": 0.45,
    "safety_weight_floor_safety_boost": 0.65,
    "safety_penalty_target": 0.65,
    "safety_penalty_scale": 1.6,
    "safety_penalty_min": 0.35,
}

