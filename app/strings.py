"""Multilingual Support String Tables (Pidgin & French).

Status: Disabled by default. Needs native review.
"""

from typing import Dict, Any

STRING_TABLES: Dict[str, Dict[str, Any]] = {
    "pcm": {
        "lang": "pcm",
        "status": "disabled",
        "review_status": "needs native review",
        "notice": "Dis report na screening aid only. E no be engineering certification.",
        "corrosion_screening": "Corrosion screening percentage",
        "recommendation": "Wetin you suppose do"
    },
    "fr": {
        "lang": "fr",
        "status": "disabled",
        "review_status": "needs native review",
        "notice": "Ce rapport est une aide au criblage et ne constitue pas une certification d'ingenierie.",
        "corrosion_screening": "Pourcentage de criblage de corrosion",
        "recommendation": "Recommandation"
    }
}

def get_string(lang: str, key: str, default: str = "") -> str:
    table = STRING_TABLES.get(lang)
    if not table or table.get("status") == "disabled":
        return default
    return table.get(key, default)
