"""Paper-facing names for internal semi-synthetic configuration values."""

from __future__ import annotations

import json
from pathlib import Path

DISPLAY_NAMES_PATH = Path(__file__).with_name("semisynthetic_display_names.json")
with DISPLAY_NAMES_PATH.open() as display_names_file:
    DISPLAY_NAMES = json.load(display_names_file)


def display_field(field: str) -> str:
    """Return the paper-facing name of a configuration field."""
    return DISPLAY_NAMES["fields"].get(field, field)


def display_value(field: str, value) -> str:
    """Return the paper-facing name of one configuration value."""
    text = str(value)
    if field == "family":
        family, separator, burn_in = text.partition("/burn-in=")
        family_name = DISPLAY_NAMES["values"][field].get(family, family)
        return f"{family_name} (burn-in {burn_in})" if separator else family_name
    return DISPLAY_NAMES["values"].get(field, {}).get(text, text)


def display_assignment(field: str, value) -> str:
    """Format one configuration field/value pair for a plot label."""
    return f"{display_field(field)}: {display_value(field, value)}"


SERIES_SHORT_NAMES = DISPLAY_NAMES.get("series_short_names", [])


def match_series_name(row) -> str | None:
    """Name of the first series_short_names entry whose match pairs all hold for `row`.

    Fields a rule omits are wildcards, and a missing or NaN column stringifies to "nan",
    so a rule naming `oracle_variant` never matches an arm that has none.
    """
    for entry in SERIES_SHORT_NAMES:
        if all(str(row.get(field)) == value for field, value in entry["match"].items()):
            return entry["name"]
    return None


def name_series_short(df, identity_cols: list[str]):
    """Paper name per row; every arm must be named, and a name must determine the arm.

    `identity_cols` is what makes two rows the same arm -- the caller's hue columns. A
    field the match rules ignore (search_all_sizes, say) may not vary under one name, or
    two arms would draw as a single legend entry and silently merge curves.
    """
    names = df.apply(match_series_name, axis=1)
    unnamed = df.loc[names.isna(), identity_cols].drop_duplicates()
    assert unnamed.empty, (
        "no series_short_names entry matches:\n"
        f"{unnamed.to_string(index=False)}\n"
        "add one to semisynthetic_display_names.json, or pass --style diagnostic"
    )
    for name, group in df.groupby(names):
        varying = [col for col in identity_cols if group[col].astype(str).nunique() > 1]
        assert not varying, (
            f"paper name {name!r} covers rows that differ in {varying}; add {varying} "
            "to its match rule in semisynthetic_display_names.json"
        )
    return names


def order_series_names(names) -> list[str]:
    """The JSON order, filtered to the names present -- the legend order."""
    present = set(names)
    ordered = []
    for entry in SERIES_SHORT_NAMES:
        if entry["name"] in present and entry["name"] not in ordered:
            ordered.append(entry["name"])
    return ordered
