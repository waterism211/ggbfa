"""The four confirmatory members of gg-BFA."""

from typing import Tuple


MEMBERS: Tuple[str, ...] = ("1f", "2f+s", "2f", "mf")

_DESCRIPTIONS = {
    "1f": "one factor per variable, loadings fixed at the community membership",
    "2f+s": "at most one cross-loading per row, at most a fraction of each community",
    "2f": "at most one cross-loading per row",
    "mf": "spike-and-slab cross-loadings on every other factor",
}


def normalize_member(value: str) -> str:
    """Return the member name, accepting surrounding spaces and upper case."""

    if not isinstance(value, str):
        raise TypeError("member must be a string")
    member = value.strip().lower()
    if member not in MEMBERS:
        raise ValueError(f"member must be one of: {', '.join(MEMBERS)}")
    return member


def member_label(value: str) -> str:
    """Return the label used in the paper, e.g. ``gg-BFA (2f+s)``."""

    return f"gg-BFA ({normalize_member(value)})"


__all__ = ["MEMBERS", "member_label", "normalize_member"]
