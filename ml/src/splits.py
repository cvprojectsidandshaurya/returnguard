"""Split group logic. Shared by the split builder and the validator."""

from __future__ import annotations

import pandas as pd


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def split_groups(df: pd.DataFrame) -> pd.Series:
    """Map each garment_id to its split group id.

    Garments are joined when they share a design_id or a lookalike_group.
    The resulting connected component is the smallest unit that may be
    assigned to a split without leaking a design across train and test.
    """
    uf = _UnionFind()
    for garment_id in df["garment_id"].unique():
        uf.find(garment_id)

    for key in ("design_id", "lookalike_group"):
        if key not in df.columns:
            continue
        for value, group in df.groupby(key):
            if value is None or str(value).strip() == "" or pd.isna(value):
                continue
            members = group["garment_id"].unique()
            for other in members[1:]:
                uf.union(members[0], other)

    roots = {g: uf.find(g) for g in df["garment_id"].unique()}
    return pd.Series(roots, name="split_group")
