"""Disjoint-set (Union-Find) implementation for track merging."""

from __future__ import annotations


class UnionFind:
    """Simple path-compressed union find."""

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}
        self.rank: dict[int, int] = {}

    def add(self, x: int) -> None:
        """Add one element if absent."""

        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: int) -> int:
        """Find representative with path compression."""

        px = self.parent[x]
        if px != x:
            self.parent[x] = self.find(px)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        """Union sets containing a and b."""

        self.add(a)
        self.add(b)
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
