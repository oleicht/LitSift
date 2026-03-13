from dataclasses import dataclass


@dataclass(frozen=True)
class Paper:
    title: str
    abstract: str
    tldr: str = "n/a"
    authors: tuple[str, ...] = ()
    name: str = ""      # e.g. "ICML 2023 Poster"
    score: float = 0.0
    id: str = ""
