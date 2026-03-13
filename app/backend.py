import json
import logging
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import requests
from requests.exceptions import ConnectionError, Timeout

import numpy as np
from numpy.linalg import norm
import openreview
import polars as pl
import voyageai
from tqdm import tqdm


config = json.loads((Path(__file__).parent / "user.json").read_text())

vo = voyageai.Client(api_key=config["voyageai"]["secret"])

client = openreview.api.OpenReviewClient(
    baseurl="https://api2.openreview.net",
    username=config["openreview"]["username"],
    password=config["openreview"]["password"],
)

# for older conferences, need to use the apiv1
# it needs be called this way here
# legacy_client = openreview.Client(
#         baseurl="https://api.openreview.net",
#         username=config["openreview"]["username"],
#         password=config["openreview"]["password"],
# )

OVERVIEW_FILE = Path(__file__).parent / "cache" / "overview.parquet"
_LOG_FILE = Path(__file__).parent.parent / "scraping_logs.txt"
_EMBEDDING_MODEL = "voyage-4"
_FILENAME_UNSAFE = re.compile(r"[^\w\s\-.]")

_CONFERENCES = [
    "ICML.cc",
    "NeurIPS.cc",
    "ICLR.cc",
    "robot-learning.org/CoRL",
    "rl-conference.cc/RLC",
    "aclweb.org/ACL",
    "EMNLP",
    "AAAI.org",
]


@dataclass
class _Venue:
    name: str
    year: int
    workshops: list[str] | None = None


def _scrape_log() -> logging.Logger:
    logger = logging.getLogger("litsift.scraping")
    if not logger.handlers:
        handler = logging.FileHandler(_LOG_FILE)
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _safe_get(content: dict, field: str):
    if field in content:
        return content[field]["value"]
    if field == "TLDR":
        return "n/a"
    if field == "authors":
        return ["n/a"]
    if field == "keywords":
        return ["n/a"]
    if field == "pdf":
        _scrape_log().warning(
            "Missing pdf in: %s", content.get("venue", {}).get("value", "?")
        )
        return "n/a"
    if field == "abstract":
        _scrape_log().warning(
            "Missing abstract in: %s", content.get("venue", {}).get("value", "?")
        )
        return "n/a"
    raise ValueError(f"Key {field} unexpectedly not in content")


def _paper_to_row(paper) -> tuple:
    fields = ["title", "authors", "abstract", "venue", "pdf", "TLDR", "keywords"]
    paper_json = paper.to_json()
    content = paper_json["content"]
    return (paper_json["id"],) + tuple(_safe_get(content, f) for f in fields)


def _fetch_notes(venueid: str) -> list:
    try:
        return client.get_all_notes(content={"venueid": venueid})
    except (Timeout, ConnectionError) as e:
        _scrape_log().warning("Request failed for %s: %s", venueid, e)
        return []


def _discover_venues() -> list[_Venue]:
    venues = []
    for prefix in _CONFERENCES:
        groups = client.get_all_groups(prefix=prefix)
        conference_years: set[str] = set()
        workshops: dict[str, set[str]] = defaultdict(set)
        for g in groups:
            ws_match = re.findall(
                rf"{re.escape(prefix)}/(20\d\d)/Workshop/([^/]*)", g.id
            )
            conf_match = re.findall(rf"{re.escape(prefix)}/(20\d\d)/Conference", g.id)
            if ws_match:
                year, name = ws_match[0]
                workshops[year].add(name)
            elif conf_match:
                conference_years.add(conf_match[0])
        sorted_workshops = {k: sorted(v) for k, v in workshops.items()}
        for year in conference_years:
            venues.append(
                _Venue(
                    name=prefix, year=int(year), workshops=sorted_workshops.get(year)
                )
            )
    return venues


def _build_overview() -> pl.DataFrame:
    log = _scrape_log()
    venues = _discover_venues()
    log.info("Discovered %d venue/year combinations", len(venues))
    rows = []
    for venue in venues:
        main = _fetch_notes(f"{venue.name}/{venue.year}/Conference")
        if not main:
            log.info("No papers found in %s/%s", venue.name, venue.year)
            continue
        pk = (venue.name, venue.year, "conference")
        rows.extend(pk + _paper_to_row(paper) for paper in main)
        log.info(
            "Fetched %d papers from %s/%s conference", len(main), venue.name, venue.year
        )

        workshop_notes = []
        if venue.workshops:
            for ws in venue.workshops:
                notes = _fetch_notes(f"{venue.name}/{venue.year}/Workshop/{ws}")
                workshop_notes.extend(notes)
        if workshop_notes:
            pk = (venue.name, venue.year, "workshop")
            rows.extend(pk + _paper_to_row(paper) for paper in workshop_notes)
            log.info(
                "Fetched %d workshop papers from %s/%s",
                len(workshop_notes),
                venue.name,
                venue.year,
            )

    return pl.DataFrame(
        rows,
        schema=[
            "venue",
            "year",
            "track",
            "id",
            "title",
            "authors",
            "abstract",
            "name",
            "link",
            "tldr",
            "keywords",
        ],
        orient="row",
    )


def get_available_venues() -> list[tuple[str, int, str, int]]:
    """Return all (venue, year, track, paper_count) combos present in the overview."""
    counts = (
        _load_overview()
        .group_by(["venue", "year", "track"])
        .len()
        .sort(["venue", "year", "track"])
    )
    return [
        (r["venue"], r["year"], r["track"], r["len"])
        for r in counts.iter_rows(named=True)
    ]


_selected_venues: list[tuple[str, int, str]] | None = None


def get_selected_venues() -> list[tuple[str, int, str]]:
    """Return the current session's venue selection, defaulting to all available venues."""
    global _selected_venues
    if _selected_venues is None:
        _selected_venues = [(r[0], r[1], r[2]) for r in get_available_venues()]
    return _selected_venues


def set_selected_venues(venues: list[tuple[str, int, str]]) -> None:
    global _selected_venues
    _selected_venues = list(venues)


def _venue_cache_dir(venue: str, year: int, track: str) -> Path:
    safe_venue = venue.replace(".", "").replace("/", "-")
    return Path(__file__).parent / "cache" / safe_venue / str(year) / track


@lru_cache()
def _load_overview() -> pl.DataFrame:
    if not OVERVIEW_FILE.exists():
        log = _scrape_log()
        log.info("Overview file not found — scraping from OpenReview")
        OVERVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)
        df = _build_overview()
        df.write_parquet(OVERVIEW_FILE)
        log.info("Overview written to %s (%d papers)", OVERVIEW_FILE, len(df))
        return df
    return pl.read_parquet(OVERVIEW_FILE)


@lru_cache()
def get_data(venue: str, year: int, track: str) -> pl.DataFrame:
    df = _load_overview().filter(
        (pl.col("venue") == venue)
        & (pl.col("year") == year)
        & (pl.col("track") == track)
    )
    if df.is_empty():
        warnings.warn(
            f"No papers found for venue='{venue}', year={year}, track='{track}'. "
            "Check that this combination exists in the overview cache."
        )
    return df


def get_all_data() -> pl.DataFrame:
    return pl.concat([get_data(v, y, t) for v, y, t in get_selected_venues()])


def get_reviews(paper_id: str):
    return client.get_notes(forum=paper_id)


@lru_cache()
def generate_embeddings(venue: str, year: int, track: str) -> pl.DataFrame:
    cache_dir = _venue_cache_dir(venue, year, track)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_embeddings_file = cache_dir / f"{_EMBEDDING_MODEL}-embeddings.parquet"
    data = get_data(venue, year, track)

    if cached_embeddings_file.exists():
        df = pl.read_parquet(cached_embeddings_file)
        if "id" not in df.columns:
            # Migrate: add paper IDs by position — safe here because the embeddings
            # were built from this exact data in the same function.
            df = df.with_columns(pl.Series("id", data["id"].to_list()))
            df.write_parquet(cached_embeddings_file)
        return df

    paper_strings = [
        f"Title: {t} Abstract: {a} TLDR: {tldr} Keywords: {', '.join(kws)}"
        for t, a, tldr, kws in zip(
            data["title"].to_list(),
            data["abstract"].to_list(),
            data["tldr"].to_list(),
            data["keywords"].to_list(),
        )
    ]
    df = _generate_voyageai_embeddings_robustly(
        paper_strings,
        cached_embeddings_file,
        vo_model_str=_EMBEDDING_MODEL,
    )
    df = df.with_columns(pl.Series("id", data["id"].to_list()))
    df.write_parquet(cached_embeddings_file)
    return df


def _generate_voyageai_embeddings_robustly(
    paper_strings, cached_embeddings_file, vo_model_str
):
    """Send small chunks to the VoyageAI API and store them locally as chunk files."""
    text_blocks = []
    rc = 0
    start = 0
    for i, ps in enumerate(paper_strings):
        rc += vo.count_tokens([ps], model=vo_model_str)
        if rc > 5_000:
            rc = vo.count_tokens([ps], model=vo_model_str)
            text_blocks += [(start, i)]
            start = i
    text_blocks += [(start, len(paper_strings))]

    for i, tb in tqdm(enumerate(text_blocks), total=len(text_blocks)):
        chunk_name = cached_embeddings_file.parent / f"chunk_{i}.parquet"
        papers = paper_strings[slice(*tb)]
        if chunk_name.exists():
            disk_chunk = pl.read_parquet(chunk_name)
            if disk_chunk["title_abstract"].to_list() != papers:
                raise ValueError(
                    f"Chunk {i} content mismatch — delete the cache directory and retry"
                )
            if disk_chunk["range"].to_list() != list(range(*tb)):
                raise ValueError(
                    f"Chunk {i} range mismatch — delete the cache directory and retry"
                )
            continue
        res = vo.embed(papers, model=vo_model_str, input_type="document")
        base = pl.DataFrame(
            {"range": list(range(*tb)), "title_abstract": papers}
        ).hstack(
            pl.from_numpy(
                np.array(res.embeddings, dtype=np.float32),
                schema=[f"emb{jj}" for jj in range(1024)],
            )
        )
        base.write_parquet(chunk_name)

    chunk_files = sorted(
        cached_embeddings_file.parent.glob("chunk_*.parquet"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if len(chunk_files) != len(text_blocks):
        raise RuntimeError(
            f"Expected {len(text_blocks)} embedding chunks but found {len(chunk_files)}. "
            "Delete the cache directory and retry."
        )
    return (
        pl.concat([pl.read_parquet(c) for c in chunk_files])
        .sort("range")
        .drop("range")
        .rename({"title_abstract": "paper_strings"})
    )


def get_rankings(query: str) -> list[tuple]:
    venues = get_selected_venues()
    embeddings = pl.concat([generate_embeddings(v, y, t) for v, y, t in venues])
    data = get_all_data()

    emb_cols = [c for c in embeddings.columns if c.startswith("emb")]
    x_embeddings = embeddings.select(emb_cols).to_numpy().astype(np.float32)
    latent_query = np.array(
        vo.embed([query], model=_EMBEDDING_MODEL, input_type="query").embeddings,
        dtype=np.float32,
    )
    scores = (
        latent_query
        @ x_embeddings.T
        / (norm(x_embeddings, axis=1) * norm(latent_query, axis=1))
    )[0]

    score_df = pl.DataFrame(
        {"id": embeddings["id"].to_list(), "score": scores.tolist()}
    )
    merged = data.join(score_df, on="id", how="inner").sort("score", descending=True)

    return [
        (
            row["title"],
            row["abstract"],
            row["tldr"],
            tuple(row["authors"]),
            row["name"],
            float(row["score"]),
            row["id"],
        )
        for row in merged.iter_rows(named=True)
    ]


def download(paper_title: str) -> None:
    data = get_all_data()
    match = data.filter(pl.col("title") == paper_title)
    if match.is_empty():
        raise ValueError(f"Paper '{paper_title}' not found in any configured venue.")

    downloads_dir = Path(__file__).parent.parent / "downloads"
    downloads_dir.mkdir(exist_ok=True)

    safe_title = _FILENAME_UNSAFE.sub("", paper_title).strip()[:200]
    file_on_disk = downloads_dir / f"{safe_title}.pdf"
    if file_on_disk.exists():
        return

    url = f"https://openreview.net/pdf?id={match['id'][0]}"
    response = requests.get(url)
    response.raise_for_status()
    with open(file_on_disk, "wb") as f:
        f.write(response.content)
