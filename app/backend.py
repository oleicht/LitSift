import json
import warnings
from functools import lru_cache
from pathlib import Path
import requests

import numpy as np
from numpy.linalg import norm
import openreview
import polars as pl
import voyageai
from tqdm import tqdm


config = json.load(open(Path(__file__).parent / "user.json"))

vo = voyageai.Client(api_key=config["voyageai"]["secret"])

client = openreview.api.OpenReviewClient(
    baseurl="https://api2.openreview.net",
    username=config["openreview"]["username"],
    password=config["openreview"]["password"],
)

OVERVIEW_FILE = Path(__file__).parent / "cache" / "overview.parquet"


def _venues_from_config() -> list[tuple[str, int, str]]:
    return [(v["venue"], v["year"], v["track"]) for v in config["openreview"]["venues"]]


def _venue_cache_dir(venue: str, year: int, track: str) -> Path:
    safe_venue = venue.replace(".", "").replace("/", "-")
    return Path(__file__).parent / "cache" / safe_venue / str(year) / track


@lru_cache()
def _load_overview() -> pl.DataFrame:
    return pl.read_parquet(OVERVIEW_FILE)


@lru_cache()
def get_data(venue: str, year: int, track: str) -> pl.DataFrame:
    df = _load_overview().filter(
        (pl.col("venue") == venue) & (pl.col("year") == year) & (pl.col("track") == track)
    )
    if df.is_empty():
        warnings.warn(
            f"No papers found for venue='{venue}', year={year}, track='{track}'. "
            "Check that this combination exists in the overview cache."
        )
    return df


def get_all_data() -> pl.DataFrame:
    return pl.concat([get_data(v, y, t) for v, y, t in _venues_from_config()])


def get_reviews(paper_id: str):
    return client.get_notes(forum=paper_id)


@lru_cache()
def generate_embeddings(venue: str, year: int, track: str) -> pl.DataFrame:
    cache_dir = _venue_cache_dir(venue, year, track)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_embeddings_file = cache_dir / f"{config['ranking']['model']}-embeddings.parquet"

    if cached_embeddings_file.exists():
        return pl.read_parquet(cached_embeddings_file)

    data = get_data(venue, year, track)
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
        vo_model_str=config["ranking"]["model"],
    )
    df.write_parquet(cached_embeddings_file)
    return df


def _generate_voyageai_embeddings_robustly(paper_strings, cached_embeddings_file, vo_model_str):
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
            assert disk_chunk["title_abstract"].to_list() == papers
            assert disk_chunk["range"].to_list() == list(range(*tb))
            continue
        try:
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
        except Exception as e:
            print(f"Problem with chunk {i}")
            print(e)

    chunks = list(cached_embeddings_file.parent.glob("chunk_*.parquet"))
    return (
        pl.concat([pl.read_parquet(c) for c in chunks])
        .sort("range")
        .drop("range")
        .rename({"title_abstract": "paper_strings"})
    )


def get_rankings(query: str) -> list[tuple[str, str]]:
    venues = _venues_from_config()
    embeddings = pl.concat([generate_embeddings(v, y, t) for v, y, t in venues])

    paper_strings = embeddings["paper_strings"]
    emb_cols = [c for c in embeddings.columns if c.startswith("emb")]
    x_embeddings = embeddings.select(emb_cols).to_numpy().astype(np.float32)
    latent_query = np.array(
        vo.embed([query], model=config["ranking"]["model"], input_type="query").embeddings,
        dtype=np.float32,
    )
    qk = (
        -latent_query
        @ x_embeddings.T
        / (norm(x_embeddings, axis=1) * norm(latent_query, axis=1))
    )[0]

    preferences = qk.argsort()
    results = []
    for s in paper_strings[preferences.tolist()]:
        if s is None:
            continue
        title = s.split(" Abstract: ")[0][len("Title: "):]
        abstract = s.split(" Abstract: ")[1].split(" TLDR: ")[0]
        results.append((title, abstract))
    return results


def download(paper_title: str) -> None:
    data = get_all_data()
    match = data.filter(pl.col("title") == paper_title)
    if match.is_empty():
        raise ValueError(f"Paper '{paper_title}' not found in any configured venue.")

    row = match.row(0, named=True)
    downloads_dir = _venue_cache_dir(row["venue"], row["year"], row["track"]) / "downloads"
    downloads_dir.mkdir(exist_ok=True)

    file_on_disk = downloads_dir / f"{paper_title}.pdf"
    if file_on_disk.exists():
        raise ValueError("Skip download. File already on disk")

    url = f"https://openreview.net/pdf?id={match['id'][0]}"
    response = requests.get(url)
    with open(file_on_disk, "wb") as f:
        f.write(response.content)
