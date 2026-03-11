import json
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


@lru_cache()
def get_data():
    venue_str = config["openreview"]["venue"].replace(".", "").replace("/", "-")
    cached_data_file = Path(__file__).parent / "cache" / f"{venue_str}/papers.parquet"
    cached_data_file.parent.mkdir(exist_ok=True, parents=True)
    if cached_data_file.exists():
        return pl.read_parquet(cached_data_file)

    all_papers = client.get_all_notes(
        content={"venueid": config["openreview"]["venue"]}
    )
    fields = ["title", "authors", "abstract", "venue", "pdf"]
    extracted = [
        (paper.to_json()["id"],)
        + tuple(
            paper.to_json()["content"].get(f, {"value": ["n/a"]})["value"]
            for f in fields
        )
        for paper in all_papers
    ]

    df = pl.DataFrame(extracted, schema=["id"] + fields, orient="row")
    df.write_parquet(cached_data_file)
    return df


def get_reviews(paper_id: str):
    return client.get_notes(forum=paper_id)


@lru_cache()
def generate_embeddings():
    # ToDo: parametrise the option to include authors+institution here
    venue_str = config["openreview"]["venue"].replace(".", "").replace("/", "-")
    cached_embeddings_file = (
        Path(__file__).parent
        / "cache"
        / f"{venue_str}/{config['ranking']['model']}-embeddings.parquet"
    )
    if cached_embeddings_file.exists():
        return pl.read_parquet(cached_embeddings_file)

    data = get_data()
    paper_strings = [
        f"Title: {t} Abstract: {a}"
        for t, a in zip(data["title"].to_list(), data["abstract"].to_list())
    ]
    df = generate_voyageai_embeddings_robustly(
        paper_strings,
        cached_embeddings_file,
        vo_model_str=config["ranking"]["model"],
    )
    df.write_parquet(cached_embeddings_file)
    return df


def generate_voyageai_embeddings_robustly(
    paper_strings, cached_embeddings_file, vo_model_str
):
    """Send small chunks to server and store them locally"""

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
            print(f"Problem with {i}")
            print(e)

    chunks = list(cached_embeddings_file.parent.glob("chunk_*.parquet"))
    return (
        pl.concat([pl.read_parquet(c) for c in chunks])
        .sort("range")
        .drop("range")
        .rename({"title_abstract": "paper_strings"})
    )


def get_rankings(query):
    embeddings = generate_embeddings()
    paper_strings = embeddings["paper_strings"]
    emb_cols = [c for c in embeddings.columns if c.startswith("emb")]
    x_embeddings = embeddings.select(emb_cols).to_numpy().astype(np.float32)
    latent_query = np.array(
        vo.embed(
            [query], model=config["ranking"]["model"], input_type="query"
        ).embeddings,
        dtype=np.float32,
    )
    qk = (
        -latent_query
        @ x_embeddings.T
        / (norm(x_embeddings, axis=1) * norm(latent_query, axis=1))
    )[0]

    preferences = qk.argsort()
    return [
        (
            s.split(" Abstract: ")[0][len("Title: "):],
            s.split(" Abstract: ")[1],
        )
        for s in paper_strings[preferences.tolist()]
        if s is not None
    ]


def download(paper_title):
    venue_str = config["openreview"]["venue"].replace(".", "").replace("/", "-")
    cached_data_file = Path(__file__).parent / "cache" / f"{venue_str}/downloads"
    cached_data_file.mkdir(exist_ok=True)
    file_on_disk = cached_data_file / f"""{paper_title}.pdf"""
    if file_on_disk.exists():
        raise ValueError("Skip download. File already on disk")

    data = get_data()
    match = data.filter(pl.col("title") == paper_title)
    assert len(match) == 1, match
    url = f"https://openreview.net/pdf?id={match['id'][0]}"
    response = requests.get(url)

    with open(file_on_disk, "wb") as f:
        f.write(response.content)


if __name__ == "__main__":
    df = get_data()
    # get_rankings("Test")
    # download("Rolling Diffusion Models")
