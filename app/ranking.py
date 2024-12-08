import json
from functools import lru_cache
from pathlib import Path
import requests

import numpy as np
from numpy.linalg import norm
import openreview
import pandas as pd
from tqdm import trange, tqdm


config = json.load(open(Path(__file__).parent / "user.json"))

if config["ranking"]["model"] == "voyage-large-2-instruct":
    import voyageai

    vo = voyageai.Client(api_key=config["voyageai"]["secret"])
elif config["ranking"]["model"] == "all-mpnet-base-v2":
    from sentence_transformers import SentenceTransformer

    pass
else:
    raise ValueError(config["ranking"]["model"])


@lru_cache()
def get_data():
    venue_str = config["openreview"]["venue"].replace(".", "").replace("/", "-")
    cached_data_file = Path(__file__).parent / "cache" / f"{venue_str}/papers.parquet"
    cached_data_file.parent.mkdir(exist_ok=True, parents=True)
    if cached_data_file.exists():
        return pd.read_parquet(cached_data_file)

    client = openreview.api.OpenReviewClient(
        baseurl="https://api2.openreview.net",
        username=config["openreview"]["username"],
        password=config["openreview"]["password"],
    )
    all_papers = client.get_all_notes(
        content={"venueid": config["openreview"]["venue"]}
    )
    fields = ["title", "authors", "abstract", "venue", "pdf"]  # ,"keywords",]
    extracted = [
        (paper.to_json()["id"],)
        + tuple(
            paper.to_json()["content"].get(f, {"value": ["n/a"]})["value"]
            for f in fields
        )
        for paper in all_papers
    ]

    df = pd.DataFrame(extracted, columns=["id"] + fields)
    df.to_parquet(cached_data_file)
    return df


@lru_cache()
def load_model(name):
    model_path = Path(__file__).parent / f"cache/models/{name}"
    try:
        model = SentenceTransformer(model_path)
    except:
        model = SentenceTransformer(name)
        model.save(str(model_path))
    return model


@lru_cache()
def generate_embeddings(model=None):
    # ToDo: parametrise the option to include authors+institution here
    venue_str = config["openreview"]["venue"].replace(".", "").replace("/", "-")
    cached_embeddings_file = (
        Path(__file__).parent
        / "cache"
        / f"{venue_str}/{config['ranking']['model']}-embeddings.parquet"
    )
    if cached_embeddings_file.exists():
        return pd.read_parquet(cached_embeddings_file)

    data = get_data()
    paper_strings = data.apply(
        lambda row: "Title: " + row.title + " Abstract: " + row.abstract, axis=1
    )

    if config["ranking"]["model"] == "voyage-large-2-instruct":
        df = generate_voyageai_embeddings_robustly(
            paper_strings, cached_embeddings_file
        )

    elif config["ranking"]["model"] == "all-mpnet-base-v2":
        embeddings = np.zeros((data.shape[0], 768), dtype=np.float32)

        for i in trange(paper_strings.shape[0]):
            embeddings[i, :] = model.encode(paper_strings.iloc[i])

        df = pd.DataFrame(embeddings)
        df["paper_strings"] = paper_strings

    df.to_parquet(cached_embeddings_file)
    return df


def generate_voyageai_embeddings_robustly(paper_strings, cached_embeddings_file):
    """Send small chunks to server and store them locally"""
    vo_model_str = "voyage-large-2-instruct"

    text_blocks = []
    rc = 0
    start = 0
    for i, ps in enumerate(paper_strings.tolist()):
        rc += vo.count_tokens([ps], model=vo_model_str)
        if rc > 5_000:
            rc = vo.count_tokens([ps], model=vo_model_str)
            text_blocks += [(start, i)]
            start = i
    text_blocks += [(start, len(paper_strings))]

    for i, tb in tqdm(enumerate(text_blocks), total=len(text_blocks)):
        chunk_name = cached_embeddings_file.parent / f"chunk_{i}.parquet"
        papers = paper_strings.tolist()[slice(*tb)]
        if chunk_name.exists():
            disk_chunk = pd.read_parquet(chunk_name)
            assert disk_chunk["title_abstract"].tolist() == papers
            assert disk_chunk["range"].tolist() == list(range(*tb))
            continue
        try:
            res = vo.embed(papers, model=vo_model_str, input_type="document")
            # result = vo.embed(texts, model=vo_model_str, input_type="query")
            base = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "range": range(*tb),
                            "title_abstract": papers,
                        }
                    ),
                    pd.DataFrame(
                        np.array(res.embeddings, dtype=np.float32),
                        columns=[f"emb{jj}" for jj in range(1024)],
                    ),
                ],
                axis=1,
            )
            base.to_parquet(chunk_name)
        except Exception as e:
            print(f"Problem with {i}")
            print(e)

    chunks = list(cached_embeddings_file.parent.glob("chunk_*.parquet"))
    dfc = pd.concat([pd.read_parquet(c) for c in chunks])
    return (
        dfc.set_index("range")
        .sort_index()
        .rename(columns={"title_abstract": "paper_strings"})
    )


def get_rankings(query):
    if config["ranking"]["model"] == "all-mpnet-base-v2":
        model = load_model(config["ranking"]["model"])
        embeddings = generate_embeddings(model)
        paper_strings = embeddings[["paper_strings"]]
        x_embeddings = embeddings[
            [c for c in embeddings.columns if c != "paper_strings"]
        ].to_numpy()
        latent_query = model.encode([query])
        qk = -model.similarity(latent_query, x_embeddings).numpy()[0]

    elif config["ranking"]["model"] == "voyage-large-2-instruct":
        embeddings = generate_embeddings()
        x_embeddings = embeddings[
            [c for c in embeddings.columns if c != "paper_strings"]
        ].to_numpy()
        paper_strings = embeddings[["paper_strings"]]
        latent_query = np.array(
            vo.embed(
                [query], model="voyage-large-2-instruct", input_type="query"
            ).embeddings,
            dtype=np.float32,
        )
        # note: empirically, the voyage-ai embeddings seems to have norm=1
        qk = (
            -latent_query
            @ x_embeddings.T
            / (norm(x_embeddings, axis=1) * norm(latent_query, axis=1))
        )[0]

    preferences = qk.argsort()
    return [
        (
            row.paper_strings.split(" Abstract: ")[0][len("Title: ") :],
            row.paper_strings.split(" Abstract: ")[1],
        )
        for row in paper_strings.iloc[preferences].itertuples()
        if row.paper_strings is not None
    ]


def download(paper_title):
    venue_str = config["openreview"]["venue"].replace(".", "").replace("/", "-")
    cached_data_file = Path(__file__).parent / "cache" / f"{venue_str}/downloads"
    cached_data_file.mkdir(exist_ok=True)
    file_on_disk = cached_data_file / f"""{paper_title}.pdf"""
    if file_on_disk.exists():
        raise ValueError("Skip download. File already on disk")

    data = get_data()
    match = data.query(f"title=='''{paper_title}'''")
    assert match.shape[0] == 1, match
    id_ = match["id"].iloc[0]
    url = f"https://openreview.net/pdf?id={id_}"
    response = requests.get(url)

    with open(file_on_disk, "wb") as f:
        f.write(response.content)


if __name__ == "__main__":
    df = get_data()
    # get_rankings("Test")
    # download("Rolling Diffusion Models")
