# Sifting through ML conference papers

Simple app that allows to interact with conference papers. At its heart, it relies on ranking papers by queries given by the user.

## Setup

- create [openreview](https://openreview.net/) account
- create [voyageai](https://www.voyageai.com/) account & add billing information. Make sure you understand the [pricing](https://docs.voyageai.com/docs/pricing).
- clone repo, put secrets in `example_user.json` and rename to `user.json`, install the code
- run app via `lit-sift` in command line
- enter your query and click 'Submit'. The first time you do this, it will take a few minutes conference (!) to download the abstract and create the embeddings. Progress bars can be see in the command line
- from then on, it is only a few seconds for each ranking

## Navigation

- navigate the papers with right/left arrows
- download short-cut is `d`, papers are stored in ./downloads
- view review/rebuttal via shortcut `r` or the review button. The pop-up can be closed with `Esc`
