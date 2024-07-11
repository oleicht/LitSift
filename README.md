## Literature sifting

Simple app that ranks conference papers by relevance to a given query. 

### Setup:
- create [openreview](https://openreview.net/) account
- create [voyageai](https://www.voyageai.com/) account & add billing information (make sure you understand the pricing. For my usage of this app for a single conference I'm well within the 50mn free token limit. You need the billing information so you don't get rate limited when initialising the paper embeddings.)
- clone repo, put secrets in `example_user.json` and rename to `user.json`
- install code
- run app via `lit-sift` in command line
- enter your query and click 'Submit'. The first time you do this, it will take a few minutes to download the papers and create embeddings. Progress bars can be see in the command line
- from then on, it is only a few seconds for each ranking

### Navigation:
- you can navigate the title list with up/down arrows
- hitting enter (or double clicking) on a title, a pop-up window with the abstract appears
- you can close the pop-up with Esc
- or download the paper via the button, papers are stored in app/cache/{conference}/downloads
