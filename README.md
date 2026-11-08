# Ani4D project page

Static site for **Ani4D: Video-Driven Skeletal Animation of Arbitrary 3D Assets**, served by GitHub Pages from this
branch's root (`index.html`, `media/`).

The page and its media are generated; do not edit them by hand. The sources live in the main project repository under
`scratchpad/project_page/`: `template.html` (layout, styles, scripts), `build_page.py` (writes `index.html` with the data
inlined), and `render/` (the Blender renders of every video: `make_compare.py` for the AniBench comparison grids,
`make_card.py` for the result cards, `make_showcase.py` for the reel, generated-motion and DAVIS clips).

The page carries `noindex` while the paper is under review.
