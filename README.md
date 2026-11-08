# Ani4D project page

Static site for **Ani4D: Video-Driven Skeletal Animation of Arbitrary 3D Assets**, served by GitHub Pages from this
branch's root (`index.html`, `media/`, `js/`).

The page and its media are generated; do not edit them by hand. The sources live in the main project repository under
`scratchpad/project_page/`: `template.html` (layout, styles, scripts), `build_page.py` (writes `index.html` with the data
inlined), and `render/` (the Blender renders of every video: `make_compare.py` for the AniBench comparison grids,
`make_card.py` for the result cards, `make_showcase.py` for the reel, generated-motion and DAVIS clips, `make_3d.py` for
`media/3d/`, the meshes the comparison player's drag-to-turn row draws, `make_gen_clips.py` for `media/gen/`, the
generated-action clips).

`js/turn3d.js` is that drag-to-turn row (WebGL); `js/three.module.min.js` and `js/OrbitControls.js` are three.js 0.170.0
(MIT), the controls' import path pointed at the local copy.

The page carries `noindex` while the paper is under review.
