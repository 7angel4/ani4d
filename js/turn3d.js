// Ani4D project page: outputs drawn live with WebGL, so viewers can turn them. One canvas holds one viewport per shown
// method (one row, or two in the comparison player's edge-stretch view: the input camera's view above, the turnable view
// below), all sharing one orbit camera. Data (render/make_3d.py): media/3d/<id>.bin holds the input mesh's faces and
// colours, the midpoint subdivision the renders use, and the ground truth; media/3d/<id>_v<kk>.bin holds the five
// outputs for camera kk. The look follows the Blender renders (render/bl_video.py): three lights placed relative to the
// camera and a floor that only carries the shadow.
//
// Edge stretch: the input mesh is cut into surface patches about STRETCH.scale of its size (the vertices of a grid cell,
// split into the pieces connected inside the cell, so a patch never joins two parts that merely lie close, such as a hand
// resting on a thigh); for every pair of neighbouring patches, how much more or less an output stretches the distance
// between them than the ground truth does, |d_out - d_gt| / d_rest. A vertex takes its patch's mean, softened over its
// mesh neighbours. At that scale the colour shows parts that bend, stretch or are dragged along, not the sub-millimetre
// jitter that single edges of a dense mesh pick up. It is drawn over the asset's own colours, desaturated and lightened:
// nothing below STRETCH.lo, then yellow, orange and red, opaque from STRETCH.full and darkest from STRETCH.hi.
import * as THREE from "./three.module.min.js";
import { OrbitControls } from "./OrbitControls.js";

export const STRETCH = {
  scale: 0.05, lo: 0.04, full: 0.2, hi: 0.45,
  stops: [[0.04, [255, 214, 102]], [0.1, [253, 141, 60]], [0.2, [222, 33, 38]], [0.45, [120, 0, 40]]],
};
const BG = 0xf2f3f5;
const FX = 50 / 36;                                  // the renders' 50 mm lens on a 36 mm sensor
const FOV = 2 * Math.atan(18 / 50) * 180 / Math.PI;
const lin = v => (v /= 255) <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
const TOLIN = new Float32Array(256);                 // sRGB byte -> linear
for (let i = 0; i < 256; i++) TOLIN[i] = lin(i);
const NS = 1024, HEAT = new Float32Array(NS * 4);    // stretch 0 .. STRETCH.hi -> sRGB heat colour (0..1) and opacity
for (let i = 0; i < NS; i++) {
  const s = i / (NS - 1) * STRETCH.hi, st = STRETCH.stops;
  let k = 0;
  while (k < st.length - 2 && s > st[k + 1][0]) k++;
  const f = Math.max(0, Math.min(1, (s - st[k][0]) / (st[k + 1][0] - st[k][0])));
  for (let c = 0; c < 3; c++) HEAT[4 * i + c] = (st[k][1][c] * (1 - f) + st[k + 1][1][c] * f) / 255;
  HEAT[4 * i + 3] = s <= STRETCH.lo ? 0 : Math.pow(Math.min(1, (s - STRETCH.lo) / (STRETCH.full - STRETCH.lo)), 0.75);
}

async function fetchBin(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  let ab = await r.arrayBuffer(), u8 = new Uint8Array(ab);
  if (u8[0] === 0x1f && u8[1] === 0x8b) {
    ab = await new Response(new Blob([u8]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
    u8 = new Uint8Array(ab);
  }
  if (String.fromCharCode(u8[0], u8[1], u8[2], u8[3]) !== "A4D1") throw new Error(`${url}: not an A4D1 file`);
  const hl = new DataView(ab).getUint32(4, true);
  const hdr = JSON.parse(new TextDecoder().decode(u8.subarray(8, 8 + hl)));
  const base = 8 + hl, blobs = {};
  for (const b of hdr.blobs) {
    const n = b.shape.reduce((a, x) => a * x, 1), off = base + b.off;
    const T = { u1: Uint8Array, u2: Uint16Array, u4: Uint32Array, i2: Int16Array, f4: Float32Array }[b.dtype];
    blobs[b.name] = Object.assign({ a: new T(ab, off, n) }, b);
  }
  return { hdr, blobs };
}

// one method's 16 frames (make_3d.encode_positions, undone): [Float32Array(3 n) per frame], xyz interleaved
function decodePositions(b, T, origin, step) {
  const N = b.n, S = 3 * N, M = T * S, by = b.a, r = new Int32Array(M);
  if (b.enc === "i2") for (let i = 0; i < M; i++) r[i] = ((by[i] | (by[M + i] << 8)) << 16) >> 16;
  else for (let i = 0; i < M; i++) r[i] = by[i] | (by[M + i] << 8) | (by[2 * M + i] << 16) | (by[3 * M + i] << 24);
  for (let i = 0; i < S && T > 1; i++) r[S + i] += r[i];
  for (let t = 2; t < T; t++) { const o = t * S; for (let i = 0; i < S; i++) r[o + i] += 2 * r[o - S + i] - r[o - 2 * S + i]; }
  const out = [];
  for (let t = 0; t < T; t++) {
    const P = new Float32Array(S), o = t * S;
    for (let a = 0; a < 3; a++) { const oa = o + a * N, g = origin[a]; for (let v = 0; v < N; v++) P[3 * v + a] = g + r[oa + v] * step; }
    out.push(P);
  }
  return out;
}

// the surface patches of the header: label per vertex, a representative vertex per patch (nearest its centroid), and
// the pairs of patches a mesh edge joins
function patchGraph(V0, n, F0, h) {
  const inv = 1 / h, key = new Float64Array(n), nf = F0.length / 3;
  for (let i = 0; i < n; i++) {
    key[i] = (Math.floor(V0[3 * i] * inv) + 4096) * 67108864 + (Math.floor(V0[3 * i + 1] * inv) + 4096) * 8192
      + (Math.floor(V0[3 * i + 2] * inv) + 4096);
  }
  const par = new Int32Array(n);
  for (let i = 0; i < n; i++) par[i] = i;
  const find = i => { while (par[i] !== i) { par[i] = par[par[i]]; i = par[i]; } return i; };
  for (let f = 0; f < nf; f++) for (let e = 0; e < 3; e++) {
    const a = F0[3 * f + e], b = F0[3 * f + (e + 1) % 3];
    if (key[a] === key[b]) { const ra = find(a), rb = find(b); if (ra !== rb) par[ra] = rb; }
  }
  const lab = new Int32Array(n), ids = new Map();
  let nc = 0;
  for (let i = 0; i < n; i++) { const r = find(i); let l = ids.get(r); if (l === undefined) { l = nc++; ids.set(r, l); } lab[i] = l; }
  const cen = new Float64Array(3 * nc), cnt = new Int32Array(nc);
  for (let i = 0; i < n; i++) { const l = lab[i]; for (let a = 0; a < 3; a++) cen[3 * l + a] += V0[3 * i + a]; cnt[l]++; }
  for (let l = 0; l < nc; l++) for (let a = 0; a < 3; a++) cen[3 * l + a] /= cnt[l];
  const rep = new Int32Array(nc), best = new Float64Array(nc).fill(Infinity);
  for (let i = 0; i < n; i++) {
    const l = lab[i], d = (V0[3 * i] - cen[3 * l]) ** 2 + (V0[3 * i + 1] - cen[3 * l + 1]) ** 2 + (V0[3 * i + 2] - cen[3 * l + 2]) ** 2;
    if (d < best[l]) { best[l] = d; rep[l] = i; }
  }
  const seen = new Set(), ce = [];
  for (let f = 0; f < nf; f++) for (let e = 0; e < 3; e++) {
    const la = lab[F0[3 * f + e]], lb = lab[F0[3 * f + (e + 1) % 3]];
    if (la === lb) continue;
    const k = la < lb ? la * nc + lb : lb * nc + la;
    if (!seen.has(k)) { seen.add(k); ce.push(la, lb); }
  }
  return { lab, rep, ce: Int32Array.from(ce), nc };
}

function camAxes(az, el) {                           // vidlib.cam_axes
  const a = az * Math.PI / 180, e = el * Math.PI / 180;
  const fwd = new THREE.Vector3(-Math.cos(e) * Math.cos(a), -Math.cos(e) * Math.sin(a), -Math.sin(e));
  const right = new THREE.Vector3(-Math.sin(a), Math.cos(a), 0);
  return { fwd, right, up: new THREE.Vector3().crossVectors(right, fwd) };
}

function frameCamera(P, az, el, margin = 0.07) {     // vidlib.frame_camera: target and distance that fit every point
  const { fwd, right, up } = camAxes(az, el), n = P.length / 3, c = new THREE.Vector3();
  for (let i = 0; i < n; i++) c.x += P[3 * i], c.y += P[3 * i + 1], c.z += P[3 * i + 2];
  c.multiplyScalar(1 / n);
  const proj = (ax, i) => (P[3 * i] - c.x) * ax.x + (P[3 * i + 1] - c.y) * ax.y + (P[3 * i + 2] - c.z) * ax.z;
  for (let it = 0; it < 4; it++) {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < n; i++) { const x = proj(right, i), y = proj(up, i); x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    c.addScaledVector(right, 0.5 * (x0 + x1)).addScaledVector(up, 0.5 * (y0 + y1));
  }
  let need = 0;
  for (let i = 0; i < n; i++) {
    const x = Math.abs(proj(right, i)), y = Math.abs(proj(up, i)), z = proj(fwd, i);
    need = Math.max(need, Math.max(x, y) * 2 * FX / (1 - 2 * margin) - z);
  }
  return { c, d: need, fwd };
}

export class Turn3D {
  // home: the view a load and a double-click go to ("turn" = 55 degrees from the input camera, "focus" = opts.focus)
  constructor(canvas, { root = "media/3d/", onState = () => {}, home = "turn", focus = null } = {}) {
    this.canvas = canvas; this.root = root; this.onState = onState; this.home = home; this.focus = focus;
    const r = this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    r.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    r.setScissorTest(true);
    this.camera = new THREE.PerspectiveCamera(FOV, 1, 0.01, 200);
    this.camera.up.set(0, 0, 1);
    this.camTop = new THREE.PerspectiveCamera(FOV, 1, 0.01, 200);    // the input camera's view, for the two-row layout
    this.camTop.up.set(0, 0, 1);
    const ctl = this.controls = new OrbitControls(this.camera, canvas);
    ctl.enablePan = false;
    ctl.rotateSpeed = 0.85;
    ctl.zoomSpeed = 0.8;
    ctl.minPolarAngle = 0.05;
    ctl.maxPolarAngle = Math.PI * 0.6;             // down to a little below the floor, never under it
    ctl.addEventListener("change", () => this.render());
    canvas.style.touchAction = "pan-y";              // on a touchscreen a sideways drag turns, a vertical swipe still scrolls
    // the wheel zooms only with ctrl or cmd (a trackpad pinch sends ctrlKey), so the page still scrolls over the canvas;
    // in the two-row layout the top row (the input camera) does not turn
    canvas.addEventListener("wheel", e => { if ((!e.ctrlKey && !e.metaKey) || this.inTop(e)) e.stopImmediatePropagation(); }, { capture: true });
    canvas.addEventListener("pointerdown", e => { if (this.inTop(e)) e.stopImmediatePropagation(); }, { capture: true });
    canvas.addEventListener("dblclick", e => { if (!this.inTop(e)) this.preset(this.home); });
    this.keys = []; this.rows = 1; this.frame = 0; this.mode = "app"; this.skel = false; this.spin = false;
    this.panels = {}; this.files = new Map(); this.gt = null; this.token = 0;
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }

  inTop(e) { return this.rows === 2 && e.offsetY < this.canvas.clientHeight / 2; }

  get(url) {                                         // a small cache of fetched files (base files are shared by cameras)
    if (!this.files.has(url)) {
      const p = fetchBin(url);
      p.catch(() => this.files.delete(url));
      this.files.set(url, p);
      if (this.files.size > 6) this.files.delete(this.files.keys().next().value);
    }
    return this.files.get(url);
  }

  async load(id, k) {
    const tok = ++this.token, kk = String(k).padStart(2, "0");
    this.onState("loading");
    let base, cam;
    try { [base, cam] = await Promise.all([this.get(`${this.root}${id}.bin`), this.get(`${this.root}${id}_v${kk}.bin`)]); }
    catch (e) { if (tok === this.token) this.onState("error", e); return false; }
    if (tok !== this.token) return false;
    if (!this.gt || this.gt.id !== id) this.setBase(id, base);
    this.setCamera(cam);
    this.preset(this.home);
    this.update();
    this.render();
    this.onState("ready");
    return true;
  }

  setBase(id, { hdr, blobs }) {
    const T = hdr.frames, origin = hdr.origin, step = hdr.step, nv = hdr.nv, nl = hdr.nv_view;
    const F0 = blobs.F0.a, sub = [];
    for (let i = 0; i < hdr.levels; i++) sub.push(blobs[`sub${i}`].a);
    const F = blobs.F ? blobs.F.a : F0;
    const col = new Float32Array(3 * nl), pale = new Float32Array(3 * nl);
    for (let v = 0; v < nl; v++) {                   // the colours, and their pale version under the stretch colours
      const c = [0, 1, 2].map(a => blobs.col.a[3 * v + a]), g = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
      for (let a = 0; a < 3; a++) {
        col[3 * v + a] = TOLIN[c[a]];
        pale[3 * v + a] = (0.42 + 0.58 * (0.3 * c[a] + 0.7 * g) / 255) * 0.97;
      }
    }
    const gt = decodePositions(blobs.V_gt, T, origin, step), V0 = gt[0], size = step * 8191;
    const pg = patchGraph(V0, nv, F0, STRETCH.scale * size), d0 = new Float32Array(pg.ce.length / 2), ccnt = new Float32Array(pg.nc);
    for (let e = 0; e < d0.length; e++) {
      const la = pg.ce[2 * e], lb = pg.ce[2 * e + 1], a = 3 * pg.rep[la], b = 3 * pg.rep[lb];
      d0[e] = Math.hypot(V0[a] - V0[b], V0[a + 1] - V0[b + 1], V0[a + 2] - V0[b + 2]);
      ccnt[la] += 1; ccnt[lb] += 1;
    }
    this.gt = { id, hdr, T, F0, F, sub, col, pale, pg, d0, ccnt, size, V: gt, J: blobs.J_gt.a, parents: blobs.parents.a, nj: hdr.joints };
    this.disposePanels();
  }

  setCamera({ hdr, blobs }) {
    const g = this.gt, seqs = { gt: { V: g.V, J: g.J } };
    for (const m of hdr.methods) seqs[m] = { V: decodePositions(blobs[`V_${m}`], g.T, g.hdr.origin, g.hdr.step), J: blobs[`J_${m}`] ? blobs[`J_${m}`].a : null };
    this.seqs = seqs;
    this.missing = new Set(hdr.missing || []);
    // the input camera's azimuth and elevation (it looks along -z of cam_R), for the presets
    const R = hdr.cam_R, f = [-R[0][2], -R[1][2], -R[2][2]];
    this.camAz = Math.atan2(-f[1], -f[0]) * 180 / Math.PI;
    this.camEl = Math.asin(Math.max(-1, Math.min(1, -f[2]))) * 180 / Math.PI;
    // a point sample of every output over all frames, to frame the shared camera on all of them (as the videos do)
    const pts = [], per = Math.max(1, Math.floor(g.hdr.nv * g.T * Object.keys(seqs).length / 24000));
    for (const s of Object.values(seqs)) for (let t = 0; t < g.T; t++) {
      const P = s.V[t];
      for (let i = (t * 7) % per; i < P.length / 3; i += per) pts.push(P[3 * i], P[3 * i + 1], P[3 * i + 2]);
    }
    this.pts = new Float32Array(pts);
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < this.pts.length; i += 3) for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], this.pts[i + a]); hi[a] = Math.max(hi[a], this.pts[i + a]); }
    this.box = { lo, hi, size: Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) };
    this.place(this.camTop, frameCamera(this.pts, this.camAz, this.camEl));
    this.buildPanels();
  }

  disposePanels() {
    for (const p of Object.values(this.panels)) p.scene.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); });
    this.panels = {};
  }

  buildPanels() {
    const g = this.gt, size = g.size;
    const want = Object.keys(this.seqs);
    for (const k of Object.keys(this.panels)) if (!want.includes(k)) delete this.panels[k];
    for (const key of want) {
      if (this.panels[key]) continue;
      const scene = new THREE.Scene();
      scene.add(new THREE.HemisphereLight(0xffffff, 0xe6e8ec, 2.05));
      const key1 = new THREE.DirectionalLight(0xffffff, 2.9), fill = new THREE.DirectionalLight(0xffffff, 0.85),
            rim = new THREE.DirectionalLight(0xffffff, 0.9), sun = new THREE.DirectionalLight(0xffffff, 0.5);
      sun.castShadow = true;
      sun.shadow.mapSize.set(1024, 1024);
      sun.shadow.bias = -0.0004;
      sun.shadow.normalBias = 0.01 * size;
      sun.shadow.radius = 4;
      for (const L of [key1, fill, rim, sun]) { scene.add(L); scene.add(L.target); }
      const nv = g.hdr.nv_view;
      const geom = new THREE.BufferGeometry();
      geom.setIndex(new THREE.BufferAttribute(g.F, 1));
      geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3 * nv), 3));
      geom.setAttribute("color", new THREE.BufferAttribute(g.col.slice(), 3));
      const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, metalness: 0, side: THREE.DoubleSide });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.castShadow = true;
      mesh.frustumCulled = false;
      scene.add(mesh);
      const floor = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.ShadowMaterial({ opacity: 0.2 }));
      floor.receiveShadow = true;
      scene.add(floor);
      const p = { scene, mesh, geom, floor, lights: { key: key1, fill, rim, sun }, skel: null };
      if (this.seqs[key].J) p.skel = this.makeSkeleton(scene, g.nj, size);
      this.panels[key] = p;
    }
    // floor under the ground truth's lowest point, sized to every output's extent
    for (const p of Object.values(this.panels)) {
      const b = this.box, cx = (b.lo[0] + b.hi[0]) / 2, cy = (b.lo[1] + b.hi[1]) / 2;
      p.floor.position.set(cx, cy, g.hdr.floor - 0.002 * b.size);
      p.floor.scale.set(b.size * 8, b.size * 8, 1);
      const s = p.lights.sun.shadow.camera, h = b.size * 1.2;
      s.left = -h; s.right = h; s.top = h; s.bottom = -h; s.near = 0.01; s.far = b.size * 12; s.updateProjectionMatrix();
    }
  }

  makeSkeleton(scene, nj, size) {
    const mk = (geo, color) => {
      const m = new THREE.InstancedMesh(geo, new THREE.MeshBasicMaterial({ color, depthTest: false, depthWrite: false, transparent: true }), nj);
      m.renderOrder = 10; m.frustumCulled = false; m.visible = false; scene.add(m); return m;
    };
    const joints = mk(new THREE.SphereGeometry(1, 14, 10), 0xffffff), bones = mk(new THREE.CylinderGeometry(1, 1, 1, 10), 0x3b6be6);
    const red = new THREE.Color(0xe0312b), green = new THREE.Color(0x2ba44e);
    for (let j = 0; j < nj; j++) joints.setColorAt(j, this.gt.parents[j] < 0 ? green : red);
    return { joints, bones, rj: 0.014 * size, rb: 0.0065 * size };
  }

  // positions at the viewing level, subdivided in place (a level's vertices are a prefix of the next's)
  fill(out, P0) {
    out.set(P0);
    let n = P0.length / 3;
    for (const pr of this.gt.sub) for (let e = 0; e < pr.length; e += 2, n++) {
      const a = 3 * pr[e], b = 3 * pr[e + 1], o = 3 * n;
      out[o] = 0.5 * (out[a] + out[b]); out[o + 1] = 0.5 * (out[a + 1] + out[b + 1]); out[o + 2] = 0.5 * (out[a + 2] + out[b + 2]);
    }
  }

  // per-vertex edge stretch (see the header) of Pm against the ground truth Pg, at the viewing level
  stretch(Pm, Pg) {
    const g = this.gt, nv = g.hdr.nv, G = g.pg, nc = G.nc, F0 = g.F0, nf = F0.length / 3;
    const cs = this._cs && this._cs.length === nc ? this._cs : (this._cs = new Float32Array(nc));
    const s = this._s && this._s.length === g.hdr.nv_view ? this._s : (this._s = new Float32Array(g.hdr.nv_view));
    const t = this._t && this._t.length === nv ? this._t : (this._t = new Float32Array(nv)), w = this._w || (this._w = new Float32Array(nv));
    cs.fill(0);
    for (let e = 0; e < g.d0.length; e++) {
      if (g.d0[e] <= 1e-9) continue;
      const la = G.ce[2 * e], lb = G.ce[2 * e + 1], a = 3 * G.rep[la], b = 3 * G.rep[lb];
      const dm = Math.hypot(Pm[a] - Pm[b], Pm[a + 1] - Pm[b + 1], Pm[a + 2] - Pm[b + 2]);
      const dg = Math.hypot(Pg[a] - Pg[b], Pg[a + 1] - Pg[b + 1], Pg[a + 2] - Pg[b + 2]);
      const d = Math.abs(dm - dg) / g.d0[e];
      cs[la] += d; cs[lb] += d;
    }
    for (let l = 0; l < nc; l++) cs[l] /= Math.max(1, g.ccnt[l]);
    for (let v = 0; v < nv; v++) s[v] = cs[G.lab[v]];
    for (let pass = 0; pass < 2; pass++) {           // soften the patch borders over the mesh edges
      t.fill(0); w.fill(0);
      for (let f = 0; f < nf; f++) for (let e = 0; e < 3; e++) {
        const a = F0[3 * f + e], b = F0[3 * f + (e + 1) % 3];
        t[a] += s[b]; w[a] += 1; t[b] += s[a]; w[b] += 1;
      }
      for (let v = 0; v < nv; v++) s[v] = w[v] ? 0.5 * s[v] + 0.5 * t[v] / w[v] : s[v];
    }
    let n = nv;
    for (const pr of g.sub) for (let e = 0; e < pr.length; e += 2, n++) s[n] = 0.5 * (s[pr[e]] + s[pr[e + 1]]);
    return s;
  }

  stretchColours(out, s) {
    const pale = this.gt.pale, k = (NS - 1) / STRETCH.hi;
    for (let v = 0; v < s.length; v++) {
      const i = 4 * Math.min(NS - 1, Math.round(s[v] * k)), a = HEAT[i + 3];
      for (let c = 0; c < 3; c++) out[3 * v + c] = TOLIN[Math.round(255 * (pale[3 * v + c] * (1 - a) + HEAT[i + c] * a))];
    }
  }

  update() {
    if (!this.gt || !this.seqs) return;
    const f = Math.min(this.frame, this.gt.T - 1), Pg = this.seqs.gt.V[f];
    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), yAxis = new THREE.Vector3(0, 1, 0);
    const a = new THREE.Vector3(), b = new THREE.Vector3(), d = new THREE.Vector3(), sc = new THREE.Vector3();
    for (const key of this.keys) {
      const p = this.panels[key], s = this.seqs[key];
      if (!p || !s) continue;
      const pos = p.geom.attributes.position;
      this.fill(pos.array, s.V[f]);
      pos.needsUpdate = true;
      p.geom.computeVertexNormals();
      const col = p.geom.attributes.color;
      if (this.mode === "strain") this.stretchColours(col.array, this.stretch(s.V[f], Pg)); else col.array.set(this.gt.col);
      col.needsUpdate = true;
      if (p.skel) {
        const on = this.skel && !!s.J;
        p.skel.joints.visible = p.skel.bones.visible = on;
        if (on) {
          const J = s.J, nj = this.gt.nj, o = 3 * f * nj, par = this.gt.parents;
          for (let j = 0; j < nj; j++) {
            a.set(J[o + 3 * j], J[o + 3 * j + 1], J[o + 3 * j + 2]);
            p.skel.joints.setMatrixAt(j, m4.compose(a, q.identity(), sc.setScalar(p.skel.rj * (par[j] < 0 ? 1.25 : 1))));
            const pj = par[j];
            if (pj < 0) { p.skel.bones.setMatrixAt(j, m4.makeScale(0, 0, 0)); continue; }
            b.set(J[o + 3 * pj], J[o + 3 * pj + 1], J[o + 3 * pj + 2]);
            d.subVectors(a, b);
            const len = d.length();
            q.setFromUnitVectors(yAxis, len > 1e-9 ? d.clone().divideScalar(len) : yAxis);
            p.skel.bones.setMatrixAt(j, m4.compose(a.clone().add(b).multiplyScalar(0.5), q, sc.set(p.skel.rb, len, p.skel.rb)));
          }
          p.skel.joints.instanceMatrix.needsUpdate = p.skel.bones.instanceMatrix.needsUpdate = true;
        }
      }
    }
  }

  setFrame(f) { if (f === this.frame) return; this.frame = f; this.update(); this.render(); }
  setMode(m) { this.mode = m; this.update(); this.render(); }
  setSkeleton(on) { this.skel = on; this.update(); this.render(); }
  setCols(keys) { this.keys = keys.slice(); this.update(); this.render(); }
  setRows(n) { this.rows = n; this.render(); }

  setSpin(on) {
    this.spin = on;
    this.controls.autoRotate = on;
    this.controls.autoRotateSpeed = 2.4;
    const tick = () => { if (!this.spin) return; this.controls.update(); requestAnimationFrame(tick); };
    if (on) requestAnimationFrame(tick);
  }

  place(cam, { c, d, fwd }) {
    cam.position.copy(c).addScaledVector(fwd, -d);
    cam.near = Math.max(0.002, d / 200); cam.far = d * 20 + this.box.size * 10;
    cam.lookAt(c);
    cam.updateProjectionMatrix();
    return c;
  }

  // turn: 55 degrees from the input camera; input: the input camera's direction; top: from above; focus: opts.focus =
  // {target, radius, az, el} in the clip's world frame (az and el absolute, or daz / del relative to the input camera)
  preset(name) {
    if (!this.pts) return;
    let fit;
    if (name === "focus" && this.focus) {
      const fo = this.focus, az = fo.az !== undefined ? fo.az : this.camAz + (fo.daz || 0), el = fo.el !== undefined ? fo.el : this.camEl + (fo.del || 0);
      const { fwd } = camAxes(az, el), c = new THREE.Vector3(...fo.target);
      fit = { c, d: fo.radius * 2 * FX / (1 - 2 * 0.06) + fo.radius * 0.4, fwd };
    } else {
      const az = name === "input" ? this.camAz : this.camAz + 55;
      const el = name === "input" ? this.camEl : name === "top" ? 62 : Math.max(this.camEl, 18);
      fit = frameCamera(this.pts, az, el);
    }
    this.controls.target.copy(this.place(this.camera, fit));
    this.controls.update();
    this.render();
  }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.render();
  }

  placeLights(p, cam, target) {                      // bl_video.area_lights / aim_sun, relative to the camera's azimuth
    const v = cam.position.clone().sub(target), az = Math.atan2(v.y, v.x) * 180 / Math.PI;
    const r = Math.max(1, this.box.size * 4);
    const put = (L, daz, el) => {
      const a = (az + daz) * Math.PI / 180, e = el * Math.PI / 180;
      L.position.set(target.x + r * Math.cos(e) * Math.cos(a), target.y + r * Math.cos(e) * Math.sin(a), target.z + r * Math.sin(e));
      L.target.position.copy(target);
      L.target.updateMatrixWorld();
    };
    put(p.lights.key, -35, 45); put(p.lights.fill, 60, 15); put(p.lights.rim, 170, 35); put(p.lights.sun, -20, 64);
  }

  render() {
    const n = this.keys.length, r = this.renderer;
    const W = this.canvas.clientWidth, H = this.canvas.clientHeight;
    if (!n || !W || !H || !this.box) return;
    const tw = W / n, th = H / this.rows;
    const topTarget = new THREE.Vector3();
    this.camTop.getWorldDirection(topTarget).multiplyScalar(this.camTop.position.distanceTo(this.controls.target)).add(this.camTop.position);
    for (let row = 0; row < this.rows; row++) {
      const cam = this.rows === 2 && row === 0 ? this.camTop : this.camera;
      const target = cam === this.camera ? this.controls.target : topTarget;
      cam.aspect = tw / th;
      cam.updateProjectionMatrix();
      const y = Math.round((this.rows - 1 - row) * th);
      for (let i = 0; i < n; i++) {
        const x = Math.round(i * tw), w = Math.round((i + 1) * tw) - x, p = this.panels[this.keys[i]];
        r.setViewport(x, y, w, Math.round(th)); r.setScissor(x, y, w, Math.round(th));
        r.setClearColor(BG, 1);
        if (!p || !this.seqs || !this.seqs[this.keys[i]]) { r.clear(); continue; }
        this.placeLights(p, cam, target);
        r.render(p.scene, cam);
      }
    }
  }
}
