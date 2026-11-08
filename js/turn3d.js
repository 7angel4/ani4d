// Ani4D project page: the comparison player's drag-to-turn row. Every method's output is drawn with WebGL in one canvas
// laid over the grid's bottom row, one viewport per shown column, all sharing one orbit camera, so dragging any column
// turns them all. Data (render/make_3d.py): media/3d/<id>.bin holds the input mesh's faces and colours, the midpoint
// subdivision the renders use, and the ground truth; media/3d/<id>_v<kk>.bin holds the five outputs for camera kk.
// The look follows the Blender renders (render/bl_video.py): three lights placed relative to the camera, a floor that
// only carries the shadow, the same edge-strain ramp (0 grey to 0.06 dark red).
import * as THREE from "./three.module.min.js";
import { OrbitControls } from "./OrbitControls.js";

const SMAX = 0.06;
const BG = 0xf2f3f5;
const FX = 50 / 36;                                  // the renders' 50 mm lens on a 36 mm sensor
const FOV = 2 * Math.atan(18 / 50) * 180 / Math.PI;
const RAMP = [[232, 230, 227], [244, 199, 160], [238, 128, 62], [196, 52, 24], [150, 16, 12]];
const lin = v => (v /= 255) <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
const LUT = new Float32Array(256 * 3);               // the ramp, interpolated in sRGB as make_compare.py does, as linear RGB
for (let i = 0; i < 256; i++) {
  const x = i / 255 * (RAMP.length - 1), k = Math.min(Math.floor(x), RAMP.length - 2), f = x - k;
  for (let c = 0; c < 3; c++) LUT[3 * i + c] = lin(RAMP[k][c] * (1 - f) + RAMP[k + 1][c] * f);
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
  constructor(canvas, { root = "media/3d/", onState = () => {} } = {}) {
    this.canvas = canvas; this.root = root; this.onState = onState;
    const r = this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    r.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    r.setScissorTest(true);
    this.camera = new THREE.PerspectiveCamera(FOV, 1, 0.01, 200);
    this.camera.up.set(0, 0, 1);
    const ctl = this.controls = new OrbitControls(this.camera, canvas);
    ctl.enablePan = false;
    ctl.rotateSpeed = 0.85;
    ctl.zoomSpeed = 0.8;
    ctl.minPolarAngle = 0.05;
    ctl.maxPolarAngle = Math.PI * 0.6;             // down to a little below the floor, never under it
    ctl.addEventListener("change", () => this.render());
    canvas.style.touchAction = "pan-y";              // on a touchscreen a sideways drag turns, a vertical swipe still scrolls
    // the wheel zooms only with ctrl or cmd (a trackpad pinch sends ctrlKey), so the page still scrolls over the grid
    canvas.addEventListener("wheel", e => { if (!e.ctrlKey && !e.metaKey) e.stopImmediatePropagation(); }, { capture: true });
    canvas.addEventListener("dblclick", () => this.preset("turn"));
    this.keys = []; this.frame = 0; this.mode = "app"; this.skel = false; this.spin = false;
    this.panels = {}; this.files = new Map(); this.gt = null; this.view = null; this.token = 0; this.view0 = "turn";
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }

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
    this.preset(this.view0);
    this.update();
    this.render();
    this.onState("ready");
    return true;
  }

  setBase(id, { hdr, blobs }) {
    const T = hdr.frames, origin = hdr.origin, step = hdr.step;
    const F0 = blobs.F0.a, pairs = [];
    for (let i = 0; i < hdr.levels; i++) pairs.push(blobs[`sub${i}`].a);
    const F = blobs.F ? blobs.F.a : F0;
    const col = new Float32Array(3 * hdr.nv_view);
    for (let i = 0; i < col.length; i++) col[i] = lin(blobs.col.a[i]);
    const gt = decodePositions(blobs.V_gt, T, origin, step);
    // edges of the input mesh and their rest lengths (the input mesh = the ground truth's frame 0), as make_compare.py
    const nf = F0.length / 3, l0 = new Float32Array(3 * nf), cnt = new Float32Array(hdr.nv), V0 = gt[0];
    for (let f = 0; f < nf; f++) for (let e = 0; e < 3; e++) {
      const a = 3 * F0[3 * f + e], b = 3 * F0[3 * f + (e + 1) % 3];
      l0[3 * f + e] = Math.hypot(V0[a] - V0[b], V0[a + 1] - V0[b + 1], V0[a + 2] - V0[b + 2]);
      cnt[F0[3 * f + e]] += 1;
    }
    this.gt = { id, hdr, T, F0, F, pairs, col, l0, cnt, V: gt, J: blobs.J_gt.a, parents: blobs.parents.a, nj: hdr.joints };
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
    this.buildPanels();
  }

  disposePanels() {
    for (const p of Object.values(this.panels)) p.scene.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); });
    this.panels = {};
  }

  buildPanels() {
    const g = this.gt, size = g.hdr.step * 8191;
    const want = Object.keys(this.seqs);
    for (const k of Object.keys(this.panels)) if (!want.includes(k)) { delete this.panels[k]; }
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

  // positions of `key` at frame f, subdivided in place into the geometry (a level's vertices are a prefix of the next's)
  fill(out, P0) {
    out.set(P0);
    let n = P0.length / 3;
    for (const pr of this.gt.pairs) for (let e = 0; e < pr.length; e += 2, n++) {
      const a = 3 * pr[e], b = 3 * pr[e + 1], o = 3 * n;
      out[o] = 0.5 * (out[a] + out[b]); out[o + 1] = 0.5 * (out[a + 1] + out[b + 1]); out[o + 2] = 0.5 * (out[a + 2] + out[b + 2]);
    }
  }

  strainColours(out, Pm, Pg) {                       // make_compare.edge_strain_dev, then the ramp
    const g = this.gt, F = g.F0, nf = F.length / 3, nv = g.hdr.nv;
    const s = this._s && this._s.length === g.hdr.nv_view ? this._s : (this._s = new Float32Array(g.hdr.nv_view));
    s.fill(0, 0, nv);
    for (let f = 0; f < nf; f++) {
      let d = 0;
      for (let e = 0; e < 3; e++) {
        const l0 = g.l0[3 * f + e];
        if (l0 <= 1e-9) continue;
        const a = 3 * F[3 * f + e], b = 3 * F[3 * f + (e + 1) % 3];
        const lm = Math.hypot(Pm[a] - Pm[b], Pm[a + 1] - Pm[b + 1], Pm[a + 2] - Pm[b + 2]);
        const lg = Math.hypot(Pg[a] - Pg[b], Pg[a + 1] - Pg[b + 1], Pg[a + 2] - Pg[b + 2]);
        d += Math.abs(lm - lg) / l0;
      }
      d /= 3;
      s[F[3 * f]] += d; s[F[3 * f + 1]] += d; s[F[3 * f + 2]] += d;
    }
    for (let v = 0; v < nv; v++) s[v] /= Math.max(1, g.cnt[v]);
    let n = nv;
    for (const pr of g.pairs) for (let e = 0; e < pr.length; e += 2, n++) s[n] = 0.5 * (s[pr[e]] + s[pr[e + 1]]);
    for (let v = 0; v < g.hdr.nv_view; v++) {
      const i = 3 * Math.min(255, Math.round(s[v] / SMAX * 255));
      out[3 * v] = LUT[i]; out[3 * v + 1] = LUT[i + 1]; out[3 * v + 2] = LUT[i + 2];
    }
  }

  update() {
    if (!this.gt) return;
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
      if (this.mode === "strain") this.strainColours(col.array, s.V[f], Pg); else col.array.set(this.gt.col);
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

  setSpin(on) {
    this.spin = on;
    this.controls.autoRotate = on;
    this.controls.autoRotateSpeed = 2.4;
    const tick = () => { if (!this.spin) return; this.controls.update(); requestAnimationFrame(tick); };
    if (on) requestAnimationFrame(tick);
  }

  preset(name) {
    if (!this.pts) return;
    this.view0 = name === "input" || name === "top" ? this.view0 : name;
    const az = name === "input" ? this.camAz : this.camAz + 55;
    const el = name === "input" ? this.camEl : name === "top" ? 62 : Math.max(this.camEl, 18);
    const { c, d, fwd } = frameCamera(this.pts, az, el);
    this.controls.target.copy(c);
    this.camera.position.copy(c).addScaledVector(fwd, -d);
    this.camera.near = Math.max(0.005, d / 100); this.camera.far = d * 20 + this.box.size * 10;
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.render();
  }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.render();
  }

  placeLights(p) {                                   // bl_video.area_lights / aim_sun, relative to the camera's azimuth
    const t = this.controls.target, v = this.camera.position.clone().sub(t), az = Math.atan2(v.y, v.x) * 180 / Math.PI;
    const r = Math.max(1, this.box.size * 4);
    const put = (L, daz, el) => {
      const a = (az + daz) * Math.PI / 180, e = el * Math.PI / 180;
      L.position.set(t.x + r * Math.cos(e) * Math.cos(a), t.y + r * Math.cos(e) * Math.sin(a), t.z + r * Math.sin(e));
      L.target.position.copy(t);
      L.target.updateMatrixWorld();
    };
    put(p.lights.key, -35, 45); put(p.lights.fill, 60, 15); put(p.lights.rim, 170, 35); put(p.lights.sun, -20, 64);
  }

  render() {
    const n = this.keys.length, r = this.renderer;
    const W = this.canvas.clientWidth, H = this.canvas.clientHeight;
    if (!n || !W || !H) return;
    const tw = W / n;
    this.camera.aspect = tw / H;
    this.camera.updateProjectionMatrix();
    for (let i = 0; i < n; i++) {
      const x = Math.round(i * tw), w = Math.round((i + 1) * tw) - x, p = this.panels[this.keys[i]];
      r.setViewport(x, 0, w, H); r.setScissor(x, 0, w, H);
      r.setClearColor(BG, 1);
      if (!p || !this.seqs || !this.seqs[this.keys[i]]) { r.clear(); continue; }
      this.placeLights(p);
      r.render(p.scene, this.camera);
    }
  }
}
