/* 区域标定：在实时快照上绘制水面区/饮水机区/水面线并保存到系统设置。 */
"use strict";
const state = {
  fw: 1280, fh: 720,
  waterZone: null, fountainZone: null, waterSurfaceY: null,
  mode: "water", dragStart: null, dragNow: null,
  img: null, scale: 1, // display px per frame px
};
const cv = document.getElementById("cv");
const ctx = cv.getContext("2d");
const $ = (id) => document.getElementById(id);
const status = (t) => { $("status").textContent = t; };

function frameRect() { // display-space rect of the whole frame
  return { w: state.fw * state.scale, h: state.fh * state.scale };
}

function toFrame(ev) {
  const r = cv.getBoundingClientRect();
  return {
    x: Math.round(((ev.clientX - r.left) / state.scale)),
    y: Math.round(((ev.clientY - r.top) / state.scale)),
  };
}

function clampZone(z) {
  const x1 = Math.max(0, Math.min(z.x1, z.x2, state.fw));
  const x2 = Math.max(0, Math.min(Math.max(z.x1, z.x2), state.fw));
  const y1 = Math.max(0, Math.min(z.y1, z.y2, state.fh));
  const y2 = Math.max(0, Math.min(Math.max(z.y1, z.y2), state.fh));
  return { x1: Math.round(x1), y1: Math.round(y1), x2: Math.round(x2), y2: Math.round(y2) };
}

function draw() {
  if (!state.img) return;
  const dW = cv.width = Math.min(1060, cv.parentElement.clientWidth - 2);
  state.scale = dW / state.fw;
  cv.height = Math.round(state.fh * state.scale);
  ctx.drawImage(state.img, 0, 0, cv.width, cv.height);
  const k = state.scale;
  const box = (z, color) => {
    if (!z) return;
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.strokeRect(z.x1 * k, z.y1 * k, (z.x2 - z.x1) * k, (z.y2 - z.y1) * k);
    ctx.fillStyle = color + "22";
    ctx.fillRect(z.x1 * k, z.y1 * k, (z.x2 - z.x1) * k, (z.y2 - z.y1) * k);
  };
  box(state.waterZone, "#1e88e5");
  box(state.fountainZone, "#fb8c00");
  if (state.waterSurfaceY != null) {
    ctx.strokeStyle = "#f4c20d"; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, state.waterSurfaceY * k);
    ctx.lineTo(cv.width, state.waterSurfaceY * k);
    ctx.stroke();
  }
  if (state.dragStart && state.dragNow) {
    ctx.strokeStyle = "#ffffff88"; ctx.lineWidth = 1.5;
    ctx.strokeRect(
      Math.min(state.dragStart.x, state.dragNow.x) * k,
      Math.min(state.dragStart.y, state.dragNow.y) * k,
      Math.abs(state.dragNow.x - state.dragStart.x) * k,
      Math.abs(state.dragNow.y - state.dragStart.y) * k);
  }
}

function syncInputs() {
  const set = (id, v) => { $(id).value = (v == null ? "" : v); };
  set("wx1", state.waterZone && state.waterZone.x1);
  set("wy1", state.waterZone && state.waterZone.y1);
  set("wx2", state.waterZone && state.waterZone.x2);
  set("wy2", state.waterZone && state.waterZone.y2);
  set("fx1", state.fountainZone && state.fountainZone.x1);
  set("fy1", state.fountainZone && state.fountainZone.y1);
  set("fx2", state.fountainZone && state.fountainZone.x2);
  set("fy2", state.fountainZone && state.fountainZone.y2);
  set("wsy", state.waterSurfaceY);
}

async function loadSnapshot() {
  status("加载快照…");
  const img = new Image();
  img.onload = () => {
    state.img = img; draw(); syncInputs(); status("快照已加载");
  };
  img.onerror = () => status("快照加载失败：请确认相机抓帧服务运行中（/media/live/latest.jpg）");
  img.src = "/media/live/latest.jpg?t=" + Date.now();
}

async function loadSettings() {
  const s = await fetch("/api/settings").then((r) => r.json());
  state.fw = Number(s.frameWidth) || 1280;
  state.fh = Number(s.frameHeight) || 720;
  const norm = (z) => (z && typeof z === "object" &&
    [z.x1, z.y1, z.x2, z.y2].every((v) => v != null) ? z : null);
  state.waterZone = norm(s.waterZone);
  state.fountainZone = norm(s.fountainZone);
  state.waterSurfaceY = (s.waterSurfaceY == null ? null : Number(s.waterSurfaceY));
  $("dim").textContent = state.fw + "×" + state.fh;
}

function bindMouse() {
  cv.addEventListener("mousedown", (ev) => {
    if (state.mode === "line") return;
    state.dragStart = toFrame(ev); state.dragNow = state.dragStart; draw();
  });
  cv.addEventListener("mousemove", (ev) => {
    if (!state.dragStart || state.mode === "line") return;
    state.dragNow = toFrame(ev); draw();
  });
  window.addEventListener("mouseup", (ev) => {
    if (!state.dragStart || state.mode === "line") return;
    const z = clampZone({ x1: state.dragStart.x, y1: state.dragStart.y,
                          x2: toFrame(ev).x, y2: toFrame(ev).y });
    if (z.x2 - z.x1 < 4 || z.y2 - z.y1 < 4) { state.dragStart = null; draw(); return; }
    if (state.mode === "water") state.waterZone = z; else state.fountainZone = z;
    state.dragStart = null; syncInputs(); draw();
  });
  cv.addEventListener("click", (ev) => {
    if (state.mode !== "line") return;
    state.waterSurfaceY = Math.round(toFrame(ev).y);
    syncInputs(); draw();
  });
}

function bindInputs() {
  const bind = (id, apply) => {
    $(id).addEventListener("change", () => {
      const v = parseInt($(id).value, 10);
      apply(Number.isFinite(v) ? v : null); draw();
    });
  };
  const zi = (id, key, field) => bind(id, (v) => {
    if (!state[key]) state[key] = { x1: 0, y1: 0, x2: 0, y2: 0 };
    if (v != null) state[key][field] = v;
  });
  zi("wx1", "waterZone", "x1"); zi("wy1", "waterZone", "y1");
  zi("wx2", "waterZone", "x2"); zi("wy2", "waterZone", "y2");
  zi("fx1", "fountainZone", "x1"); zi("fy1", "fountainZone", "y1");
  zi("fx2", "fountainZone", "x2"); zi("fy2", "fountainZone", "y2");
  bind("wsy", (v) => { state.waterSurfaceY = v; });
  document.querySelectorAll('input[name="mode"]').forEach((el) => {
    el.addEventListener("change", () => { state.mode = el.value; });
  });
}

async function save() {
  const payload = {};
  if (state.waterZone) payload.waterZone = state.waterZone;
  if (state.fountainZone) payload.fountainZone = state.fountainZone;
  if (state.waterSurfaceY != null) payload.waterSurfaceY = state.waterSurfaceY;
  if (!Object.keys(payload).length) { status("没有可保存的区域"); return; }
  status("保存中…");
  try {
    const res = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json().catch(() => ({}));
    status(res.ok && data.ok !== false
      ? "已保存，检测器将在下一轮检测自动生效"
      : "保存失败: " + (data.error || res.status));
  } catch (e) { status("保存失败: " + e); }
}

document.getElementById("save").addEventListener("click", save);
document.getElementById("refresh").addEventListener("click", loadSnapshot);
document.getElementById("clear").addEventListener("click", () => {
  if (state.mode === "water") state.waterZone = null;
  else if (state.mode === "fountain") state.fountainZone = null;
  else state.waterSurfaceY = null;
  syncInputs(); draw();
});
window.addEventListener("resize", draw);

(async function init() {
  bindMouse(); bindInputs();
  try { await loadSettings(); } catch (e) { status("设置读取失败: " + e); }
  await loadSnapshot();
})();
