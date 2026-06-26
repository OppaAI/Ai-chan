// gesture.js
// Natural, loop-friendly gesture poses for archive/test-3d-model.html.
// Each exported gesture accepts the tester ctx and writes target rotations.

export function blend(from, to, t) {
  return from + (to - from) * t;
}

const TAU = Math.PI * 2;
const clamp01 = v => Math.max(0, Math.min(1, v));
const ease = v => v * v * (3 - 2 * v);
function wave(t, speed = 1, phase = 0) {
  return Math.sin(t * speed + phase);
}

function holdCycle(t, speed = 0.35, hold = 0.42) {
  const c = (t * speed) % 1;
  if (c < 0.22) return ease(c / 0.22);
  if (c < 0.22 + hold) return 1;
  return ease(1 - (c - 0.22 - hold) / (0.78 - hold));
}

function strokeCycle(t, speed = 1, downPortion = 0.72) {
  const c = (t * speed) % 1;
  const v = c < downPortion ? c / downPortion : 1 - (c - downPortion) / (1 - downPortion);
  return ease(clamp01(v));
}

function sideBones(ctx, side = ctx.side) {
  const left = side < 0;
  return {
    s: left ? -1 : 1,
    UA: left ? ctx.lUA : ctx.rUA,
    LA: left ? ctx.lLA : ctx.rLA,
    H: left ? ctx.lH : ctx.rH,
    restUA: left ? ctx.REST.leftUpperArm : ctx.REST.rightUpperArm,
    restLA: left ? ctx.REST.leftLowerArm : ctx.REST.rightLowerArm,
    restH: left ? ctx.REST.leftHand : ctx.REST.rightHand,
    ioUA: left ? ctx.io.lUA : ctx.io.rUA,
    ioLA: left ? ctx.io.lLA : ctx.io.rLA,
    ioH: left ? ctx.io.lH : ctx.io.rH,
  };
}

function poseArm(ctx, side, upper = {}, lower = {}, hand = {}) {
  const b = sideBones(ctx, side);
  const set = (bone, rest, io, vals) => {
    if (!bone) return;
    for (const axis of ['x', 'y', 'z']) {
      if (vals[axis] !== undefined) bone.rotation[axis] = blend((rest[axis] ?? 0) + (io[axis] ?? 0), vals[axis], 1);
    }
  };
  set(b.UA, b.restUA, b.ioUA, upper);
  set(b.LA, b.restLA, b.ioLA, lower);
  set(b.H, b.restH, b.ioH, hand);
}

function look(ctx, { x, y, z, neckX, neckY, neckZ }, amount) {
  if (neckX === undefined && x !== undefined) neckX = x * 0.45;
  if (neckY === undefined && y !== undefined) neckY = y * 0.45;
  if (neckZ === undefined && z !== undefined) neckZ = z * 0.45;
  const { head, neck, io } = ctx;
  if (head) {
    if (x !== undefined) head.rotation.x = blend(io.head.x, x, amount);
    if (y !== undefined) head.rotation.y = blend(io.head.y, y, amount);
    if (z !== undefined) head.rotation.z = blend(io.head.z, z, amount);
  }
  if (neck) {
    if (neckX !== undefined) neck.rotation.x = blend(io.neck?.x ?? 0, neckX, amount);
    if (neckY !== undefined) neck.rotation.y = blend(io.neck?.y ?? 0, neckY, amount);
    if (neckZ !== undefined) neck.rotation.z = blend(io.neck?.z ?? 0, neckZ, amount);
  }
}

export function idleBreathing(ctx) {
  const { t, intensity = 1, head, neck, io } = ctx;
  const a = intensity;
  look(ctx, {
    x: io.head.x + wave(t, 0.53) * 0.018 * a,
    y: io.head.y + wave(t, 0.31) * 0.055 * a + wave(t, 0.97, 1.2) * 0.012 * a,
    z: io.head.z + wave(t, 0.27, 1.1) * 0.025 * a,
  }, 1);
  if (neck) neck.rotation.x += wave(t, 0.71) * 0.008 * a;
  const armSway = wave(t, 0.42) * 0.03 * a;
  poseArm(ctx, -1, { x: ctx.REST.leftUpperArm.x + wave(t, 0.49) * 0.02 * a, z: ctx.REST.leftUpperArm.z + armSway }, { x: ctx.REST.leftLowerArm.x + wave(t, 0.67) * 0.014 * a }, { y: ctx.REST.leftHand.y + wave(t, 0.58) * 0.018 * a });
  poseArm(ctx, 1, { x: ctx.REST.rightUpperArm.x + wave(t, 0.47, 1.1) * 0.02 * a, z: ctx.REST.rightUpperArm.z + armSway }, { x: ctx.REST.rightLowerArm.x + wave(t, 0.63, 0.8) * 0.014 * a }, { y: ctx.REST.rightHand.y + wave(t, 0.54, 1.4) * 0.018 * a });
}

export function lookAround(ctx) {
  const { t, intensity = 1, side } = ctx;
  const a = holdCycle(t, 0.18, 0.5) * intensity;
  look(ctx, { x: -0.015 * a + wave(t, 0.7) * 0.018, y: side * (0.18 + 0.05 * wave(t, 0.8)) * a, z: -side * 0.035 * a }, 1);
}

export function meetGaze(ctx) {
  const a = holdCycle(ctx.t, 0.24, 0.56) * ctx.intensity;
  look(ctx, { x: 0.02, y: 0, z: 0, neckX: 0.006, neckY: 0, neckZ: 0 }, a);
}

export function curiousTilt(ctx) {
  const a = holdCycle(ctx.t, 0.28, 0.46) * ctx.intensity;
  look(ctx, { x: -0.035 * a, y: ctx.side * 0.055 * a, z: ctx.side * 0.16 * a }, 1);
}

export function headNod(ctx) {
  const n = Math.sin((ctx.t * (ctx.strokeSpeed ?? 1.2)) % 1 * TAU * 1.5) * ctx.intensity;
  look(ctx, { x: n * 0.11, y: 0, z: ctx.side * 0.012 }, 1);
}

export function hairBrush(ctx) {
  const { t, intensity = 1, strokeSpeed = 0.9, side } = ctx;
  const b = strokeCycle(t, strokeSpeed, 0.76);
  const a = holdCycle(t, 0.23, 0.5) * intensity;
  poseArm(ctx, side,
    { x: -0.48 * a, z: side * (0.82 + b * 0.08) * a },
    { x: -(1.18 - b * 0.45) * a, z: side * 0.18 * a },
    { x: -(0.20 + b * 0.18) * a, y: -side * (0.50 - b * 0.16) * a, z: side * (0.16 + b * 0.10) * a }
  );
  look(ctx, { x: -0.025 * a, y: side * (0.035 + b * 0.025) * a, z: -side * 0.075 * a }, 1);
}

export function lookAtHand(ctx) {
  const a = holdCycle(ctx.t, 0.25, 0.46) * ctx.intensity;
  poseArm(ctx, ctx.side, { x: -0.42 * a, z: ctx.side * 0.68 * a }, { x: -0.92 * a, z: ctx.side * 0.14 * a }, { x: 0.18 * a, y: -ctx.side * 0.38 * a, z: ctx.side * 0.08 * wave(ctx.t, 3) * a });
  look(ctx, { x: -0.08 * a, y: ctx.side * 0.18 * a, z: -ctx.side * 0.025 * a }, 1);
}

export function fingerPlay(ctx) {
  const a = holdCycle(ctx.t, 0.3, 0.5) * ctx.intensity;
  poseArm(ctx, ctx.side, { x: -0.16 * a, z: ctx.side * 0.18 * a }, { x: -0.42 * a, z: ctx.side * 0.06 * a }, { x: 0.04 * wave(ctx.t, 5.2) * a, y: -ctx.side * 0.28 * a, z: ctx.side * 0.16 * wave(ctx.t, 7.1) * a });
  look(ctx, { x: 0.055 * a, y: ctx.side * 0.11 * a, z: ctx.side * 0.02 * a }, 1);
}

export function raiseHand(ctx) {
  const a = holdCycle(ctx.t, 0.23, 0.44) * ctx.intensity;
  const flutter = wave(ctx.t, 8.5) * 0.10 * a;
  poseArm(ctx, ctx.side, { x: -0.72 * a, z: ctx.side * 0.44 * a }, { x: -0.86 * a, z: ctx.side * 0.10 * a }, { x: 0.10 * a, y: -ctx.side * 0.22 * a, z: ctx.side * flutter });
  look(ctx, { x: -0.025 * a, y: ctx.side * 0.10 * a, z: -ctx.side * 0.018 * a }, 1);
}

export function chinTouch(ctx) {
  const a = holdCycle(ctx.t, 0.22, 0.55) * ctx.intensity;
  poseArm(ctx, ctx.side, { x: -0.55 * a, z: ctx.side * 0.58 * a }, { x: -1.02 * a, z: ctx.side * 0.10 * a }, { x: -0.18 * a, y: -ctx.side * 0.20 * a, z: ctx.side * 0.06 * a });
  look(ctx, { x: 0.055 * a, y: -ctx.side * 0.035 * a, z: ctx.side * 0.045 * a }, 1);
}

export function touchCollar(ctx) {
  const a = holdCycle(ctx.t, 0.28, 0.42) * ctx.intensity;
  poseArm(ctx, ctx.side, { x: -0.46 * a, z: ctx.side * 0.44 * a }, { x: -0.82 * a, z: ctx.side * 0.18 * a }, { x: 0.12 * a, y: -ctx.side * 0.24 * a, z: ctx.side * 0.05 * wave(ctx.t, 4) * a });
  look(ctx, { x: -0.035 * a, y: ctx.side * 0.08 * a, z: -ctx.side * 0.03 * a }, 1);
}

export function wristFlick(ctx) {
  const a = holdCycle(ctx.t, 0.55, 0.15) * ctx.intensity;
  poseArm(ctx, ctx.side, { z: ctx.side * 0.16 * a }, { y: -ctx.side * 0.34 * a }, { y: -ctx.side * 0.25 * a, z: ctx.side * 0.48 * wave(ctx.t, 11) * a });
  look(ctx, { y: ctx.side * 0.06 * a, z: -ctx.side * 0.015 * a }, 1);
}

export function shoulderRoll(ctx) {
  const r = Math.sin((ctx.t * 0.55) % 1 * TAU) * ctx.intensity;
  poseArm(ctx, ctx.side, { x: -0.20 * r, y: ctx.side * 0.10 * r, z: ctx.side * 0.14 * r }, {}, {});
  look(ctx, { x: -0.02 * Math.abs(r), z: -ctx.side * 0.045 * r }, 1);
}

export function handOnHip(ctx) {
  const a = holdCycle(ctx.t, 0.21, 0.58) * ctx.intensity;
  poseArm(ctx, ctx.side, { x: -0.18 * a, z: ctx.side * 0.58 * a }, { x: -0.18 * a, z: ctx.side * 0.34 * a }, { y: -ctx.side * 0.12 * a, z: ctx.side * 0.22 * a });
  look(ctx, { y: -ctx.side * 0.05 * a, z: ctx.side * 0.025 * a }, 1);
}

export function adjustSleeve(ctx) {
  const a = holdCycle(ctx.t, 0.32, 0.34) * ctx.intensity;
  const workingSide = -ctx.side;
  poseArm(ctx, workingSide, { x: -0.30 * a, z: workingSide * 0.36 * a }, { x: -0.54 * a, y: -workingSide * 0.35 * a }, { x: -0.16 * a, z: workingSide * (0.12 + 0.08 * wave(ctx.t, 9)) * a });
  look(ctx, { x: 0.04 * a, y: workingSide * 0.08 * a }, 1);
}

export function brushShoulder(ctx) {
  const a = holdCycle(ctx.t, 0.35, 0.25) * ctx.intensity;
  const workingSide = -ctx.side;
  const brush = wave(ctx.t, 9.5) * a;
  poseArm(ctx, workingSide, { x: -0.28 * a, z: workingSide * 0.24 * a }, { x: -0.45 * a }, { z: workingSide * 0.26 * brush, y: -workingSide * 0.12 * a });
  look(ctx, { y: -workingSide * 0.08 * a, z: workingSide * 0.025 * a }, 1);
}

export function stretchArm(ctx) {
  const a = holdCycle(ctx.t, 0.18, 0.42) * ctx.intensity;
  poseArm(ctx, ctx.side, { x: -0.82 * a, z: ctx.side * 0.30 * a }, { x: -0.32 * a, z: ctx.side * 0.04 * a }, { x: 0.18 * a, y: -ctx.side * 0.10 * a });
  look(ctx, { x: -0.04 * a, y: ctx.side * 0.12 * a, z: -ctx.side * 0.025 * a }, 1);
}

export function stretchNeck(ctx) {
  const a = holdCycle(ctx.t, 0.2, 0.5) * ctx.intensity;
  look(ctx, { x: -0.10 * a, y: -ctx.side * 0.04 * a, z: ctx.side * 0.08 * a }, 1);
}

export function shiftWeight(ctx) {
  const a = Math.sin((ctx.t * 0.22) % 1 * Math.PI) * ctx.intensity;
  look(ctx, { x: 0.005, y: -ctx.side * 0.025 * a, z: ctx.side * 0.04 * a }, 1);
  poseArm(ctx, -1, { z: ctx.REST.leftUpperArm.z - ctx.side * 0.035 * a }, {}, {});
  poseArm(ctx, 1, { z: ctx.REST.rightUpperArm.z - ctx.side * 0.035 * a }, {}, {});
}

export function sway(ctx) {
  const s = wave(ctx.t, 0.8) * ctx.intensity;
  look(ctx, { x: 0.01 * wave(ctx.t, 1.1), y: 0.035 * s, z: -0.045 * s }, 1);
  poseArm(ctx, -1, { z: ctx.REST.leftUpperArm.z - 0.045 * s }, { x: ctx.REST.leftLowerArm.x + 0.018 * s }, {});
  poseArm(ctx, 1, { z: ctx.REST.rightUpperArm.z - 0.045 * s }, { x: ctx.REST.rightLowerArm.x - 0.018 * s }, {});
}

export function crossArms(ctx) {
  const a = holdCycle(ctx.t, 0.17, 0.5) * ctx.intensity;
  poseArm(ctx, -1, { x: -0.16 * a, z: -0.34 * a }, { x: -0.36 * a, z: -0.25 * a }, { z: -0.10 * a });
  poseArm(ctx, 1, { x: -0.16 * a, z: 0.34 * a }, { x: -0.36 * a, z: 0.25 * a }, { z: 0.10 * a });
  look(ctx, { x: 0.035 * a, y: ctx.side * 0.025 * a }, 1);
}

export function fingerPlay(ctx) {
  const a = holdCycle(ctx.t, 0.3, 0.5) * ctx.intensity;
  poseArm(ctx, ctx.side, { x: -0.22, z: ctx.side * 0.34 }, { x: -0.62, z: ctx.side * 0.12 }, { x: 0.06 * wave(ctx.t, 5.2), y: -ctx.side * 0.36, z: ctx.side * 0.22 * wave(ctx.t, 7.1) }, a);
  look(ctx, { x: 0.055 * a, y: ctx.side * 0.11 * a, z: ctx.side * 0.02 * a }, 1);
}

export function raiseHand(ctx) {
  const a = holdCycle(ctx.t, 0.23, 0.44) * ctx.intensity;
  const flutter = wave(ctx.t, 8.5) * 0.14;
  poseArm(ctx, ctx.side, { x: -0.88, z: ctx.side * 0.78 }, { x: -1.05, z: ctx.side * 0.18 }, { x: 0.15, y: -ctx.side * 0.28, z: ctx.side * flutter }, a);
  look(ctx, { x: -0.025 * a, y: ctx.side * 0.12 * a, z: -ctx.side * 0.018 * a }, 1);
}

export function chinTouch(ctx) {
  const a = holdCycle(ctx.t, 0.22, 0.55) * ctx.intensity;
  poseArm(ctx, ctx.side, { x: -0.66, z: ctx.side * 0.74 }, { x: -1.18, z: ctx.side * 0.16 }, { x: -0.22, y: -ctx.side * 0.28, z: ctx.side * 0.08 }, a);
  look(ctx, { x: 0.055 * a, y: -ctx.side * 0.035 * a, z: ctx.side * 0.045 * a }, 1);
}

export function touchCollar(ctx) {
  const a = holdCycle(ctx.t, 0.28, 0.42) * ctx.intensity;
  poseArm(ctx, ctx.side, { x: -0.58, z: ctx.side * 0.66 }, { x: -1.02, z: ctx.side * 0.24 }, { x: 0.18, y: -ctx.side * 0.30, z: ctx.side * 0.07 * wave(ctx.t, 4) }, a);
  look(ctx, { x: -0.035 * a, y: ctx.side * 0.08 * a, z: -ctx.side * 0.03 * a }, 1);
}

export function wristFlick(ctx) {
  const a = holdCycle(ctx.t, 0.55, 0.15) * ctx.intensity;
  poseArm(ctx, ctx.side, { x: -0.10, z: ctx.side * 0.28 }, { x: -0.30, y: -ctx.side * 0.42 }, { y: -ctx.side * 0.34, z: ctx.side * 0.60 * wave(ctx.t, 11) }, a);
  look(ctx, { y: ctx.side * 0.06 * a, z: -ctx.side * 0.015 * a }, 1);
}

export function shoulderRoll(ctx) {
  const a = holdCycle(ctx.t, 0.5, 0.08) * ctx.intensity;
  const r = Math.sin((ctx.t * 0.55) % 1 * TAU);
  poseArm(ctx, ctx.side, { x: -0.32 * r, y: ctx.side * 0.18 * r, z: ctx.side * 0.28 * r }, {}, {}, a);
  look(ctx, { x: -0.02 * Math.abs(r) * a, z: -ctx.side * 0.06 * r * a }, 1);
}

export function handOnHip(ctx) {
  const a = holdCycle(ctx.t, 0.21, 0.58) * ctx.intensity;
  poseArm(ctx, ctx.side, { x: -0.28, z: ctx.side * 0.88 }, { x: -0.28, z: ctx.side * 0.62 }, { y: -ctx.side * 0.18, z: ctx.side * 0.34 }, a);
  look(ctx, { y: -ctx.side * 0.05 * a, z: ctx.side * 0.025 * a }, 1);
}

export function adjustSleeve(ctx) {
  const a = holdCycle(ctx.t, 0.32, 0.34) * ctx.intensity;
  const workingSide = -ctx.side;
  poseArm(ctx, workingSide, { x: -0.42, z: workingSide * 0.58 }, { x: -0.76, y: -workingSide * 0.44 }, { x: -0.22, z: workingSide * (0.18 + 0.10 * wave(ctx.t, 9)) }, a);
  look(ctx, { x: 0.04 * a, y: workingSide * 0.08 * a }, 1);
}

export function brushShoulder(ctx) {
  const a = holdCycle(ctx.t, 0.35, 0.25) * ctx.intensity;
  const workingSide = -ctx.side;
  const brush = wave(ctx.t, 9.5);
  poseArm(ctx, workingSide, { x: -0.44, z: workingSide * 0.56 }, { x: -0.78, z: workingSide * 0.14 }, { z: workingSide * 0.34 * brush, y: -workingSide * 0.18 }, a);
  look(ctx, { y: -workingSide * 0.08 * a, z: workingSide * 0.025 * a }, 1);
}

export function stretchArm(ctx) {
  const a = holdCycle(ctx.t, 0.18, 0.42) * ctx.intensity;
  poseArm(ctx, ctx.side, { x: -1.08, z: ctx.side * 0.52 }, { x: -0.48, z: ctx.side * 0.08 }, { x: 0.24, y: -ctx.side * 0.16 }, a);
  look(ctx, { x: -0.04 * a, y: ctx.side * 0.12 * a, z: -ctx.side * 0.025 * a }, 1);
}

export function stretchNeck(ctx) {
  const a = holdCycle(ctx.t, 0.2, 0.5) * ctx.intensity;
  look(ctx, { x: -0.10 * a, y: -ctx.side * 0.04 * a, z: ctx.side * 0.08 * a }, 1);
}

export function shiftWeight(ctx) {
  const a = Math.sin((ctx.t * 0.22) % 1 * Math.PI) * ctx.intensity;
  look(ctx, { x: 0.005, y: -ctx.side * 0.025 * a, z: ctx.side * 0.04 * a }, 1);
  poseArm(ctx, -1, { z: ctx.REST.leftUpperArm.z - ctx.side * 0.035 * a }, {}, {});
  poseArm(ctx, 1, { z: ctx.REST.rightUpperArm.z - ctx.side * 0.035 * a }, {}, {});
}

export function sway(ctx) {
  const s = wave(ctx.t, 0.8) * ctx.intensity;
  look(ctx, { x: 0.01 * wave(ctx.t, 1.1), y: 0.035 * s, z: -0.045 * s }, 1);
  poseArm(ctx, -1, { z: ctx.REST.leftUpperArm.z - 0.045 * s }, { x: ctx.REST.leftLowerArm.x + 0.018 * s }, {});
  poseArm(ctx, 1, { z: ctx.REST.rightUpperArm.z - 0.045 * s }, { x: ctx.REST.rightLowerArm.x - 0.018 * s }, {});
}

export function crossArms(ctx) {
  const a = holdCycle(ctx.t, 0.17, 0.5) * ctx.intensity;
  poseArm(ctx, -1, { x: -0.16 * a, z: -0.34 * a }, { x: -0.36 * a, z: -0.25 * a }, { z: -0.10 * a });
  poseArm(ctx, 1, { x: -0.16 * a, z: 0.34 * a }, { x: -0.36 * a, z: 0.25 * a }, { z: 0.10 * a });
  look(ctx, { x: 0.035 * a, y: ctx.side * 0.025 * a }, 1);
}
