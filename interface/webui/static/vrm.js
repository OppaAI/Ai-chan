/**
 * vrm.js
 * Three.js VRM loader and real-time skeletal animation engine.
 * Loads Aiko.vrm, manages idle breathing/sway, gesture library (20+ animations),
 * thinking poses (chin-think, hand-near-mouth, etc.), and mouth shapes via blendshapes.
 * Expression/viseme/pose updates driven by window.aikoSetX() callbacks from WebSocket.
 *
 * Animation layers:
 *   - idle: breathing, natural sway, head look-around (computed via sine waves)
 *   - gestures: 20 contextual animations (lookAround, hairBrush, raiseHand, etc.)
 *   - thinking: 4 introspective poses (cycled during LLM inference)
 *   - blink: automatic eye closure + reopening
 *   - mouth: vowel-based visemes (aa, ih, ou, ee, oh) + RMS-driven amplitude
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';

// ── renderer ────────────────────────────────────────────────────────────────
const canvas = document.getElementById('canvas');
const vrmSide = document.getElementById('vrm-side');
const fill = document.getElementById('progress-fill');
const loadMsg = document.getElementById('load-msg');

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setClearColor(0x0a0a0f);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(8, 1, 0.1, 100);
camera.position.set(0.00, 1.36, 5.0);

const controls = new OrbitControls(camera, canvas);
controls.target.set(0.00, 1.33, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.enablePan = false;
controls.minDistance = 1.0;
controls.maxDistance = 3.2;
controls.update();

scene.add(new THREE.AmbientLight(0xc8b0ff, 0.6));
const dir = new THREE.DirectionalLight(0xffffff, 1.2);
dir.position.set(1, 3, 2);
scene.add(dir);
const rim = new THREE.DirectionalLight(0x7b4fd4, 0.4);
rim.position.set(-2, 1, -1);
scene.add(rim);
const fillL = new THREE.DirectionalLight(0xd4b0ff, 0.3);
fillL.position.set(0, -1, 2);
scene.add(fillL);
scene.add(new THREE.GridHelper(10, 20, 0x1a0a2a, 0x100820));

function resize() {
  const w = vrmSide.clientWidth;
  const h = vrmSide.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
resize();
window.addEventListener('resize', resize);

// ── VRM state ────────────────────────────────────────────────────────────────
let vrm = null;
const clock = new THREE.Clock();

const exprTargets = {};
const exprCurrent = {};
const EXPR_LERP = 6;
let exprResetTimer = null;
const EXPR_RESET_DELAY = 4000;

let blinkTimer = 0;
let blinkPhase = 'wait';
let blinkT = 0;
const BLINK_CLOSE_DUR = 0.07;
const BLINK_OPEN_DUR = 0.10;
function nextBlinkWait() { return 3.0 + Math.random() * 4.0; }
blinkTimer = nextBlinkWait();

let t = 0;
const REST = {
  leftUpperArm: { x: -0.02, y: 0.00, z: 1.28 },
  rightUpperArm: { x: -0.02, y: 0.00, z: -1.28 },
  leftLowerArm: { x: 0.12, y: 0.00, z: 0.08 },
  rightLowerArm: { x: 0.12, y: 0.00, z: -0.08 },
  leftHand: { x: 0.00, y: 0.08, z: 0.00 },
  rightHand: { x: 0.00, y: -0.08, z: 0.00 },
};

const FINGER_BONES = [
  'leftThumbProximal', 'leftThumbDistal', 'leftIndexProximal', 'leftIndexIntermediate', 'leftIndexDistal',
  'leftMiddleProximal', 'leftMiddleIntermediate', 'leftRingProximal', 'leftLittleProximal',
  'rightThumbProximal', 'rightThumbDistal', 'rightIndexProximal', 'rightIndexIntermediate', 'rightIndexDistal',
  'rightMiddleProximal', 'rightMiddleIntermediate', 'rightRingProximal', 'rightLittleProximal',
];

function getBone(name) {
  return vrm?.humanoid?.getRawBoneNode(name) ?? null;
}

let gestureState = 'none';
let gestureT = 0;
let gestureDuration = 0;
let gestureCooldown = 2.0 + Math.random() * 3.0;
let gestureTarget = {};
let lastVisemeAt = 0;

// ── Pose snapshot & blend-out system ─────────────────────────────────────
// When a gesture/thinking pose ends, we capture bone rotations into a snapshot
// and smoothly interpolate back to idle, eliminating jerky transitions.
let blendOutActive = false;
let blendOutT = 0;
const BLEND_OUT_DURATION = 0.45;
let poseSnapshot = {};

// Gesture blend-in: ramps intensity 0→1 at gesture start
let gestureBlendIn = 0;
const GESTURE_BLEND_IN_DURATION = 0.32;

// Bone names tracked for snapshot blending
const BLENDABLE_BONES = [
  'head', 'neck', 'spine', 'chest', 'hips',
  'leftUpperArm', 'rightUpperArm',
  'leftLowerArm', 'rightLowerArm',
  'leftHand', 'rightHand',
];

function capturePoseSnapshot() {
  poseSnapshot = {};
  for (const name of BLENDABLE_BONES) {
    const bone = getBone(name);
    if (bone) {
      poseSnapshot[name] = {
        x: bone.rotation.x,
        y: bone.rotation.y,
        z: bone.rotation.z,
      };
    }
  }
}

function smoothstep(edge0, edge1, x) {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function startBlendOut() {
  capturePoseSnapshot();
  blendOutActive = true;
  blendOutT = 0;
}

function applyBlendOut(dt) {
  if (!blendOutActive || !vrm?.humanoid) return false;
  blendOutT += dt;
  const progress = Math.min(1, blendOutT / BLEND_OUT_DURATION);
  const weight = 1 - smoothstep(0, 1, progress); // 1→0 smooth fade

  if (progress >= 1) {
    blendOutActive = false;
    return false;
  }

  // Interpolate each bone from snapshot toward its current (idle) value
  for (const name of BLENDABLE_BONES) {
    const snap = poseSnapshot[name];
    const bone = getBone(name);
    if (!snap || !bone) continue;
    bone.rotation.x = bone.rotation.x + (snap.x - bone.rotation.x) * weight;
    bone.rotation.y = bone.rotation.y + (snap.y - bone.rotation.y) * weight;
    bone.rotation.z = bone.rotation.z + (snap.z - bone.rotation.z) * weight;
  }
  return true;
}

const GESTURES = [
  'lookAround',
  'lookAtHand',
  'hairBrush',
  'fingerPlay',
  'meetGaze',
  'curiousTilt',
  'shiftWeight',
  'stretchNeck',
  'raiseHand',
  'chinTouch',
  'shoulderRoll',
  'sway',
  'headNod',
  'wristFlick',
  'adjustSleeve',
  'handOnHip',
  'crossArms',
  'touchCollar',
  'brushShoulder',
  'stretchArm',
  'leanIn',
  'openPalm',
  'speakingNod',
  // ── new natural gestures ──
  'earTuck',           // tucks hair behind ear
  'gentleStretch',     // subtle chest-opening stretch
  'handsClasp',        // hands clasp together briefly in front
  'thoughtfulLook',    // slow gaze upward with head tilt
];

const SPEAKING_GESTURES = [
  'leanIn', 'openPalm', 'speakingNod', 'curiousTilt', 'wristFlick',
  'emphasizePoint',    // hand rises to emphasize
  'bothHandsExplain',  // both hands open outward
];

const GESTURE_DURATION = {
  lookAround: 3.8,
  lookAtHand: 3.4,
  hairBrush: 3.2,
  fingerPlay: 3.8,
  meetGaze: 3.0,
  curiousTilt: 2.8,
  shiftWeight: 3.4,
  stretchNeck: 3.0,
  raiseHand: 3.6,
  chinTouch: 4.0,
  shoulderRoll: 2.8,
  sway: 4.5,
  headNod: 3.0,
  wristFlick: 2.2,
  adjustSleeve: 2.5,
  handOnHip: 3.2,
  crossArms: 3.0,
  touchCollar: 2.8,
  brushShoulder: 2.4,
  stretchArm: 3.5,
  leanIn: 2.8,
  openPalm: 2.4,
  speakingNod: 2.1,
  // new
  earTuck: 3.0,
  gentleStretch: 3.8,
  handsClasp: 3.2,
  thoughtfulLook: 3.5,
  emphasizePoint: 2.6,
  bothHandsExplain: 2.8,
  contemplativeNod: 3.4,
  tapFinger: 3.6,
};

function easeInOutSine(v) {
  return -(Math.cos(Math.PI * v) - 1) / 2;
}

function holdCurve(progress, inPortion = 0.28, outPortion = 0.30) {
  if (progress < inPortion) return easeInOutSine(progress / inPortion);
  if (progress > 1 - outPortion) return easeInOutSine((1 - progress) / outPortion);
  return 1;
}

function pickGesture() {
  const pool = speakingRecently() ? SPEAKING_GESTURES : GESTURES;
  const gesture = pool[Math.floor(Math.random() * pool.length)];
  const side = Math.random() < 0.5 ? -1 : 1;
  // Snapshot current pose before starting new gesture for smooth blend
  if (gestureState !== 'none') startBlendOut();
  gestureState = gesture;
  gestureT = 0;
  gestureBlendIn = 0; // reset blend-in ramp
  gestureDuration = GESTURE_DURATION[gesture] ?? 3.0;
  gestureTarget = {
    side,
    look: side * (0.28 + Math.random() * 0.22),
    tilt: side * (0.10 + Math.random() * 0.09),
    sway: side * (0.018 + Math.random() * 0.012),
  };
}

function speakingRecently() {
  return performance.now() - lastVisemeAt < 650;
}


// ── Thinking pose animation ────────────────────────────────────────────────
let thinkingPoseActive = false;
let thinkingPose = 'chinThink';
let thinkingPoseT = 0;
let thinkingPoseCycle = 0;
let thinkingBlend = 0;
let thinkingPoseBlend = 1;
const THINKING_POSES = [
  'chinThink', 'handNearMouth', 'armsFoldThink', 'lookUpThink',
  'contemplativeNod',  // new: slow deep nod while gazing off-center
  'tapFinger',         // new: index finger taps chin repeatedly
];

function pickThinkingPose() {
  const previous = thinkingPose;
  do {
    thinkingPose = THINKING_POSES[Math.floor(Math.random() * THINKING_POSES.length)];
  } while (THINKING_POSES.length > 1 && thinkingPose === previous);
  thinkingPoseT = 0;
  thinkingPoseCycle = 3.2 + Math.random() * 2.2;
  thinkingPoseBlend = 0;
}

function applyThinkingPose(dt) {
  if (!vrm?.humanoid) return false;

  // Slower blend rate (0.6s) for smoother thinking transitions
  const blendRate = 1 / 0.60;
  const prevBlend = thinkingBlend;
  thinkingBlend += ((thinkingPoseActive ? 1 : 0) - thinkingBlend) * Math.min(1, blendRate * dt);
  if (!thinkingPoseActive && thinkingBlend < 0.002) {
    thinkingBlend = 0;
    // Snapshot on deactivate so blend-out can smooth the return to idle
    if (prevBlend >= 0.002) startBlendOut();
    return false;
  }

  thinkingPoseT += dt;
  if (thinkingPoseT > thinkingPoseCycle) pickThinkingPose();
  thinkingPoseBlend += (1 - thinkingPoseBlend) * Math.min(1, dt / 0.42);

  const h = vrm.humanoid;
  const get = n => h.getRawBoneNode(n);
  const head = get('head');
  const neck = get('neck');
  const spine = get('spine');
  const chest = get('chest');
  const lUA = get('leftUpperArm');
  const rUA = get('rightUpperArm');
  const lLA = get('leftLowerArm');
  const rLA = get('rightLowerArm');
  const lH = get('leftHand');
  const rH = get('rightHand');
  const io = idleOffset;
  const blend = (base, pose, amount) => base + pose * amount;
  const settle = Math.min(1, thinkingPoseT / 0.42);
  const held = easeInOutSine(settle) * easeInOutSine(thinkingPoseBlend) * thinkingBlend;
  const pulse = Math.sin(t * 2.1) * 0.5 + 0.5;
  const micro = Math.sin(t * 5.0) * 0.018;

  if (spine) spine.rotation.x = blend(io.spine.x, 0.025, held);
  if (chest) chest.rotation.x = blend(io.chest.x, 0.035, held);

  switch (thinkingPose) {
    case 'chinThink':
      if (head) { head.rotation.x = blend(io.head.x, 0.095 + micro, held); head.rotation.y = blend(io.head.y, -0.08, held); head.rotation.z = blend(io.head.z, -0.035, held); }
      if (neck) { neck.rotation.x = blend(io.neck.x, 0.055, held); neck.rotation.y = blend(io.neck.y, -0.045, held); }
      if (rUA) { rUA.rotation.x = blend(REST.rightUpperArm.x + io.rUA.x, -0.58, held); rUA.rotation.z = blend(REST.rightUpperArm.z + io.rUA.z, 0.68, held); }
      if (rLA) { rLA.rotation.x = blend(REST.rightLowerArm.x + io.rLA.x, -1.08, held); rLA.rotation.z = blend(REST.rightLowerArm.z + io.rLA.z, 0.12, held); }
      if (rH) { rH.rotation.x = blend(REST.rightHand.x + io.rH.x, -0.22 + pulse * 0.04, held); rH.rotation.y = blend(REST.rightHand.y + io.rH.y, -0.24, held); }
      applyFingerCurl(1, 0.35 * held);
      break;

    case 'handNearMouth':
      if (head) { head.rotation.x = blend(io.head.x, 0.05, held); head.rotation.y = blend(io.head.y, 0.10 + micro, held); }
      if (neck) neck.rotation.y = blend(io.neck.y, 0.06, held);
      if (lUA) { lUA.rotation.x = blend(REST.leftUpperArm.x + io.lUA.x, -0.62, held); lUA.rotation.z = blend(REST.leftUpperArm.z + io.lUA.z, -0.72, held); }
      if (lLA) { lLA.rotation.x = blend(REST.leftLowerArm.x + io.lLA.x, -0.98, held); lLA.rotation.z = blend(REST.leftLowerArm.z + io.lLA.z, -0.16, held); }
      if (lH) { lH.rotation.x = blend(REST.leftHand.x + io.lH.x, -0.12, held); lH.rotation.y = blend(REST.leftHand.y + io.lH.y, 0.28 + pulse * 0.04, held); }
      applyFingerCurl(-1, 0.28 * held);
      break;

    case 'armsFoldThink':
      if (head) { head.rotation.x = blend(io.head.x, 0.07 + micro, held); head.rotation.z = blend(io.head.z, 0.045, held); }
      if (neck) neck.rotation.z = blend(io.neck.z, 0.025, held);
      if (lUA) { lUA.rotation.x = blend(REST.leftUpperArm.x + io.lUA.x, -0.22, held); lUA.rotation.z = blend(REST.leftUpperArm.z + io.lUA.z, -0.62, held); }
      if (rUA) { rUA.rotation.x = blend(REST.rightUpperArm.x + io.rUA.x, -0.22, held); rUA.rotation.z = blend(REST.rightUpperArm.z + io.rUA.z, 0.62, held); }
      if (lLA) { lLA.rotation.x = blend(REST.leftLowerArm.x + io.lLA.x, -0.62, held); lLA.rotation.z = blend(REST.leftLowerArm.z + io.lLA.z, -0.46, held); }
      if (rLA) { rLA.rotation.x = blend(REST.rightLowerArm.x + io.rLA.x, -0.62, held); rLA.rotation.z = blend(REST.rightLowerArm.z + io.rLA.z, 0.46, held); }
      break;

    case 'lookUpThink':
      if (head) { head.rotation.x = blend(io.head.x, -0.10 + micro, held); head.rotation.y = blend(io.head.y, 0.18, held); head.rotation.z = blend(io.head.z, -0.025, held); }
      if (neck) { neck.rotation.x = blend(io.neck.x, -0.06, held); neck.rotation.y = blend(io.neck.y, 0.08, held); }
      if (rUA) { rUA.rotation.x = blend(REST.rightUpperArm.x + io.rUA.x, -0.35, held); rUA.rotation.z = blend(REST.rightUpperArm.z + io.rUA.z, 0.42, held); }
      if (rLA) { rLA.rotation.x = blend(REST.rightLowerArm.x + io.rLA.x, -0.72, held); }
      if (rH) rH.rotation.z = blend(REST.rightHand.z + io.rH.z, 0.18 + pulse * 0.05, held);
      break;

    case 'contemplativeNod':
      // Slow deep nod while gazing slightly off-center
      {
        const nodPhase = Math.sin(thinkingPoseT * 1.2) * held;
        if (head) { head.rotation.x = blend(io.head.x, 0.08 + nodPhase * 0.06, held); head.rotation.y = blend(io.head.y, -0.12 + micro, held); head.rotation.z = blend(io.head.z, -0.03, held); }
        if (neck) { neck.rotation.x = blend(io.neck.x, 0.04 + nodPhase * 0.03, held); neck.rotation.y = blend(io.neck.y, -0.06, held); }
        if (rUA) { rUA.rotation.x = blend(REST.rightUpperArm.x + io.rUA.x, -0.25, held); rUA.rotation.z = blend(REST.rightUpperArm.z + io.rUA.z, 0.30, held); }
        if (rLA) { rLA.rotation.x = blend(REST.rightLowerArm.x + io.rLA.x, -0.45, held); }
        if (rH) { rH.rotation.x = blend(REST.rightHand.x + io.rH.x, -0.10, held); }
      }
      break;

    case 'tapFinger':
      // Index finger rhythmically taps chin area
      {
        const tapRate = Math.sin(thinkingPoseT * 5.5); // rhythmic tapping
        const tapIntensity = Math.max(0, tapRate) * held; // only positive half = tap down
        if (head) { head.rotation.x = blend(io.head.x, 0.08 + micro, held); head.rotation.y = blend(io.head.y, 0.06, held); head.rotation.z = blend(io.head.z, 0.03, held); }
        if (neck) { neck.rotation.x = blend(io.neck.x, 0.04, held); neck.rotation.y = blend(io.neck.y, 0.03, held); }
        if (rUA) { rUA.rotation.x = blend(REST.rightUpperArm.x + io.rUA.x, -0.52, held); rUA.rotation.z = blend(REST.rightUpperArm.z + io.rUA.z, 0.62, held); }
        if (rLA) { rLA.rotation.x = blend(REST.rightLowerArm.x + io.rLA.x, -1.02, held); rLA.rotation.z = blend(REST.rightLowerArm.z + io.rLA.z, 0.10, held); }
        if (rH) { rH.rotation.x = blend(REST.rightHand.x + io.rH.x, -0.18 - tapIntensity * 0.08, held); rH.rotation.y = blend(REST.rightHand.y + io.rH.y, -0.20, held); }
        applyFingerCurl(1, 0.40 * held);
      }
      break;
  }

  return true;
}

function resetFingerCurl() {
  for (const name of FINGER_BONES) {
    const bone = getBone(name);
    if (bone) bone.rotation.z = 0;
  }
}

function applyFingerCurl(side, intensity) {
  const prefix = side < 0 ? 'left' : 'right';
  const pulse = 0.5 + Math.sin(t * 8.0) * 0.5;
  const curl = intensity * (0.12 + pulse * 0.14);
  for (const name of FINGER_BONES) {
    if (!name.startsWith(prefix)) continue;
    const bone = getBone(name);
    if (bone) bone.rotation.z = curl;
  }
}

// ── Idle animation helpers ─────────────────────────────────────────────────
// We store idle offsets separately so gestures can blend cleanly
let idleOffset = {
  head: { x: 0, y: 0, z: 0 },
  neck: { x: 0, y: 0, z: 0 },
  spine: { x: 0, y: 0, z: 0 },
  chest: { x: 0, y: 0, z: 0 },
  hips: { x: 0, y: 0, z: 0, px: 0 },
  lUA: { x: 0, y: 0, z: 0 },
  rUA: { x: 0, y: 0, z: 0 },
  lLA: { x: 0, y: 0, z: 0 },
  rLA: { x: 0, y: 0, z: 0 },
  lH: { x: 0, y: 0, z: 0 },
  rH: { x: 0, y: 0, z: 0 },
};

function computeIdleOffsets(dt) {
  t += dt;
  const io = idleOffset;

  // Breathing - chest and spine
  const breath = Math.sin(t * 0.83) * 0.013;
  io.chest.x = breath;
  io.spine.x = breath * 0.5;

  // Hips - gentle sway and bob
  io.hips.z = Math.sin(t * 0.41) * 0.012;
  io.hips.x = Math.sin(t * 0.67) * 0.008;
  io.hips.px = Math.sin(t * 0.41) * 0.003;

  // Head - natural idle look around (more varied)
  io.head.y = Math.sin(t * 0.31) * 0.055 + Math.sin(t * 1.13) * 0.012 + Math.sin(t * 0.17) * 0.025;
  io.head.z = Math.sin(t * 0.27 + 1.1) * 0.018 + Math.sin(t * 0.71) * 0.006;
  io.head.x = Math.sin(t * 0.53) * 0.012 + Math.sin(t * 0.19) * 0.008;

  // Neck follows head with damping
  io.neck.y = io.head.y * 0.3 + Math.sin(t * 0.43) * 0.015;
  io.neck.z = io.head.z * 0.3;
  io.neck.x = io.head.x * 0.4 + Math.sin(t * 0.61) * 0.010;

  // ── Arms: much more natural idle movement ────────────────────────────────
  // Upper arms: gentle swaying, as if relaxed at sides but alive
  io.lUA.x = Math.sin(t * 0.47) * 0.025 + Math.sin(t * 0.23) * 0.015;
  io.lUA.y = Math.sin(t * 0.33) * 0.012 + Math.sin(t * 0.71) * 0.008;
  io.lUA.z = Math.sin(t * 0.41) * 0.022 + Math.sin(t * 0.19 + 0.5) * 0.018;

  io.rUA.x = Math.sin(t * 0.53 + 0.9) * 0.025 + Math.sin(t * 0.29 + 1.1) * 0.015;
  io.rUA.y = Math.sin(t * 0.35 + 0.4) * 0.012 + Math.sin(t * 0.67 + 0.3) * 0.008;
  io.rUA.z = Math.sin(t * 0.37 + 0.7) * 0.022 + Math.sin(t * 0.21 + 0.8) * 0.018;

  // Lower arms: subtle forearm rotation, like hands naturally shifting
  io.lLA.x = Math.sin(t * 0.61) * 0.018 + Math.sin(t * 0.31) * 0.012;
  io.lLA.z = Math.sin(t * 0.43) * 0.014 + Math.sin(t * 0.17 + 0.3) * 0.010;
  io.lLA.y = Math.sin(t * 0.27) * 0.008;

  io.rLA.x = Math.sin(t * 0.57 + 1.4) * 0.018 + Math.sin(t * 0.33 + 0.6) * 0.012;
  io.rLA.z = Math.sin(t * 0.51 + 0.5) * 0.014 + Math.sin(t * 0.19 + 0.9) * 0.010;
  io.rLA.y = Math.sin(t * 0.29 + 0.8) * 0.008;

  // Hands: gentle wrist rolls and finger-like micro-movements
  io.lH.x = Math.sin(t * 0.39) * 0.015;
  io.lH.y = Math.sin(t * 0.33) * 0.020 + Math.sin(t * 0.71) * 0.010;
  io.lH.z = Math.sin(t * 0.45) * 0.012;

  io.rH.x = Math.sin(t * 0.37 + 0.5) * 0.015;
  io.rH.y = Math.sin(t * 0.29 + 1.2) * 0.020 + Math.sin(t * 0.73 + 0.4) * 0.010;
  io.rH.z = Math.sin(t * 0.43 + 0.7) * 0.012;

  return io;
}

function applyIdle(dt) {
  if (!vrm?.humanoid) return;
  const h = vrm.humanoid;
  const get = n => h.getRawBoneNode(n);
  const io = computeIdleOffsets(dt);

  const chest = get('chest');
  const spine = get('spine');
  if (chest) chest.rotation.x = io.chest.x;
  if (spine) spine.rotation.x = io.spine.x;

  const hips = get('hips');
  if (hips) {
    hips.rotation.z = io.hips.z;
    hips.rotation.x = io.hips.x;
    hips.position.x = io.hips.px;
  }

  const head = get('head');
  if (head) {
    head.rotation.y = io.head.y;
    head.rotation.z = io.head.z;
    head.rotation.x = io.head.x;
  }
  const neck = get('neck');
  if (neck) {
    neck.rotation.y = io.neck.y;
    neck.rotation.z = io.neck.z;
    neck.rotation.x = io.neck.x;
  }

  const lUA = get('leftUpperArm');
  const rUA = get('rightUpperArm');
  const lLA = get('leftLowerArm');
  const rLA = get('rightLowerArm');
  const lH = get('leftHand');
  const rH = get('rightHand');

  if (lUA) {
    lUA.rotation.x = REST.leftUpperArm.x + io.lUA.x;
    lUA.rotation.y = REST.leftUpperArm.y + io.lUA.y;
    lUA.rotation.z = REST.leftUpperArm.z + io.lUA.z;
  }
  if (rUA) {
    rUA.rotation.x = REST.rightUpperArm.x + io.rUA.x;
    rUA.rotation.y = REST.rightUpperArm.y + io.rUA.y;
    rUA.rotation.z = REST.rightUpperArm.z + io.rUA.z;
  }
  if (lLA) {
    lLA.rotation.x = REST.leftLowerArm.x + io.lLA.x;
    lLA.rotation.y = REST.leftLowerArm.y + io.lLA.y;
    lLA.rotation.z = REST.leftLowerArm.z + io.lLA.z;
  }
  if (rLA) {
    rLA.rotation.x = REST.rightLowerArm.x + io.rLA.x;
    rLA.rotation.y = REST.rightLowerArm.y + io.rLA.y;
    rLA.rotation.z = REST.rightLowerArm.z + io.rLA.z;
  }
  if (lH) {
    lH.rotation.x = REST.leftHand.x + io.lH.x;
    lH.rotation.y = REST.leftHand.y + io.lH.y;
    lH.rotation.z = REST.leftHand.z + io.lH.z;
  }
  if (rH) {
    rH.rotation.x = REST.rightHand.x + io.rH.x;
    rH.rotation.y = REST.rightHand.y + io.rH.y;
    rH.rotation.z = REST.rightHand.z + io.rH.z;
  }
}

function applyGestures(dt) {
  if (!vrm?.humanoid) return;

  if (gestureState === 'none') {
    if (speakingRecently()) gestureCooldown = Math.min(gestureCooldown, 1.0);
    gestureCooldown -= dt;
    if (gestureCooldown <= 0) {
      if (speakingRecently()) {
        pickGesture();
        gestureCooldown = 2.8 + Math.random() * 2.4;
      } else {
        pickGesture();
        gestureCooldown = 4.5 + Math.random() * 7.0;
      }
    }
    return;
  }

  // Blend-in ramp: smoothly increase gesture intensity from 0→1
  gestureBlendIn = Math.min(1, gestureBlendIn + dt / GESTURE_BLEND_IN_DURATION);
  const blendInFactor = smoothstep(0, 1, gestureBlendIn);

  gestureT += dt;
  const progress = Math.min(1, gestureT / gestureDuration);
  const eased = easeInOutSine(progress);
  const held = holdCurve(progress) * blendInFactor;
  const intensity = Math.sin(progress * Math.PI) * blendInFactor;
  const side = gestureTarget.side;

  const head = getBone('head');
  const neck = getBone('neck');
  const spine = getBone('spine');
  const chest = getBone('chest');
  const hips = getBone('hips');
  const lUA = getBone('leftUpperArm');
  const rUA = getBone('rightUpperArm');
  const lLA = getBone('leftLowerArm');
  const rLA = getBone('rightLowerArm');
  const lH = getBone('leftHand');
  const rH = getBone('rightHand');

  // Helper to blend gesture with idle base
  const blend = (base, gesture, amt) => base + gesture * amt;
  const io = idleOffset;

  switch (gestureState) {

    case 'lookAround':
      if (head) {
        head.rotation.y = blend(io.head.y, gestureTarget.look * 1.2, held);
        head.rotation.x = blend(io.head.x, Math.sin(eased * Math.PI) * 0.04, held);
      }
      if (neck) {
        neck.rotation.y = blend(io.neck.y, gestureTarget.look * 0.45, held);
        neck.rotation.z = blend(io.neck.z, gestureTarget.tilt * 0.3, held);
      }
      if (spine) spine.rotation.y = blend(0, gestureTarget.look * 0.08, held);
      break;

    case 'lookAtHand': {
      const s = side < 0 ? -1 : 1;
    
      let handBone = null;
    
      // -----------------------------
      // Pose arm
      // -----------------------------
      if (s < 0) {
        handBone = lH;
    
        if (lUA) {
          lUA.rotation.x = blend(
            REST.leftUpperArm.x + io.lUA.x,
            -0.45 * intensity,
            1
          );
    
          lUA.rotation.z = blend(
            REST.leftUpperArm.z + io.lUA.z,
            -0.70 * intensity,
            1
          );
        }
    
        if (lLA) {
          lLA.rotation.x = blend(
            REST.leftLowerArm.x + io.lLA.x,
            -0.95 * intensity,
            1
          );
    
          lLA.rotation.z = blend(
            REST.leftLowerArm.z + io.lLA.z,
            -0.15 * intensity,
            1
          );
        }
    
        if (lH) {
          lH.rotation.x = blend(
            REST.leftHand.x + io.lH.x,
            0.25 * intensity,
            1
          );
    
          lH.rotation.y = blend(
            REST.leftHand.y + io.lH.y,
            0.45 * intensity,
            1
          );
        }
      } else {
        handBone = rH;
    
        if (rUA) {
          rUA.rotation.x = blend(
            REST.rightUpperArm.x + io.rUA.x,
            -0.45 * intensity,
            1
          );
    
          rUA.rotation.z = blend(
            REST.rightUpperArm.z + io.rUA.z,
            0.70 * intensity,
            1
          );
        }
    
        if (rLA) {
          rLA.rotation.x = blend(
            REST.rightLowerArm.x + io.rLA.x,
            -0.95 * intensity,
            1
          );
    
          rLA.rotation.z = blend(
            REST.rightLowerArm.z + io.rLA.z,
            0.15 * intensity,
            1
          );
        }
    
        if (rH) {
          rH.rotation.x = blend(
            REST.rightHand.x + io.rH.x,
            0.25 * intensity,
            1
          );
    
          rH.rotation.y = blend(
            REST.rightHand.y + io.rH.y,
            -0.45 * intensity,
            1
          );
        }
      }
    
      // -----------------------------
      // True head tracking
      // -----------------------------
      if (head && handBone) {
        const handPos = new THREE.Vector3();
        const headPos = new THREE.Vector3();
    
        handBone.getWorldPosition(handPos);
        head.getWorldPosition(headPos);
    
        const dir = handPos.sub(headPos).normalize();
    
        const yaw = Math.atan2(dir.x, dir.z);
    
        const pitch = Math.atan2(
          -dir.y,
          Math.sqrt(dir.x * dir.x + dir.z * dir.z)
        );
    
        // Neck follows partially
        if (neck) {
          neck.rotation.y = blend(
            io.neck.y,
            yaw * 0.35 * intensity,
            1
          );
    
          neck.rotation.x = blend(
            io.neck.x,
            pitch * 0.35 * intensity,
            1
          );
        }
    
        // Head follows more strongly
        head.rotation.y = blend(
          io.head.y,
          yaw * 0.75 * intensity,
          1
        );
    
        head.rotation.x = blend(
          io.head.x,
          pitch * 0.75 * intensity,
          1
        );
      }
    
      break;
    }

    case 'hairBrush':
      if (side < 0) {
        if (lUA) { lUA.rotation.z = blend(REST.leftUpperArm.z + io.lUA.z, -intensity * 0.90, 1); lUA.rotation.x = blend(REST.leftUpperArm.x + io.lUA.x, -intensity * 0.45, 1); }
        if (lLA) { lLA.rotation.x = blend(REST.leftLowerArm.x + io.lLA.x, -intensity * 0.80, 1); lLA.rotation.z = blend(REST.leftLowerArm.z + io.lLA.z, -intensity * 0.32, 1); }
        if (lH) { lH.rotation.y = blend(REST.leftHand.y + io.lH.y, intensity * 0.38, 1); lH.rotation.z = blend(REST.leftHand.z + io.lH.z, -intensity * 0.22, 1); }
      } else {
        if (rUA) { rUA.rotation.z = blend(REST.rightUpperArm.z + io.rUA.z, intensity * 0.90, 1); rUA.rotation.x = blend(REST.rightUpperArm.x + io.rUA.x, -intensity * 0.45, 1); }
        if (rLA) { rLA.rotation.x = blend(REST.rightLowerArm.x + io.rLA.x, -intensity * 0.80, 1); rLA.rotation.z = blend(REST.rightLowerArm.z + io.rLA.z, intensity * 0.32, 1); }
        if (rH) { rH.rotation.y = blend(REST.rightHand.y + io.rH.y, -intensity * 0.38, 1); rH.rotation.z = blend(REST.rightHand.z + io.rH.z, intensity * 0.22, 1); }
      }
      if (head) { head.rotation.z = blend(io.head.z, -side * intensity * 0.07, 1); head.rotation.y = blend(io.head.y, side * intensity * 0.05, 1); }
      break;

    case 'fingerPlay':
      if (head) { head.rotation.x = blend(io.head.x, intensity * 0.08, 1); head.rotation.y = blend(io.head.y, side * intensity * 0.14, 1); }
      if (side < 0) {
        if (lLA) { lLA.rotation.x = blend(REST.leftLowerArm.x + io.lLA.x, -intensity * 0.38, 1); }
        if (lH) { lH.rotation.y = blend(REST.leftHand.y + io.lH.y, intensity * 0.26, 1); lH.rotation.z = blend(REST.leftHand.z + io.lH.z, Math.sin(gestureT * 5.0) * 0.10 * intensity, 1); }
      } else {
        if (rLA) { rLA.rotation.x = blend(REST.rightLowerArm.x + io.rLA.x, -intensity * 0.38, 1); }
        if (rH) { rH.rotation.y = blend(REST.rightHand.y + io.rH.y, -intensity * 0.26, 1); rH.rotation.z = blend(REST.rightHand.z + io.rH.z, -Math.sin(gestureT * 5.0) * 0.10 * intensity, 1); }
      }
      applyFingerCurl(side, intensity);
      break;

    case 'meetGaze':
      if (head) {
        head.rotation.y = blend(io.head.y, 0, held * 0.85);
        head.rotation.z = blend(io.head.z, 0, held * 0.80);
        head.rotation.x = blend(io.head.x, 0.025, held * 0.5);
      }
      if (neck) {
        neck.rotation.y = blend(io.neck.y, 0, held * 0.60);
        neck.rotation.z = blend(io.neck.z, 0, held * 0.60);
      }
      break;

    case 'curiousTilt':
      if (head) {
        head.rotation.z = blend(io.head.z, gestureTarget.tilt * 1.3, held);
        head.rotation.x = blend(io.head.x, -0.025 * intensity, 1);
        head.rotation.y = blend(io.head.y, side * 0.06, held);
      }
      if (neck) {
        neck.rotation.z = blend(io.neck.z, gestureTarget.tilt * 0.60, held);
        neck.rotation.y = blend(io.neck.y, side * 0.04, held);
      }
      break;

    case 'shiftWeight':
      if (hips) {
        hips.position.x = blend(io.hips.px, side * Math.sin(eased * Math.PI) * 0.022, 1);
        hips.rotation.z = blend(io.hips.z, side * Math.sin(eased * Math.PI) * 0.025, 1);
      }
      if (spine) spine.rotation.z = blend(0, -side * Math.sin(eased * Math.PI) * 0.020, 1);
      if (chest) chest.rotation.z = blend(0, -side * Math.sin(eased * Math.PI) * 0.014, 1);
      if (head) head.rotation.z = blend(io.head.z, side * Math.sin(eased * Math.PI) * 0.018, 1);
      break;

    case 'stretchNeck':
      if (neck) { neck.rotation.x = blend(io.neck.x, -intensity * 0.10, 1); neck.rotation.z = blend(io.neck.z, gestureTarget.tilt * 0.5, held); }
      if (head) { head.rotation.x = blend(io.head.x, -intensity * 0.08, 1); head.rotation.z = blend(io.head.z, gestureTarget.tilt * 0.3, held); }
      if (chest) chest.rotation.x = blend(io.chest.x, -intensity * 0.04, 1);
      break;

    case 'raiseHand':
      if (side < 0) {
        if (lUA) { lUA.rotation.x = blend(REST.leftUpperArm.x + io.lUA.x, -intensity * 0.70, 1); lUA.rotation.z = blend(REST.leftUpperArm.z + io.lUA.z, -intensity * 0.40, 1); }
        if (lLA) { lLA.rotation.x = blend(REST.leftLowerArm.x + io.lLA.x, -intensity * 0.90, 1); lLA.rotation.z = blend(REST.leftLowerArm.z + io.lLA.z, -intensity * 0.14, 1); }
        if (lH) { lH.rotation.x = blend(REST.leftHand.x + io.lH.x, intensity * 0.12, 1); }
      } else {
        if (rUA) { rUA.rotation.x = blend(REST.rightUpperArm.x + io.rUA.x, -intensity * 0.70, 1); rUA.rotation.z = blend(REST.rightUpperArm.z + io.rUA.z, intensity * 0.40, 1); }
        if (rLA) { rLA.rotation.x = blend(REST.rightLowerArm.x + io.rLA.x, -intensity * 0.90, 1); rLA.rotation.z = blend(REST.rightLowerArm.z + io.rLA.z, intensity * 0.14, 1); }
        if (rH) { rH.rotation.x = blend(REST.rightHand.x + io.rH.x, intensity * 0.12, 1); }
      }
      if (head) { head.rotation.y = blend(io.head.y, side * intensity * 0.10, 1); head.rotation.x = blend(io.head.x, -intensity * 0.03, 1); }
      break;

    case 'chinTouch':
      if (side < 0) {
        if (lUA) { lUA.rotation.x = blend(REST.leftUpperArm.x + io.lUA.x, -intensity * 0.60, 1); lUA.rotation.z = blend(REST.leftUpperArm.z + io.lUA.z, -intensity * 0.60, 1); }
        if (lLA) { lLA.rotation.x = blend(REST.leftLowerArm.x + io.lLA.x, -intensity * 1.05, 1); lLA.rotation.z = blend(REST.leftLowerArm.z + io.lLA.z, -intensity * 0.10, 1); }
        if (lH) { lH.rotation.x = blend(REST.leftHand.x + io.lH.x, -intensity * 0.18, 1); lH.rotation.y = blend(REST.leftHand.y + io.lH.y, intensity * 0.20, 1); }
      } else {
        if (rUA) { rUA.rotation.x = blend(REST.rightUpperArm.x + io.rUA.x, -intensity * 0.60, 1); rUA.rotation.z = blend(REST.rightUpperArm.z + io.rUA.z, intensity * 0.60, 1); }
        if (rLA) { rLA.rotation.x = blend(REST.rightLowerArm.x + io.rLA.x, -intensity * 1.05, 1); rLA.rotation.z = blend(REST.rightLowerArm.z + io.rLA.z, intensity * 0.10, 1); }
        if (rH) { rH.rotation.x = blend(REST.rightHand.x + io.rH.x, -intensity * 0.18, 1); rH.rotation.y = blend(REST.rightHand.y + io.rH.y, -intensity * 0.20, 1); }
      }
      if (head) { head.rotation.x = blend(io.head.x, intensity * 0.06, 1); head.rotation.z = blend(io.head.z, side * intensity * 0.05, 1); }
      if (neck) neck.rotation.x = blend(io.neck.x, intensity * 0.04, 1);
      break;

    case 'shoulderRoll':
      if (side < 0) {
        if (lUA) {
          const roll = Math.sin(progress * Math.PI * 1.5) * intensity;
          lUA.rotation.x = blend(REST.leftUpperArm.x + io.lUA.x, -roll * 0.28, 1);
          lUA.rotation.z = blend(REST.leftUpperArm.z + io.lUA.z, roll * 0.18, 1);
          lUA.rotation.y = blend(REST.leftUpperArm.y + io.lUA.y, roll * 0.10, 1);
        }
      } else {
        if (rUA) {
          const roll = Math.sin(progress * Math.PI * 1.5) * intensity;
          rUA.rotation.x = blend(REST.rightUpperArm.x + io.rUA.x, -roll * 0.28, 1);
          rUA.rotation.z = blend(REST.rightUpperArm.z + io.rUA.z, -roll * 0.18, 1);
          rUA.rotation.y = blend(REST.rightUpperArm.y + io.rUA.y, -roll * 0.10, 1);
        }
      }
      if (head) head.rotation.z = blend(io.head.z, -side * Math.sin(progress * Math.PI * 1.5) * intensity * 0.06, 1);
      if (neck) neck.rotation.z = blend(io.neck.z, -side * Math.sin(progress * Math.PI * 1.5) * intensity * 0.04, 1);
      break;

    case 'sway':
      {
        const swayVal = Math.sin(progress * Math.PI * 2) * intensity;
        if (hips) { hips.position.x = blend(io.hips.px, swayVal * 0.020, 1); hips.rotation.z = blend(io.hips.z, swayVal * 0.022, 1); }
        if (spine) spine.rotation.z = blend(0, swayVal * -0.014, 1);
        if (chest) chest.rotation.z = blend(0, swayVal * -0.010, 1);
        if (head) head.rotation.z = blend(io.head.z, swayVal * -0.016, 1);
        if (neck) neck.rotation.z = blend(io.neck.z, swayVal * -0.010, 1);
        if (lUA) lUA.rotation.z = blend(REST.leftUpperArm.z + io.lUA.z, -swayVal * 0.06, 1);
        if (rUA) rUA.rotation.z = blend(REST.rightUpperArm.z + io.rUA.z, -swayVal * 0.06, 1);
      }
      break;

    case 'headNod':
      {
        const nod = Math.sin(progress * Math.PI * 3.5) * intensity * 0.14;
        if (head) { head.rotation.x = blend(io.head.x, nod, 1); }
        if (neck) { neck.rotation.x = blend(io.neck.x, nod * 0.5, 1); }
        if (head) head.rotation.y = blend(io.head.y, 0, held * 0.4);
      }
      break;

    case 'wristFlick':
      if (side < 0) {
        if (lLA) { lLA.rotation.y = blend(REST.leftLowerArm.y + io.lLA.y, intensity * 0.55, 1); }
        if (lH) { lH.rotation.z = blend(REST.leftHand.z + io.lH.z, intensity * 0.50, 1); lH.rotation.y = blend(REST.leftHand.y + io.lH.y, intensity * 0.30, 1); }
      } else {
        if (rLA) { rLA.rotation.y = blend(REST.rightLowerArm.y + io.rLA.y, -intensity * 0.55, 1); }
        if (rH) { rH.rotation.z = blend(REST.rightHand.z + io.rH.z, -intensity * 0.50, 1); rH.rotation.y = blend(REST.rightHand.y + io.rH.y, -intensity * 0.30, 1); }
      }
      if (head) head.rotation.y = blend(io.head.y, side * intensity * 0.08, 1);
      break;

    // ── new natural idle gestures ──────────────────────────────────────────

    case 'adjustSleeve':
      // Hand reaches to opposite forearm, tugs at sleeve
      if (side < 0) {
        if (rUA) { rUA.rotation.x = blend(REST.rightUpperArm.x + io.rUA.x, -intensity * 0.35, 1); rUA.rotation.z = blend(REST.rightUpperArm.z + io.rUA.z, intensity * 0.25, 1); }
        if (rLA) { rLA.rotation.x = blend(REST.rightLowerArm.x + io.rLA.x, -intensity * 0.55, 1); rLA.rotation.y = blend(REST.rightLowerArm.y + io.rLA.y, -intensity * 0.40, 1); }
        if (rH) { rH.rotation.x = blend(REST.rightHand.x + io.rH.x, -intensity * 0.20, 1); rH.rotation.z = blend(REST.rightHand.z + io.rH.z, -intensity * 0.15, 1); }
      } else {
        if (lUA) { lUA.rotation.x = blend(REST.leftUpperArm.x + io.lUA.x, -intensity * 0.35, 1); lUA.rotation.z = blend(REST.leftUpperArm.z + io.lUA.z, -intensity * 0.25, 1); }
        if (lLA) { lLA.rotation.x = blend(REST.leftLowerArm.x + io.lLA.x, -intensity * 0.55, 1); lLA.rotation.y = blend(REST.leftLowerArm.y + io.lLA.y, intensity * 0.40, 1); }
        if (lH) { lH.rotation.x = blend(REST.leftHand.x + io.lH.x, -intensity * 0.20, 1); lH.rotation.z = blend(REST.leftHand.z + io.lH.z, intensity * 0.15, 1); }
      }
      if (head) head.rotation.x = blend(io.head.x, intensity * 0.05, 1);
      break;

    case 'handOnHip':
      // One hand rests on hip, elbow out
      if (side < 0) {
        if (lUA) { lUA.rotation.z = blend(REST.leftUpperArm.z + io.lUA.z, -intensity * 0.55, 1); lUA.rotation.x = blend(REST.leftUpperArm.x + io.lUA.x, -intensity * 0.20, 1); }
        if (lLA) { lLA.rotation.z = blend(REST.leftLowerArm.z + io.lLA.z, -intensity * 0.35, 1); lLA.rotation.x = blend(REST.leftLowerArm.x + io.lLA.x, -intensity * 0.15, 1); }
        if (lH) { lH.rotation.z = blend(REST.leftHand.z + io.lH.z, -intensity * 0.25, 1); lH.rotation.y = blend(REST.leftHand.y + io.lH.y, intensity * 0.10, 1); }
      } else {
        if (rUA) { rUA.rotation.z = blend(REST.rightUpperArm.z + io.rUA.z, intensity * 0.55, 1); rUA.rotation.x = blend(REST.rightUpperArm.x + io.rUA.x, -intensity * 0.20, 1); }
        if (rLA) { rLA.rotation.z = blend(REST.rightLowerArm.z + io.rLA.z, intensity * 0.35, 1); rLA.rotation.x = blend(REST.rightLowerArm.x + io.rLA.x, -intensity * 0.15, 1); }
        if (rH) { rH.rotation.z = blend(REST.rightHand.z + io.rH.z, intensity * 0.25, 1); rH.rotation.y = blend(REST.rightHand.y + io.rH.y, -intensity * 0.10, 1); }
      }
      if (head) head.rotation.y = blend(io.head.y, -side * intensity * 0.06, 1);
      if (spine) spine.rotation.z = blend(0, side * intensity * 0.012, 1);
      break;

    case 'crossArms':
      // Subtle arm-cross: both arms shift inward slightly
      {
        const cross = intensity * 0.35;
        if (lUA) { lUA.rotation.z = blend(REST.leftUpperArm.z + io.lUA.z, -cross * 0.40, 1); lUA.rotation.x = blend(REST.leftUpperArm.x + io.lUA.x, -cross * 0.15, 1); }
        if (rUA) { rUA.rotation.z = blend(REST.rightUpperArm.z + io.rUA.z, cross * 0.40, 1); rUA.rotation.x = blend(REST.rightUpperArm.x + io.rUA.x, -cross * 0.15, 1); }
        if (lLA) { lLA.rotation.z = blend(REST.leftLowerArm.z + io.lLA.z, -cross * 0.30, 1); lLA.rotation.x = blend(REST.leftLowerArm.x + io.lLA.x, -cross * 0.25, 1); }
        if (rLA) { rLA.rotation.z = blend(REST.rightLowerArm.z + io.rLA.z, cross * 0.30, 1); rLA.rotation.x = blend(REST.rightLowerArm.x + io.rLA.x, -cross * 0.25, 1); }
        if (head) head.rotation.x = blend(io.head.x, intensity * 0.04, 1);
      }
      break;

    case 'touchCollar':
      // Hand comes up to collar/neck area
      if (side < 0) {
        if (lUA) { lUA.rotation.x = blend(REST.leftUpperArm.x + io.lUA.x, -intensity * 0.50, 1); lUA.rotation.z = blend(REST.leftUpperArm.z + io.lUA.z, -intensity * 0.45, 1); }
        if (lLA) { lLA.rotation.x = blend(REST.leftLowerArm.x + io.lLA.x, -intensity * 0.85, 1); lLA.rotation.z = blend(REST.leftLowerArm.z + io.lLA.z, -intensity * 0.20, 1); }
        if (lH) { lH.rotation.x = blend(REST.leftHand.x + io.lH.x, intensity * 0.15, 1); lH.rotation.y = blend(REST.leftHand.y + io.lH.y, intensity * 0.25, 1); }
      } else {
        if (rUA) { rUA.rotation.x = blend(REST.rightUpperArm.x + io.rUA.x, -intensity * 0.50, 1); rUA.rotation.z = blend(REST.rightUpperArm.z + io.rUA.z, intensity * 0.45, 1); }
        if (rLA) { rLA.rotation.x = blend(REST.rightLowerArm.x + io.rLA.x, -intensity * 0.85, 1); rLA.rotation.z = blend(REST.rightLowerArm.z + io.rLA.z, intensity * 0.20, 1); }
        if (rH) { rH.rotation.x = blend(REST.rightHand.x + io.rH.x, intensity * 0.15, 1); rH.rotation.y = blend(REST.rightHand.y + io.rH.y, -intensity * 0.25, 1); }
      }
      if (head) { head.rotation.x = blend(io.head.x, -intensity * 0.04, 1); head.rotation.y = blend(io.head.y, side * intensity * 0.08, 1); }
      break;

    case 'brushShoulder':
      // Quick brush of shoulder, like dusting off
      {
        const brush = Math.sin(progress * Math.PI * 2.5) * intensity;
        if (side < 0) {
          if (rUA) { rUA.rotation.x = blend(REST.rightUpperArm.x + io.rUA.x, -brush * 0.30, 1); rUA.rotation.z = blend(REST.rightUpperArm.z + io.rUA.z, brush * 0.20, 1); }
          if (rLA) { rLA.rotation.x = blend(REST.rightLowerArm.x + io.rLA.x, -brush * 0.45, 1); }
          if (rH) { rH.rotation.z = blend(REST.rightHand.z + io.rH.z, -brush * 0.30, 1); }
        } else {
          if (lUA) { lUA.rotation.x = blend(REST.leftUpperArm.x + io.lUA.x, -brush * 0.30, 1); lUA.rotation.z = blend(REST.leftUpperArm.z + io.lUA.z, -brush * 0.20, 1); }
          if (lLA) { lLA.rotation.x = blend(REST.leftLowerArm.x + io.lLA.x, -brush * 0.45, 1); }
          if (lH) { lH.rotation.z = blend(REST.leftHand.z + io.lH.z, brush * 0.30, 1); }
        }
        if (head) head.rotation.y = blend(io.head.y, -side * brush * 0.10, 1);
      }
      break;

    case 'leanIn':
      {
        const lean = Math.sin(progress * Math.PI) * 0.055;
        if (spine) spine.rotation.x = blend(io.spine.x, -lean, 1);
        if (chest) chest.rotation.x = blend(io.chest.x, -lean * 0.72, 1);
        if (head) {
          head.rotation.x = blend(io.head.x, lean * 0.42, 1);
          head.rotation.y = blend(io.head.y, side * lean * 0.55, 1);
        }
        if (neck) neck.rotation.x = blend(io.neck.x, lean * 0.25, 1);
      }
      break;

    case 'openPalm':
      {
        const talk = Math.sin(progress * Math.PI * 2) * intensity;
        if (side < 0) {
          if (lUA) { lUA.rotation.x = blend(REST.leftUpperArm.x + io.lUA.x, -talk * 0.30, 1); lUA.rotation.z = blend(REST.leftUpperArm.z + io.lUA.z, -talk * 0.42, 1); }
          if (lLA) lLA.rotation.x = blend(REST.leftLowerArm.x + io.lLA.x, -talk * 0.48, 1);
          if (lH) { lH.rotation.x = blend(REST.leftHand.x + io.lH.x, talk * 0.18, 1); lH.rotation.y = blend(REST.leftHand.y + io.lH.y, talk * 0.30, 1); }
        } else {
          if (rUA) { rUA.rotation.x = blend(REST.rightUpperArm.x + io.rUA.x, -talk * 0.30, 1); rUA.rotation.z = blend(REST.rightUpperArm.z + io.rUA.z, talk * 0.42, 1); }
          if (rLA) rLA.rotation.x = blend(REST.rightLowerArm.x + io.rLA.x, -talk * 0.48, 1);
          if (rH) { rH.rotation.x = blend(REST.rightHand.x + io.rH.x, talk * 0.18, 1); rH.rotation.y = blend(REST.rightHand.y + io.rH.y, -talk * 0.30, 1); }
        }
        applyFingerCurl(side, Math.abs(talk) * 0.7);
      }
      break;

    case 'speakingNod':
      {
        const nod = Math.sin(progress * Math.PI * 2.4) * intensity * 0.075;
        if (head) { head.rotation.x = blend(io.head.x, nod, 1); head.rotation.y = blend(io.head.y, side * intensity * 0.035, 1); }
        if (neck) neck.rotation.x = blend(io.neck.x, nod * 0.42, 1);
        if (spine) spine.rotation.x = blend(io.spine.x, nod * 0.16, 1);
      }
      break;

    case 'stretchArm':
      // One arm stretches out and up, then back down
      {
        const stretch = Math.sin(progress * Math.PI) * intensity;
        if (side < 0) {
          if (lUA) { lUA.rotation.x = blend(REST.leftUpperArm.x + io.lUA.x, -stretch * 0.80, 1); lUA.rotation.z = blend(REST.leftUpperArm.z + io.lUA.z, -stretch * 0.30, 1); }
          if (lLA) { lLA.rotation.x = blend(REST.leftLowerArm.x + io.lLA.x, -stretch * 0.40, 1); }
          if (lH) { lH.rotation.x = blend(REST.leftHand.x + io.lH.x, stretch * 0.20, 1); }
        } else {
          if (rUA) { rUA.rotation.x = blend(REST.rightUpperArm.x + io.rUA.x, -stretch * 0.80, 1); rUA.rotation.z = blend(REST.rightUpperArm.z + io.rUA.z, stretch * 0.30, 1); }
          if (rLA) { rLA.rotation.x = blend(REST.rightLowerArm.x + io.rLA.x, -stretch * 0.40, 1); }
          if (rH) { rH.rotation.x = blend(REST.rightHand.x + io.rH.x, stretch * 0.20, 1); }
        }
        if (head) head.rotation.y = blend(io.head.y, side * stretch * 0.12, 1);
        if (chest) chest.rotation.x = blend(io.chest.x, stretch * 0.03, 1);
      }
      break;

    // ── new natural idle gestures ──────────────────────────────────────────

    case 'earTuck':
      // Hand reaches up to temple and sweeps hair back behind ear
      {
        const sweep = holdCurve(progress, 0.32, 0.28);
        const strokePhase = Math.sin(progress * Math.PI * 1.5) * intensity;
        if (side < 0) {
          if (lUA) { lUA.rotation.x = blend(REST.leftUpperArm.x + io.lUA.x, -sweep * 0.52, 1); lUA.rotation.z = blend(REST.leftUpperArm.z + io.lUA.z, -sweep * 0.78, 1); }
          if (lLA) { lLA.rotation.x = blend(REST.leftLowerArm.x + io.lLA.x, -sweep * 0.92, 1); lLA.rotation.z = blend(REST.leftLowerArm.z + io.lLA.z, -sweep * 0.18, 1); }
          if (lH) { lH.rotation.x = blend(REST.leftHand.x + io.lH.x, -sweep * 0.15 + strokePhase * 0.08, 1); lH.rotation.y = blend(REST.leftHand.y + io.lH.y, sweep * 0.32, 1); }
        } else {
          if (rUA) { rUA.rotation.x = blend(REST.rightUpperArm.x + io.rUA.x, -sweep * 0.52, 1); rUA.rotation.z = blend(REST.rightUpperArm.z + io.rUA.z, sweep * 0.78, 1); }
          if (rLA) { rLA.rotation.x = blend(REST.rightLowerArm.x + io.rLA.x, -sweep * 0.92, 1); rLA.rotation.z = blend(REST.rightLowerArm.z + io.rLA.z, sweep * 0.18, 1); }
          if (rH) { rH.rotation.x = blend(REST.rightHand.x + io.rH.x, -sweep * 0.15 + strokePhase * 0.08, 1); rH.rotation.y = blend(REST.rightHand.y + io.rH.y, -sweep * 0.32, 1); }
        }
        if (head) { head.rotation.z = blend(io.head.z, -side * sweep * 0.08, 1); head.rotation.y = blend(io.head.y, side * sweep * 0.06, 1); }
        if (neck) neck.rotation.z = blend(io.neck.z, -side * sweep * 0.04, 1);
      }
      break;

    case 'gentleStretch':
      // Both arms open subtly, chest-opening micro-stretch
      {
        const openPhase = Math.sin(progress * Math.PI) * intensity;
        if (lUA) { lUA.rotation.z = blend(REST.leftUpperArm.z + io.lUA.z, -openPhase * 0.18, 1); lUA.rotation.x = blend(REST.leftUpperArm.x + io.lUA.x, -openPhase * 0.10, 1); }
        if (rUA) { rUA.rotation.z = blend(REST.rightUpperArm.z + io.rUA.z, openPhase * 0.18, 1); rUA.rotation.x = blend(REST.rightUpperArm.x + io.rUA.x, -openPhase * 0.10, 1); }
        if (lLA) lLA.rotation.x = blend(REST.leftLowerArm.x + io.lLA.x, -openPhase * 0.08, 1);
        if (rLA) rLA.rotation.x = blend(REST.rightLowerArm.x + io.rLA.x, -openPhase * 0.08, 1);
        if (chest) chest.rotation.x = blend(io.chest.x, -openPhase * 0.04, 1);
        if (spine) spine.rotation.x = blend(io.spine.x, -openPhase * 0.02, 1);
        if (head) { head.rotation.x = blend(io.head.x, -openPhase * 0.05, 1); head.rotation.z = blend(io.head.z, side * openPhase * 0.02, 1); }
      }
      break;

    case 'handsClasp':
      // Both hands come together in front, brief clasp
      {
        const clasp = holdCurve(progress, 0.30, 0.32) * intensity;
        if (lUA) { lUA.rotation.x = blend(REST.leftUpperArm.x + io.lUA.x, -clasp * 0.22, 1); lUA.rotation.z = blend(REST.leftUpperArm.z + io.lUA.z, -clasp * 0.38, 1); }
        if (rUA) { rUA.rotation.x = blend(REST.rightUpperArm.x + io.rUA.x, -clasp * 0.22, 1); rUA.rotation.z = blend(REST.rightUpperArm.z + io.rUA.z, clasp * 0.38, 1); }
        if (lLA) { lLA.rotation.x = blend(REST.leftLowerArm.x + io.lLA.x, -clasp * 0.52, 1); lLA.rotation.z = blend(REST.leftLowerArm.z + io.lLA.z, -clasp * 0.22, 1); }
        if (rLA) { rLA.rotation.x = blend(REST.rightLowerArm.x + io.rLA.x, -clasp * 0.52, 1); rLA.rotation.z = blend(REST.rightLowerArm.z + io.rLA.z, clasp * 0.22, 1); }
        if (lH) { lH.rotation.y = blend(REST.leftHand.y + io.lH.y, clasp * 0.20, 1); lH.rotation.z = blend(REST.leftHand.z + io.lH.z, -clasp * 0.10, 1); }
        if (rH) { rH.rotation.y = blend(REST.rightHand.y + io.rH.y, -clasp * 0.20, 1); rH.rotation.z = blend(REST.rightHand.z + io.rH.z, clasp * 0.10, 1); }
        if (head) { head.rotation.x = blend(io.head.x, clasp * 0.03, 1); }
      }
      break;

    case 'thoughtfulLook':
      // Slow gaze upward with dreamy head tilt
      {
        const lookUp = holdCurve(progress, 0.35, 0.30) * intensity;
        const drift = Math.sin(progress * Math.PI * 0.8) * lookUp;
        if (head) {
          head.rotation.x = blend(io.head.x, -drift * 0.12, 1);
          head.rotation.y = blend(io.head.y, side * drift * 0.15, 1);
          head.rotation.z = blend(io.head.z, side * drift * 0.06, 1);
        }
        if (neck) {
          neck.rotation.x = blend(io.neck.x, -drift * 0.06, 1);
          neck.rotation.y = blend(io.neck.y, side * drift * 0.08, 1);
        }
        if (spine) spine.rotation.x = blend(io.spine.x, -drift * 0.02, 1);
      }
      break;

    // ── new speaking gestures ─────────────────────────────────────────────

    case 'emphasizePoint':
      // One hand rises to emphasize a conversational point
      {
        const emph = Math.sin(progress * Math.PI * 1.5) * intensity;
        if (side < 0) {
          if (lUA) { lUA.rotation.x = blend(REST.leftUpperArm.x + io.lUA.x, -emph * 0.48, 1); lUA.rotation.z = blend(REST.leftUpperArm.z + io.lUA.z, -emph * 0.35, 1); }
          if (lLA) { lLA.rotation.x = blend(REST.leftLowerArm.x + io.lLA.x, -emph * 0.62, 1); }
          if (lH) { lH.rotation.x = blend(REST.leftHand.x + io.lH.x, emph * 0.15, 1); lH.rotation.y = blend(REST.leftHand.y + io.lH.y, emph * 0.22, 1); }
        } else {
          if (rUA) { rUA.rotation.x = blend(REST.rightUpperArm.x + io.rUA.x, -emph * 0.48, 1); rUA.rotation.z = blend(REST.rightUpperArm.z + io.rUA.z, emph * 0.35, 1); }
          if (rLA) { rLA.rotation.x = blend(REST.rightLowerArm.x + io.rLA.x, -emph * 0.62, 1); }
          if (rH) { rH.rotation.x = blend(REST.rightHand.x + io.rH.x, emph * 0.15, 1); rH.rotation.y = blend(REST.rightHand.y + io.rH.y, -emph * 0.22, 1); }
        }
        if (head) { head.rotation.x = blend(io.head.x, -emph * 0.03, 1); head.rotation.y = blend(io.head.y, side * emph * 0.06, 1); }
        applyFingerCurl(side, Math.abs(emph) * 0.5);
      }
      break;

    case 'bothHandsExplain':
      // Both hands open outward in explanatory gesture
      {
        const explain = Math.sin(progress * Math.PI * 1.8) * intensity;
        if (lUA) { lUA.rotation.x = blend(REST.leftUpperArm.x + io.lUA.x, -explain * 0.28, 1); lUA.rotation.z = blend(REST.leftUpperArm.z + io.lUA.z, -explain * 0.32, 1); }
        if (rUA) { rUA.rotation.x = blend(REST.rightUpperArm.x + io.rUA.x, -explain * 0.28, 1); rUA.rotation.z = blend(REST.rightUpperArm.z + io.rUA.z, explain * 0.32, 1); }
        if (lLA) { lLA.rotation.x = blend(REST.leftLowerArm.x + io.lLA.x, -explain * 0.42, 1); lLA.rotation.z = blend(REST.leftLowerArm.z + io.lLA.z, -explain * 0.10, 1); }
        if (rLA) { rLA.rotation.x = blend(REST.rightLowerArm.x + io.rLA.x, -explain * 0.42, 1); rLA.rotation.z = blend(REST.rightLowerArm.z + io.rLA.z, explain * 0.10, 1); }
        if (lH) { lH.rotation.x = blend(REST.leftHand.x + io.lH.x, explain * 0.18, 1); lH.rotation.y = blend(REST.leftHand.y + io.lH.y, explain * 0.25, 1); }
        if (rH) { rH.rotation.x = blend(REST.rightHand.x + io.rH.x, explain * 0.18, 1); rH.rotation.y = blend(REST.rightHand.y + io.rH.y, -explain * 0.25, 1); }
        if (head) { head.rotation.x = blend(io.head.x, -Math.abs(explain) * 0.02, 1); head.rotation.y = blend(io.head.y, side * Math.abs(explain) * 0.04, 1); }
        if (spine) spine.rotation.x = blend(io.spine.x, -Math.abs(explain) * 0.015, 1);
        applyFingerCurl(-1, Math.abs(explain) * 0.4);
        applyFingerCurl(1, Math.abs(explain) * 0.4);
      }
      break;
  }

  if (progress >= 1) {
    startBlendOut(); // snapshot pose for smooth return to idle
    gestureState = 'none';
  }
}

function applyBlink(dt) {
  if (!vrm?.expressionManager) return;
  const em = vrm.expressionManager;
  if (blinkPhase === 'wait') {
    blinkTimer -= dt;
    if (blinkTimer <= 0) { blinkPhase = 'closing'; blinkT = 0; }
  } else if (blinkPhase === 'closing') {
    blinkT += dt;
    const w = Math.min(blinkT / BLINK_CLOSE_DUR, 1.0);
    try { em.setValue('blink', w); } catch (_) { }
    if (blinkT >= BLINK_CLOSE_DUR) { blinkPhase = 'opening'; blinkT = 0; }
  } else if (blinkPhase === 'opening') {
    blinkT += dt;
    const w = 1.0 - Math.min(blinkT / BLINK_OPEN_DUR, 1.0);
    try { em.setValue('blink', w); } catch (_) { }
    if (blinkT >= BLINK_OPEN_DUR) {
      blinkPhase = 'wait'; blinkTimer = nextBlinkWait();
      try { em.setValue('blink', 0); } catch (_) { }
    }
  }
}

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);
  controls.update();
  if (vrm) {
    vrm.update(dt);
    const em = vrm.expressionManager;
    if (em) {
      for (const [name, target] of Object.entries(exprTargets)) {
        const cur = exprCurrent[name] ?? 0;
        const next = cur + (target - cur) * Math.min(1, EXPR_LERP * dt);
        exprCurrent[name] = next;
        if (name !== 'blink') { try { em.setValue(name, next); } catch (_) { } }
      }
    }
    // 1. Apply base idle pose (breathing, sway, head look)
    applyIdle(dt);
    resetFingerCurl();
    // 2. Layer gesture or thinking on top of idle
    if (!applyThinkingPose(dt)) applyGestures(dt);
    // 3. Blend-out: smoothly interpolate from snapshot → current idle
    //    This eliminates jerky snaps when gestures/thinking end
    applyBlendOut(dt);
    applyBlink(dt);
  }
  renderer.render(scene, camera);
}
animate();

// ── VRM load ─────────────────────────────────────────────────────────────────
const loader = new GLTFLoader();
loader.register(parser => new VRMLoaderPlugin(parser));

const VRM_URL = './assets/Aiko_Yukata.vrm';

if (window.location.protocol === 'file:') {
  loadMsg.textContent = 'error: open Aiko via `python main.py` or http://localhost:8787/ — browsers block VRM fetches from file://';
  throw new Error('Aiko WebUI must be served over HTTP so assets/Aiko.vrm can be fetched.');
}

loadMsg.textContent = 'loading Aiko.vrm…';
fill.style.width = '5%';

loader.load(VRM_URL,
  (gltf) => {
    fill.style.width = '95%';
    loadMsg.textContent = 'building model…';
    vrm = gltf.userData.vrm;
    window._vrm = vrm;
    window._REST = REST;

    VRMUtils.removeUnnecessaryVertices(vrm.scene);
    vrm.scene.traverse(o => { if (o.frustumCulled) o.frustumCulled = false; });
    scene.add(vrm.scene);
    vrm.scene.rotation.y = Math.PI;

    if (vrm.expressionManager) {
      vrm.expressionManager.expressions.forEach(ex => {
        exprTargets[ex.expressionName] = 0;
        exprCurrent[ex.expressionName] = 0;
      });
    }

    fill.style.width = '100%';
    setTimeout(() => {
      const ov = document.getElementById('load-overlay');
      ov.classList.add('fade');
      setTimeout(() => ov.style.display = 'none', 800);
    }, 300);
  },
  (prog) => {
    const p = prog.total ? prog.loaded / prog.total : 0;
    fill.style.width = (5 + p * 88) + '%';
  },
  (err) => {
    const detail = err?.message || String(err);
    loadMsg.textContent = `error loading ${VRM_URL}: ${detail}. Start with python main.py and open http://localhost:8787/ instead of file://.`;
    console.error('[aiko-vrm]', err);
  }
);

// ── expression / viseme API (called by WS handler) ───────────────────────────
const VISEME = { A: 'aa', I: 'ih', U: 'ou', E: 'ee', O: 'oh' };
let activeViseme = 'aa';
let mouthOpen = 0;

function applyMouthShape(weight = mouthOpen) {
  const w = Math.max(0, Math.min(1, weight));
  ['aa', 'ih', 'ou', 'ee', 'oh'].forEach(k => exprTargets[k] = 0);
  exprTargets[activeViseme] = w;
  mouthOpen = w;
  if (w > 0.03) lastVisemeAt = performance.now();
}

const VRM_EMOJI_EXPRESSIONS = {
  '😊': 'happy', '😄': 'happy', '😁': 'happy', '😆': 'happy', '🥰': 'happy', '😍': 'happy', '🙂': 'happy', '😋': 'happy', '🌸': 'happy', '✨': 'happy', '❤️': 'happy', '💖': 'happy', '☺️': 'happy',
  '😒': 'angry', '😡': 'angry', '😠': 'angry', '😤': 'angry', '🤬': 'angry', '💢': 'angry',
  '😭': 'sorrow', '😢': 'sorrow', '🥺': 'sorrow', '☹️': 'sorrow', '🙁': 'sorrow', '😔': 'sorrow', '😞': 'sorrow', '💧': 'sorrow',
  '😮': 'surprised', '😯': 'surprised', '😲': 'surprised', '😳': 'surprised', '🤯': 'surprised', '😱': 'surprised', '⁉️': 'surprised', '❓': 'surprised',
  '😜': 'fun', '🤪': 'fun', '😏': 'fun', '😈': 'fun', '🙃': 'fun', '😉': 'fun',
  '😐': 'neutral', '😑': 'neutral', '😶': 'neutral', '🤖': 'neutral', '😴': 'neutral', '🤔': 'neutral', '💭': 'neutral'
};

const VRM_EXPR_ALIASES = {
  'sad': 'sorrow',
  'relaxed': 'fun',
  'joy': 'happy',
  'annoyed': 'angry',
  'shy': 'happy',
  'thinking': 'neutral'
};

window.aikoSetExpression = (name, intensity = 1.0) => {
  const mapped = VRM_EMOJI_EXPRESSIONS[name] || VRM_EXPR_ALIASES[name] || name;
  for (const k of Object.keys(exprTargets)) if (k !== 'blink') exprTargets[k] = 0;
  if (mapped && mapped !== 'neutral') {
    exprTargets[mapped] = intensity;
    if (mapped === 'sorrow') exprTargets['sad'] = intensity;
    if (mapped === 'fun') exprTargets['relaxed'] = intensity;
  }

  const el = document.getElementById('vrm-emotion');
  if (el) {
    el.textContent = name ? `${name} · ${Math.round(intensity * 100)}%` : '—';
    el.className = (name && name !== 'neutral') ? 'active' : '';
  }

  clearTimeout(exprResetTimer);
  if (mapped && mapped !== 'neutral') {
    exprResetTimer = setTimeout(() => window.aikoSetExpression('neutral'), EXPR_RESET_DELAY);
  }
};

window.aikoSetViseme = (viseme, weight = 1.0) => {
  activeViseme = VISEME[viseme] ?? viseme;
  applyMouthShape(mouthOpen);
  const dot = document.getElementById('speak-dot');
  const lbl = document.getElementById('speak-label');
  const speaking = mouthOpen > 0.03;
  dot.className = speaking ? 'dot speak' : 'dot';
  lbl.textContent = speaking ? activeViseme : 'idle';
};

window.aikoSetMouthOpen = (weight = 0) => {
  applyMouthShape(weight);
  const dot = document.getElementById('speak-dot');
  const lbl = document.getElementById('speak-label');
  if (mouthOpen > 0.03) {
    dot.className = 'dot speak';
    lbl.textContent = activeViseme;
  } else {
    dot.className = 'dot';
    lbl.textContent = 'idle';
  }
};

window.aikoSetPose = (name, active = true) => {
  if (name !== 'thinking') return;
  const shouldActivate = Boolean(active);
  if (shouldActivate && !thinkingPoseActive) {
    // Snapshot current pose for smooth blend into thinking
    if (gestureState !== 'none') startBlendOut();
    pickThinkingPose();
  }
  thinkingPoseActive = shouldActivate;
  if (!shouldActivate) {
    thinkingPoseT = 0;
    if (gestureState !== 'none') startBlendOut();
    gestureState = 'none';
    gestureCooldown = 1.2 + Math.random() * 2.0;
  }
};
