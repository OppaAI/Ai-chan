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
const camera = new THREE.PerspectiveCamera(22, 1, 0.1, 100);
camera.position.set(0.00, 1.33, 5.0);
//camera.fov = 12;
//camera.updateProjectionMatrix();

const controls = new OrbitControls(camera, canvas);
controls.target.set(0.00, 1.30, 0);
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
  leftUpperArm: { x: -0.02, y: 0.0, z: 1.28 },
  rightUpperArm: { x: -0.02, y: 0.0, z: -1.28 },
  leftLowerArm: { x: 0.12, y: 0.0, z: 0.08 },
  rightLowerArm: { x: 0.12, y: 0.0, z: -0.08 },
  leftHand: { x: 0.0, y: 0.08, z: 0.0 },
  rightHand: { x: 0.0, y: -0.08, z: 0.0 },
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

const GESTURES = [
  'lookAround',
  'lookAtHand',
  'hairBrush',
  'fingerPlay',
  'meetGaze',
  'curiousTilt',
  'shiftWeight',
  'stretchNeck',
  'raiseHand',    // new: hand rises up near face, thinking pose
  'chinTouch',    // new: hand up to chin, curious
  'shoulderRoll', // new: one shoulder rolls up and back
  'sway',         // new: whole-body gentle sway side to side
  'headNod',      // new: slow contemplative nod
  'wristFlick',   // new: wrist flick outward as if brushing something off
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
  const gesture = GESTURES[Math.floor(Math.random() * GESTURES.length)];
  const side = Math.random() < 0.5 ? -1 : 1;
  gestureState = gesture;
  gestureT = 0;
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

function applyFingerCurl(side, intensity) {
  const prefix = side < 0 ? 'left' : 'right';
  const pulse = 0.5 + Math.sin(t * 8.0) * 0.5;
  for (const name of FINGER_BONES) {
    if (!name.startsWith(prefix)) continue;
    const bone = getBone(name);
    if (bone) bone.rotation.z += intensity * (0.12 + pulse * 0.14);
  }
}

function applyGestures(dt) {
  if (!vrm?.humanoid) return;

  if (gestureState === 'none') {
    if (speakingRecently()) return;
    gestureCooldown -= dt;
    if (gestureCooldown <= 0) {
      pickGesture();
      gestureCooldown = 4.5 + Math.random() * 7.0;
    }
    return;
  }

  gestureT += dt;
  const progress = Math.min(1, gestureT / gestureDuration);
  const eased = easeInOutSine(progress);
  const held = holdCurve(progress);
  const intensity = Math.sin(progress * Math.PI);
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

  switch (gestureState) {

    // ── existing gestures, now more exaggerated ──────────────────────────────

    case 'lookAround':
      // Turns head noticeably to one side, slight chin dip
      if (head) {
        head.rotation.y += gestureTarget.look * 1.2 * held;
        head.rotation.x += Math.sin(eased * Math.PI) * 0.04;
      }
      if (neck) {
        neck.rotation.y += gestureTarget.look * 0.45 * held;
        neck.rotation.z += gestureTarget.tilt * 0.3 * held;
      }
      if (spine) spine.rotation.y += gestureTarget.look * 0.08 * held;
      break;

    case 'lookAtHand':
      // Arm rises noticeably; she inspects her hand, head follows
      if (side < 0) {
        if (lUA) { lUA.rotation.x = REST.leftUpperArm.x - intensity * 0.55; lUA.rotation.z = REST.leftUpperArm.z - intensity * 0.55; }
        if (lLA) { lLA.rotation.x = REST.leftLowerArm.x - intensity * 0.72; lLA.rotation.z = REST.leftLowerArm.z - intensity * 0.22; }
        if (lH) { lH.rotation.x = REST.leftHand.x + intensity * 0.22; lH.rotation.y = REST.leftHand.y + intensity * 0.30; }
      } else {
        if (rUA) { rUA.rotation.x = REST.rightUpperArm.x - intensity * 0.55; rUA.rotation.z = REST.rightUpperArm.z + intensity * 0.55; }
        if (rLA) { rLA.rotation.x = REST.rightLowerArm.x - intensity * 0.72; rLA.rotation.z = REST.rightLowerArm.z + intensity * 0.22; }
        if (rH) { rH.rotation.x = REST.rightHand.x + intensity * 0.22; rH.rotation.y = REST.rightHand.y - intensity * 0.30; }
      }
      if (head) { head.rotation.y += side * intensity * 0.28; head.rotation.x += intensity * 0.12; }
      if (neck) neck.rotation.y += side * intensity * 0.14;
      break;

    case 'hairBrush':
      // Arm sweeps up past shoulder, hand moves through hair area
      if (side < 0) {
        if (lUA) { lUA.rotation.z = REST.leftUpperArm.z - intensity * 0.90; lUA.rotation.x = REST.leftUpperArm.x - intensity * 0.45; }
        if (lLA) { lLA.rotation.x = REST.leftLowerArm.x - intensity * 0.80; lLA.rotation.z = REST.leftLowerArm.z - intensity * 0.32; }
        if (lH) { lH.rotation.y = REST.leftHand.y + intensity * 0.38; lH.rotation.z = REST.leftHand.z - intensity * 0.22; }
      } else {
        if (rUA) { rUA.rotation.z = REST.rightUpperArm.z + intensity * 0.90; rUA.rotation.x = REST.rightUpperArm.x - intensity * 0.45; }
        if (rLA) { rLA.rotation.x = REST.rightLowerArm.x - intensity * 0.80; rLA.rotation.z = REST.rightLowerArm.z + intensity * 0.32; }
        if (rH) { rH.rotation.y = REST.rightHand.y - intensity * 0.38; rH.rotation.z = REST.rightHand.z + intensity * 0.22; }
      }
      if (head) { head.rotation.z -= side * intensity * 0.07; head.rotation.y += side * intensity * 0.05; }
      break;

    case 'fingerPlay':
      // Forearm lifts, fingers curl and uncurl rhythmically
      if (head) { head.rotation.x += intensity * 0.08; head.rotation.y += side * intensity * 0.14; }
      if (side < 0) {
        if (lLA) { lLA.rotation.x = REST.leftLowerArm.x - intensity * 0.38; }
        if (lH) { lH.rotation.y = REST.leftHand.y + intensity * 0.26; lH.rotation.z = REST.leftHand.z + Math.sin(gestureT * 5.0) * 0.10 * intensity; }
      } else {
        if (rLA) { rLA.rotation.x = REST.rightLowerArm.x - intensity * 0.38; }
        if (rH) { rH.rotation.y = REST.rightHand.y - intensity * 0.26; rH.rotation.z = REST.rightHand.z - Math.sin(gestureT * 5.0) * 0.10 * intensity; }
      }
      applyFingerCurl(side, intensity);
      break;

    case 'meetGaze':
      // Head settles to neutral/forward, like making eye contact
      if (head) {
        head.rotation.y *= (1 - held * 0.85);
        head.rotation.z *= (1 - held * 0.80);
        head.rotation.x = head.rotation.x * (1 - held * 0.5) + held * 0.025;
      }
      if (neck) {
        neck.rotation.y *= (1 - held * 0.60);
        neck.rotation.z *= (1 - held * 0.60);
      }
      break;

    case 'curiousTilt':
      // Head tilts well to one side, ears toward shoulder, brows up implied
      if (head) {
        head.rotation.z += gestureTarget.tilt * 1.3 * held;
        head.rotation.x -= 0.025 * intensity;
        head.rotation.y += side * 0.06 * held;
      }
      if (neck) {
        neck.rotation.z += gestureTarget.tilt * 0.60 * held;
        neck.rotation.y += side * 0.04 * held;
      }
      break;

    case 'shiftWeight':
      // Hips slide to one side, spine and head compensate
      if (hips) {
        hips.position.x += side * Math.sin(eased * Math.PI) * 0.022;
        hips.rotation.z += side * Math.sin(eased * Math.PI) * 0.025;
      }
      if (spine) spine.rotation.z -= side * Math.sin(eased * Math.PI) * 0.020;
      if (chest) chest.rotation.z -= side * Math.sin(eased * Math.PI) * 0.014;
      if (head) head.rotation.z += side * Math.sin(eased * Math.PI) * 0.018;
      break;

    case 'stretchNeck':
      // Head tilts back, chin up, neck extends — satisfying stretch
      if (neck) { neck.rotation.x -= intensity * 0.10; neck.rotation.z += gestureTarget.tilt * 0.5 * held; }
      if (head) { head.rotation.x -= intensity * 0.08; head.rotation.z += gestureTarget.tilt * 0.3 * held; }
      if (chest) chest.rotation.x -= intensity * 0.04;
      break;

    // ── new gestures ─────────────────────────────────────────────────────────

    case 'raiseHand':
      // One hand rises to about chin/cheek height — thoughtful or greeting pose
      if (side < 0) {
        if (lUA) { lUA.rotation.x = REST.leftUpperArm.x - intensity * 0.70; lUA.rotation.z = REST.leftUpperArm.z - intensity * 0.40; }
        if (lLA) { lLA.rotation.x = REST.leftLowerArm.x - intensity * 0.90; lLA.rotation.z = REST.leftLowerArm.z - intensity * 0.14; }
        if (lH) { lH.rotation.x = REST.leftHand.x + intensity * 0.12; }
      } else {
        if (rUA) { rUA.rotation.x = REST.rightUpperArm.x - intensity * 0.70; rUA.rotation.z = REST.rightUpperArm.z + intensity * 0.40; }
        if (rLA) { rLA.rotation.x = REST.rightLowerArm.x - intensity * 0.90; rLA.rotation.z = REST.rightLowerArm.z + intensity * 0.14; }
        if (rH) { rH.rotation.x = REST.rightHand.x + intensity * 0.12; }
      }
      // Head turns slightly toward the raised hand
      if (head) { head.rotation.y += side * intensity * 0.10; head.rotation.x -= intensity * 0.03; }
      break;

    case 'chinTouch':
      // Hand comes up to rest near chin — thinking/considering pose
      if (side < 0) {
        if (lUA) { lUA.rotation.x = REST.leftUpperArm.x - intensity * 0.60; lUA.rotation.z = REST.leftUpperArm.z - intensity * 0.60; }
        if (lLA) { lLA.rotation.x = REST.leftLowerArm.x - intensity * 1.05; lLA.rotation.z = REST.leftLowerArm.z - intensity * 0.10; }
        if (lH) { lH.rotation.x = REST.leftHand.x - intensity * 0.18; lH.rotation.y = REST.leftHand.y + intensity * 0.20; }
      } else {
        if (rUA) { rUA.rotation.x = REST.rightUpperArm.x - intensity * 0.60; rUA.rotation.z = REST.rightUpperArm.z + intensity * 0.60; }
        if (rLA) { rLA.rotation.x = REST.rightLowerArm.x - intensity * 1.05; rLA.rotation.z = REST.rightLowerArm.z + intensity * 0.10; }
        if (rH) { rH.rotation.x = REST.rightHand.x - intensity * 0.18; rH.rotation.y = REST.rightHand.y - intensity * 0.20; }
      }
      // Slight head tilt and dip, as if actually resting chin on hand
      if (head) { head.rotation.x += intensity * 0.06; head.rotation.z += side * intensity * 0.05; }
      if (neck) neck.rotation.x += intensity * 0.04;
      break;

    case 'shoulderRoll':
      // One shoulder rolls up then drops — casual, relaxed movement
      if (side < 0) {
        if (lUA) {
          // Up phase (first half): shoulder rises; Down phase: drops back
          const roll = Math.sin(progress * Math.PI * 1.5) * intensity;
          lUA.rotation.x = REST.leftUpperArm.x - roll * 0.28;
          lUA.rotation.z = REST.leftUpperArm.z + roll * 0.18;
          lUA.rotation.y = REST.leftUpperArm.y + roll * 0.10;
        }
      } else {
        if (rUA) {
          const roll = Math.sin(progress * Math.PI * 1.5) * intensity;
          rUA.rotation.x = REST.rightUpperArm.x - roll * 0.28;
          rUA.rotation.z = REST.rightUpperArm.z - roll * 0.18;
          rUA.rotation.y = REST.rightUpperArm.y - roll * 0.10;
        }
      }
      // Head dips slightly to opposite side as shoulder rises
      if (head) head.rotation.z -= side * Math.sin(progress * Math.PI * 1.5) * intensity * 0.06;
      if (neck) neck.rotation.z -= side * Math.sin(progress * Math.PI * 1.5) * intensity * 0.04;
      break;

    case 'sway':
      // Whole body sways gently side to side — one full cycle
      {
        const swayVal = Math.sin(progress * Math.PI * 2) * intensity;
        if (hips) { hips.position.x += swayVal * 0.020; hips.rotation.z += swayVal * 0.022; }
        if (spine) spine.rotation.z += swayVal * -0.014;
        if (chest) chest.rotation.z += swayVal * -0.010;
        if (head) head.rotation.z += swayVal * -0.016;
        if (neck) neck.rotation.z += swayVal * -0.010;
        // Arms drift opposite to sway for counterbalance
        if (lUA) lUA.rotation.z = REST.leftUpperArm.z - swayVal * 0.06;
        if (rUA) rUA.rotation.z = REST.rightUpperArm.z - swayVal * 0.06;
      }
      break;

    case 'headNod':
      // Slow double-nod — agreeable, contemplative
      {
        const nod = Math.sin(progress * Math.PI * 3.5) * intensity * 0.14;
        if (head) { head.rotation.x += nod; }
        if (neck) { neck.rotation.x += nod * 0.5; }
        // Slight eye-contact lean on downstroke
        if (head) head.rotation.y *= (1 - held * 0.4);
      }
      break;

    case 'wristFlick':
      // Quick outward wrist rotation — dismissive or playful flick
      if (side < 0) {
        if (lLA) { lLA.rotation.y = REST.leftLowerArm.y + intensity * 0.55; }
        if (lH) { lH.rotation.z = REST.leftHand.z + intensity * 0.50; lH.rotation.y = REST.leftHand.y + intensity * 0.30; }
      } else {
        if (rLA) { rLA.rotation.y = REST.rightLowerArm.y - intensity * 0.55; }
        if (rH) { rH.rotation.z = REST.rightHand.z - intensity * 0.50; rH.rotation.y = REST.rightHand.y - intensity * 0.30; }
      }
      // Tiny head glance toward the flick
      if (head) head.rotation.y += side * intensity * 0.08;
      break;
  }

  if (progress >= 1) gestureState = 'none';
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

function applyIdle(dt) {
  if (!vrm?.humanoid) return;
  t += dt;
  const h = vrm.humanoid;
  const get = n => h.getRawBoneNode(n);

  const breath = Math.sin(t * 0.83) * 0.013;
  const chest = get('chest');
  const spine = get('spine');
  if (chest) chest.rotation.x = breath;
  if (spine) spine.rotation.x = breath * 0.5;

  const hips = get('hips');
  if (hips) {
    hips.rotation.z = Math.sin(t * 0.41) * 0.012;
    hips.rotation.x = Math.sin(t * 0.67) * 0.008;
    hips.position.x = Math.sin(t * 0.41) * 0.003;
  }

  const head = get('head');
  if (head) {
    head.rotation.y = Math.sin(t * 0.31) * 0.055 + Math.sin(t * 1.13) * 0.012;
    head.rotation.z = Math.sin(t * 0.27 + 1.1) * 0.018 + Math.sin(t * 0.71) * 0.006;
    head.rotation.x = Math.sin(t * 0.53) * 0.012;
  }
  const neck = get('neck');
  if (neck && head) {
    neck.rotation.y = head.rotation.y * 0.3;
    neck.rotation.z = head.rotation.z * 0.3;
  }

  const lUA = get('leftUpperArm');
  const rUA = get('rightUpperArm');
  const lLA = get('leftLowerArm');
  const rLA = get('rightLowerArm');
  const lH = get('leftHand');
  const rH = get('rightHand');
  if (lUA) { lUA.rotation.x = REST.leftUpperArm.x + Math.sin(t * .47) * .010; lUA.rotation.y = REST.leftUpperArm.y + Math.sin(t * .33) * .006; lUA.rotation.z = REST.leftUpperArm.z + Math.sin(t * .41) * .008; }
  if (rUA) { rUA.rotation.x = REST.rightUpperArm.x + Math.sin(t * .53 + .9) * .010; rUA.rotation.y = REST.rightUpperArm.y + Math.sin(t * .35 + .4) * .006; rUA.rotation.z = REST.rightUpperArm.z + Math.sin(t * .37 + .7) * .008; }
  if (lLA) { lLA.rotation.x = REST.leftLowerArm.x + Math.sin(t * .61) * .008; lLA.rotation.y = REST.leftLowerArm.y; lLA.rotation.z = REST.leftLowerArm.z + Math.sin(t * .43) * .004; }
  if (rLA) { rLA.rotation.x = REST.rightLowerArm.x + Math.sin(t * .57 + 1.4) * .008; rLA.rotation.y = REST.rightLowerArm.y; rLA.rotation.z = REST.rightLowerArm.z + Math.sin(t * .51 + .5) * .004; }
  if (lH) { lH.rotation.x = REST.leftHand.x; lH.rotation.y = REST.leftHand.y + Math.sin(t * .33) * .008; lH.rotation.z = REST.leftHand.z; }
  if (rH) { rH.rotation.x = REST.rightHand.x; rH.rotation.y = REST.rightHand.y + Math.sin(t * .29 + 1.2) * .008; rH.rotation.z = REST.rightHand.z; }
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
    applyIdle(dt);
    applyGestures(dt);
    applyBlink(dt);
  }
  renderer.render(scene, camera);
}
animate();

// ── VRM load ─────────────────────────────────────────────────────────────────
const loader = new GLTFLoader();
loader.register(parser => new VRMLoaderPlugin(parser));

const VRM_URL = './assets/Aiko.vrm';

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

window.aikoSetExpression = (name, intensity = 1.0) => {
  for (const k of Object.keys(exprTargets)) if (k !== 'blink') exprTargets[k] = 0;
  if (name && name !== 'neutral') exprTargets[name] = intensity;

  const el = document.getElementById('vrm-emotion');
  el.textContent = name ? `${name} · ${Math.round(intensity * 100)}%` : '—';
  el.className = (name && name !== 'neutral') ? 'active' : '';

  clearTimeout(exprResetTimer);
  if (name && name !== 'neutral') {
    exprResetTimer = setTimeout(() => window.aikoSetExpression('neutral'), EXPR_RESET_DELAY);
  }
};

window.aikoSetViseme = (viseme, weight = 1.0) => {
  const v = VISEME[viseme] ?? viseme;
  ['aa', 'ih', 'ou', 'ee', 'oh'].forEach(k => exprTargets[k] = 0);
  exprTargets[v] = weight;
  lastVisemeAt = performance.now();
  const dot = document.getElementById('speak-dot');
  const lbl = document.getElementById('speak-label');
  dot.className = 'dot speak';
  lbl.textContent = viseme;
  clearTimeout(window._vt);
  window._vt = setTimeout(() => { dot.className = 'dot'; lbl.textContent = 'idle'; }, 300);
};
