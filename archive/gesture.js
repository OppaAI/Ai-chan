// gesture.js

export function blend(from, to, t) {
  return from + (to - from) * t;
}

export function lookAtHand(ctx) {
  const {
    side,
    intensity,
    io,
    REST,
    THREE,

    head,
    neck,

    lUA,
    lLA,
    lH,

    rUA,
    rLA,
    rH
  } = ctx;

  const s = side < 0 ? -1 : 1;

  let handBone = null;

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
}
