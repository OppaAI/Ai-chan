// gesture.js
export function blend(from, to, t) {
  return from + (to - from) * t;
}

export function hairBrush(ctx) {
  const {
    side,
    intensity,
    t,
    strokeSpeed = 0.9,
    io,
    REST,
    head,
    lUA,
    lLA,
    lH,
    rUA,
    rLA,
    rH
  } = ctx;

  const left = side < 0;
  const sign = left ? -1 : 1;

  const UA = left ? lUA : rUA;
  const LA = left ? lLA : rLA;
  const H  = left ? lH  : rH;

  const restUA = left ? REST.leftUpperArm : REST.rightUpperArm;
  const restLA = left ? REST.leftLowerArm : REST.rightLowerArm;
  const restH  = left ? REST.leftHand : REST.rightHand;

  const ioUA = left ? io.lUA : io.rUA;
  const ioLA = left ? io.lLA : io.rLA;
  const ioH  = left ? io.lH  : io.rH;

  // --------------------------------------------------
  // Brush cycle
  // 0.0 = temple
  // 0.7 = shoulder
  // 1.0 = quick return
  // --------------------------------------------------

  const cycle = (t * strokeSpeed) % 1;

  let brush;

  if (cycle < 0.75) {
    // slow downward stroke
    brush = cycle / 0.75;
  } else {
    // quick lift
    brush = 1 - (cycle - 0.75) / 0.25;
  }

  // smooth easing
  brush = brush * brush * (3 - 2 * brush);

  // --------------------------------------------------
  // Upper arm
  // --------------------------------------------------

  if (UA) {

    UA.rotation.z = blend(
      restUA.z + ioUA.z,
      sign * (0.93 + brush * 0.05) * intensity,
      1
    );

    UA.rotation.x = blend(
      restUA.x + ioUA.x,
      -(0.55 - brush * 0.05) * intensity,
      1
    );
  }

  // --------------------------------------------------
  // Lower arm
  // Most of the movement
  // --------------------------------------------------

  if (LA) {

    LA.rotation.x = blend(
      restLA.x + ioLA.x,
      -(1.35 - brush * 0.55) * intensity,
      1
    );

    LA.rotation.z = blend(
      restLA.z + ioLA.z,
      sign * 0.16 * intensity,
      1
    );
  }

  // --------------------------------------------------
  // Wrist
  // --------------------------------------------------

  if (H) {

    H.rotation.y = blend(
      restH.y + ioH.y,
      -sign * (0.55 - brush * 0.18) * intensity,
      1
    );

    H.rotation.x = blend(
      restH.x + ioH.x,
      -(0.30 + brush * 0.15) * intensity,
      1
    );

    H.rotation.z = blend(
      (restH.z ?? 0) + (ioH.z ?? 0),
      sign * (0.20 + brush * 0.08) * intensity,
      1
    );
  }

  // --------------------------------------------------
  // Head
  // --------------------------------------------------

  if (head) {

    head.rotation.z = blend(
      io.head.z,
      -sign * 0.08 * intensity,
      1
    );

    head.rotation.y = blend(
      io.head.y,
      sign * (0.04 + brush * 0.02) * intensity,
      1
    );

    head.rotation.x = blend(
      io.head.x,
      -0.03 * intensity,
      1
    );
  }
}

export function lookAtHand(ctx) {
  const {
    side,
    intensity,
    io,
    REST,
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

  if (s < 0) {
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

  if (head) {
    head.rotation.y = blend(
      io.head.y,
      s * 0.18 * intensity,
      1
    );
    head.rotation.x = blend(
      io.head.x,
      -0.10 * intensity,
      1
    );
  }
  if (neck) {
    neck.rotation.y = blend(
      io.neck.y,
      s * 0.10 * intensity,
      1
    );
    neck.rotation.x = blend(
      io.neck.x,
      -0.06 * intensity,
      1
    );
  }
}
