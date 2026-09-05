/* graph-bootstrap.js — shared helpers for the Aiko graph studios.
 *
 * Prior to this module each studio (ltm, kb, dag) hand-rolled its own
 * API-base bootstrap and its own D3 zoom / drag / forceSimulation wiring.
 * Studios that still need bespoke rendering keep it client-side, but any
 * studio importing this file gets one consistent bootstrap.
 *
 * Dependencies: d3@7 must be loaded before this script.
 */
(function (global) {
  'use strict';

  /** Unified API root. Standalone pages get "/api" (their app defines
   *  `/api/*` at the root); mounted pages get "/studio/<path>/api". */
  function apiBase() {
    return global.location.pathname.replace(/\/+$/, '') + '/api';
  }

  /** Standard zoom behavior with an optional transform target. */
  function makeZoom(opts) {
    var o = Object.assign(
      { scaleExtent: [0.2, 4], target: null, onZoom: null },
      opts || {}
    );
    return d3.zoom()
      .scaleExtent(o.scaleExtent)
      .on('zoom', function (ev) {
        if (o.target) o.target.attr('transform', ev.transform);
        if (o.onZoom) o.onZoom(ev);
      });
  }

  /** Dragging that keeps a running force simulation awake while pinned.
   *  Accepts either a simulation instance or a getter function () => simulation. */
  function makeDrag(sim) {
    var getSim = typeof sim === 'function' ? sim : function () { return sim; };
    return d3.drag()
      .on('start', function (ev, d) {
        var s = getSim();
        if (!ev.active && s) s.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on('drag', function (ev, d) {
        d.fx = ev.x;
        d.fy = ev.y;
      })
      .on('end', function (ev, d) {
        var s = getSim();
        if (!ev.active && s) s.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });
  }

  /** Standard node/link forceSimulation tuned for the studio graphs.
   *  Defaults favour open, organic branching (neural-net-like) over the
   *  uniform packed-disc look you get from strong charge + strong center.
   *  Tuning notes:
   *   - charge is weaker (-90 vs -200) so nodes don't shove each other into
   *     a uniform lattice; clusters are held together by links instead.
   *   - center pull is much softer (0.02 vs 0.12) so the graph can sprawl
   *     into whatever shape its link topology wants, rather than collapsing
   *     into a circle around the canvas center.
   *   - link strength is higher (0.55 vs 0.35) so connected nodes actually
   *     pull into visible clusters/branches instead of floating independently.
   *   - an optional 'cluster' force nudges same-type nodes toward a shared
   *     centroid without hard-pinning them, producing the loose "lobes"
   *     look (memory cluster / entity cluster / episode cluster) instead of
   *     a flat left-to-right type split. */
  function makeSimulation(nodes, links, opts) {
    var o = Object.assign(
      {
        w: 800,
        h: 600,
        linkDistance: 90,
        linkStrength: 0.55,
        charge: -90,
        centerStrength: 0.02,
        clusterStrength: 0.06,
        clusterKey: function (d) { return d.type; },
        nodeRadius: function () { return 18; },
        collisionPadding: 6,
      },
      opts || {}
    );
    var centerStrength = (o.centerStrength != null) ? o.centerStrength : 0.02;
    // distanceMax raised further (1400 → 1800): with memory nodes now at -550
    // and entity nodes at -180, the repulsion range needs to extend far enough
    // for memory nodes to feel each other's push across the canvas, while entities
    // stay close to their mention sources.
    var chargeForce = d3.forceManyBody().distanceMax(1800);
    // charge may be a number or a per-node function
    chargeForce.strength(o.charge);

    var sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links)
        .id(function (d) { return d.id; })
        .distance(o.linkDistance)
        .strength(o.linkStrength))
      .force('charge', chargeForce)
      .force('center', d3.forceCenter(o.w / 2, o.h / 2).strength(centerStrength))
      .force('collision', d3.forceCollide().radius(function (d) {
        return o.nodeRadius(d) + o.collisionPadding;
      }));

    if (o.clusterStrength > 0) {
      sim.force('cluster', forceCluster(nodes, o.clusterKey, o.clusterStrength));
    }
    return sim;
  }

  /** Weak custom force: pulls nodes sharing the same clusterKey(d) toward
   *  their group's running centroid. Recomputes centroids each tick so
   *  clusters can drift and settle organically instead of being pinned to
   *  fixed coordinates. Strength should stay low (0.03–0.1) — this is meant
   *  to *bias* the layout into loose lobes, not override link/charge forces. */
  function forceCluster(nodes, keyFn, strength) {
    var s = strength == null ? 0.06 : strength;
    function force(alpha) {
      var centroids = {};
      var counts = {};
      for (var i = 0; i < nodes.length; i++) {
        var k = keyFn(nodes[i]);
        if (!centroids[k]) { centroids[k] = { x: 0, y: 0 }; counts[k] = 0; }
        centroids[k].x += nodes[i].x;
        centroids[k].y += nodes[i].y;
        counts[k] += 1;
      }
      for (var key in centroids) {
        centroids[key].x /= counts[key];
        centroids[key].y /= counts[key];
      }
      for (var j = 0; j < nodes.length; j++) {
        var n = nodes[j];
        var c = centroids[keyFn(n)];
        n.vx += (c.x - n.x) * s * alpha;
        n.vy += (c.y - n.y) * s * alpha;
      }
    }
    force.initialize = function () {};
    return force;
  }

  /* ── auth-expiry banner ────────────────────────────────────────────
   * Studio /api/* calls 401 when the login session is gone (server
   * restart wipes the in-memory session table; an ephemeral SECRET_KEY
   * does it on every restart). Frontends otherwise render silently
   * empty, which looks like broken data. This fetch watcher shows one
   * dismissible "log in again" banner on the first same-origin /api
   * 401 and otherwise leaves the response untouched, so every studio
   * that loads this file (directly or transitively) gets the hint
   * with no per-frontend changes.
   */
  var _authBannerShown = false;

  function showAuthBanner() {
    if (_authBannerShown) return;
    _authBannerShown = true;
    try {
      var bar = document.createElement('div');
      bar.setAttribute('role', 'alert');
      bar.setAttribute('id', 'aiko-auth-banner');
      bar.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:9999;'
        + 'background:#2a1b3d;color:#f3eaff;font:14px/1.4 system-ui,sans-serif;'
        + 'padding:10px 16px;text-align:center;border-bottom:2px solid #967bb6;';
      var msg = document.createElement('span');
      msg.textContent = 'Session expired — ';
      var link = document.createElement('a');
      link.href = '/';
      link.textContent = 'log in again';
      link.style.cssText = 'color:#d9c2ff;font-weight:700;';
      var tail = document.createElement('span');
      tail.textContent = ' to reload studio data.';
      var dismiss = document.createElement('button');
      dismiss.textContent = '✕';
      dismiss.setAttribute('aria-label', 'Dismiss');
      dismiss.style.cssText = 'margin-left:12px;background:none;border:1px solid #967bb6;'
        + 'color:#f3eaff;border-radius:4px;cursor:pointer;padding:0 8px;';
      dismiss.onclick = function () { bar.remove(); };
      bar.appendChild(msg);
      bar.appendChild(link);
      bar.appendChild(tail);
      bar.appendChild(dismiss);
      document.body.appendChild(bar);
    } catch (e) { /* banner must never break the page */ }
  }

  function isStudioApi(url) {
    if (typeof url !== 'string') return false;
    if (url.indexOf('http') === 0) {
      try {
        if (new URL(url).origin !== global.location.origin) return false;
      } catch (e) { return false; }
    }
    return url.indexOf('/api') !== -1;
  }

  function watchAuth() {
    if (global.fetch && !global.fetch.__aikoAuthWatch) {
      var origFetch = global.fetch;
      var watched = function (url, opts) {
        return origFetch.apply(this, arguments).then(function (resp) {
          try {
            if (resp && resp.status === 401 && isStudioApi(typeof url === 'string' ? url : (url && url.url))) {
              showAuthBanner();
            }
          } catch (e) { /* observe-only */ }
          return resp;
        });
      };
      watched.__aikoAuthWatch = true;
      global.fetch = watched;
    }
    return { showAuthBanner: showAuthBanner };
  }

  // Auto-install: every studio page loading this file gets the banner.
  try { watchAuth(); } catch (e) { /* observe-only */ }

  /** Soft outer glow SVG filter. Returns the filter node id. */
  function addGlowFilter(defs, id, stdDeviation) {
    var d = stdDeviation == null ? 1.5 : stdDeviation;
    var filt = defs.append('filter')
      .attr('id', id)
      .attr('x', '-80%').attr('y', '-80%')
      .attr('width', '260%').attr('height', '260%');
    filt.append('feGaussianBlur')
      .attr('in', 'SourceGraphic').attr('stdDeviation', d)
      .attr('result', 'b');
    filt.append('feMerge')
      .selectAll('feMergeNode')
      .data(['b', 'SourceGraphic'])
      .join('feMergeNode')
      .attr('in', function (x) { return x; });
    return id;
  }

  global.GraphBoot = {
    apiBase: apiBase,
    watchAuth: watchAuth,
    showAuthBanner: showAuthBanner,
    makeZoom: makeZoom,
    makeDrag: makeDrag,
    makeSimulation: makeSimulation,
    addGlowFilter: addGlowFilter,
  };
})(window);