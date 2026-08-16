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
    var chargeForce = d3.forceManyBody().distanceMax(520);
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
    makeZoom: makeZoom,
    makeDrag: makeDrag,
    makeSimulation: makeSimulation,
    addGlowFilter: addGlowFilter,
  };
})(window);