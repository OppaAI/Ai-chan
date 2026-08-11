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

  /** Dragging that keeps a running force simulation awake while pinned. */
  function makeDrag(sim) {
    return d3.drag()
      .on('start', function (ev, d) {
        if (!ev.active) sim.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on('drag', function (ev, d) {
        d.fx = ev.x;
        d.fy = ev.y;
      })
      .on('end', function (ev, d) {
        if (!ev.active) sim.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });
  }

  /** Standard node/link forceSimulation tuned for the studio graphs.
   *  Defaults favour open spread (demo-like) over tight centering. */
  function makeSimulation(nodes, links, opts) {
    var o = Object.assign(
      {
        w: 800,
        h: 600,
        linkDistance: 80,
        linkStrength: 0.35,
        charge: -200,
        nodeRadius: function () { return 18; },
        collisionPadding: 8,
      },
      opts || {}
    );
    return d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links)
        .id(function (d) { return d.id; })
        .distance(o.linkDistance)
        .strength(o.linkStrength))
      .force('charge', d3.forceManyBody()
        .strength(o.charge)
        .distanceMax(420))
      .force('center', d3.forceCenter(o.w / 2, o.h / 2).strength(0.55))
      .force('collision', d3.forceCollide().radius(function (d) {
        return o.nodeRadius(d) + o.collisionPadding;
      }));
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
