/**
 * Aiko GPS capture module.
 * Include in index.html with: <script src="gps.js"></script>
 * (before your main app script, since that script will construct an AikoLocation instance)
 *
 * Notes specific to phones:
 * - Geolocation requires HTTPS (or localhost). If AuRoRA's browser client is served over plain
 *   http:// from the Jetson's LAN IP, the phone browser will refuse to grant location. You'll
 *   need a self-signed cert or a reverse proxy (Caddy/nginx) terminating TLS in front of it.
 * - watchPosition (not getCurrentPosition) is what you want for "navigate me there" style
 *   turn-by-turn, since it keeps streaming updated fixes as the phone moves.
 * - iOS Safari and Android Chrome both prompt once per origin; the permission persists after
 *   that unless the user revokes it in browser settings.
 * - Must be started from a user gesture (button tap) — browsers block silent permission requests.
 */

class AikoLocation {
  constructor(ws) {
    this.ws = ws;           // pass in your existing audio WebSocket, or a second one
    this.watchId = null;
    this.lastFix = null;
  }

  startWatching() {
    if (!("geolocation" in navigator)) {
      console.error("Geolocation not supported on this browser.");
      return;
    }

    this.watchId = navigator.geolocation.watchPosition(
      (pos) => this._onFix(pos),
      (err) => this._onError(err),
      {
        enableHighAccuracy: true,  // use GPS chip, not just wifi/cell triangulation
        maximumAge: 5000,          // accept a cached fix up to 5s old
        timeout: 10000,
      }
    );
  }

  stopWatching() {
    if (this.watchId !== null) {
      navigator.geolocation.clearWatch(this.watchId);
      this.watchId = null;
    }
  }

  _onFix(position) {
    const { latitude, longitude, accuracy, heading, speed } = position.coords;
    this.lastFix = { latitude, longitude, accuracy, heading, speed, ts: Date.now() };

    // Sent as a typed message alongside your PCM audio frames, so think.py's router
    // can distinguish location updates from audio chunks on the same socket.
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "gps_update", ...this.lastFix }));
    }
  }

  _onError(err) {
    // code 1 = permission denied, 2 = position unavailable, 3 = timeout
    console.error("Geolocation error:", err.code, err.message);
  }

  // One-shot fetch for "where am I" style queries that don't need continuous tracking.
  getOnce() {
    return new Promise((resolve, reject) => {
      navigator.geolocation.getCurrentPosition(
        (pos) => resolve(pos.coords),
        (err) => reject(err),
        { enableHighAccuracy: true, timeout: 10000 }
      );
    });
  }
}
