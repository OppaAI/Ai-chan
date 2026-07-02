"""
Aiko navigation backend: OSRM routing server hosted on Modal.

Usage:
    modal run modal_osrm_app.py::build_graph      # one-time: download + preprocess BC OSM extract
    modal deploy modal_osrm_app.py                # deploy the always-available (scale-to-zero) endpoint

Once deployed, hit:
    https://<your-workspace>--aiko-osrm-route.modal.run/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false&steps=true

which is standard OSRM HTTP API syntax. From think.py's router, this is just an httpx call.
"""

import subprocess
import modal

app = modal.App("aiko-osrm")

# Persistent volume holds the raw .osm.pbf extract AND the processed .osrm graph files,
# so the 20-60s preprocessing step only ever runs once (or when you refresh the map data).
osm_volume = modal.Volume.from_name("aiko-osm-data", create_if_missing=True)
VOLUME_PATH = "/data"

# Official OSRM backend image ships osrm-extract / osrm-partition / osrm-contract / osrm-routed.
osrm_image = modal.Image.from_registry(
    "ghcr.io/project-osrm/osrm-backend:latest",
    add_python="3.11",
).pip_install("httpx")

# Swap this for a smaller/larger region as needed. Geofabrik hosts sub-provincial extracts too
# (e.g. british-columbia is already fairly small, but you could go city-level if you want faster
# builds and don't need province-wide routing).
OSM_EXTRACT_URL = "https://download.geofabrik.de/north-america/canada/british-columbia-latest.osm.pbf"
PBF_PATH = f"{VOLUME_PATH}/bc.osm.pbf"
GRAPH_BASENAME = f"{VOLUME_PATH}/bc"  # osrm files will be bc.osrm, bc.osrm.hsgr, etc.


@app.function(
    image=osrm_image,
    volumes={VOLUME_PATH: osm_volume},
    timeout=3600,  # extract/partition/contract can take a while on first run
    cpu=4,
)
def build_graph():
    """One-time (or occasional-refresh) job: download OSM extract and build the OSRM graph."""
    subprocess.run(["curl", "-L", "-o", PBF_PATH, OSM_EXTRACT_URL], check=True)

    # Contraction Hierarchies pipeline (fast queries, standard for driving directions).
    subprocess.run(["osrm-extract", "-p", "/opt/car.lua", PBF_PATH], check=True, cwd=VOLUME_PATH)
    subprocess.run(["osrm-partition", GRAPH_BASENAME + ".osrm"], check=True)
    subprocess.run(["osrm-contract", GRAPH_BASENAME + ".osrm"], check=True)

    osm_volume.commit()  # persist the built graph files
    print("Graph build complete.")


@app.function(
    image=osrm_image,
    volumes={VOLUME_PATH: osm_volume},
    scaledown_window=120,  # keep warm for 2 min after last request, then scale to zero
    cpu=2,
    memory=2048,
)
@modal.web_server(port=5000, startup_timeout=60)
def route():
    """Serves the standard OSRM HTTP API at /route/v1/driving/... , /nearest/... etc."""
    subprocess.Popen(
        ["osrm-routed", "--algorithm", "mld", GRAPH_BASENAME + ".osrm"],
        cwd=VOLUME_PATH,
    )
