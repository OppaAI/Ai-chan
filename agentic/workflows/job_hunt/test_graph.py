import sys
from pathlib import Path
sys.path.insert(0, str(Path('/home/oppa-ai/Aiko-chan')))
from agentic.workflows.job_hunt.graph import build_gen_job_post_graph
graph = build_gen_job_post_graph()
for node in graph.nodes:
    print(node.id, node.args)
