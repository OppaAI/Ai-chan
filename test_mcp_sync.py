import sys
from pathlib import Path
sys.path.insert(0, str(Path('/home/oppa-ai/Aiko-chan')))
from agentic.registry import registry

spec = registry.get("read_email")
if spec:
    print("read_email found")
    print("handler:", spec.handler)
    if spec.handler:
        import asyncio
        import inspect
        print("iscoroutinefunction:", inspect.iscoroutinefunction(spec.handler))
else:
    print("read_email not found")
