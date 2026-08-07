"""See cognitions/memory/grasp_hub.install_into_think — temporary restore.

This file was accidentally overwritten; restore from dev is required.
The live install block should be re-applied after restore:

        try:
            from cognition.memory.grasp_hub import install_into_think
            if install_into_think(think_ref):
                log.info("[wakeup] Grasp live hub installed on AikoThink")
        except Exception:
            log.debug("[wakeup] Grasp live hub not installed", exc_info=True)

Placed after: raise RuntimeError("AikoThink boot failed") from think_exc
"""
raise RuntimeError(
    "system/wakeup.py was corrupted in a partial commit. "
    "Restore from dev and re-apply the grasp_hub install block — see module docstring."
)
