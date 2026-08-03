import sys
from pathlib import Path

# Add mcp_server to path
sys.path.insert(0, str(Path(__file__).parent / "interface" / "mcp_server"))

from system.config import load_config
load_config()

from interface.mcp_server.social.services.discord import post_discord

result = post_discord(text="Test message from Aiko")
print(result)