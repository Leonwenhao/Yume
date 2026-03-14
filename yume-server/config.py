import os
from dotenv import load_dotenv

load_dotenv()

FAL_KEY = os.getenv("FAL_KEY", "")
MARBLE_API_KEY = os.getenv("MARBLE_API_KEY", "")
MESHY_API_KEY = os.getenv("MESHY_API_KEY", "")

# fal_client reads FAL_KEY from os.environ via a cached_property on first access.
# Ensure the key is in the environment before fal_client is ever imported.
if FAL_KEY:
    os.environ["FAL_KEY"] = FAL_KEY
YUME_PORT = int(os.getenv("YUME_PORT", "8000"))
YUME_ASSETS_DIR = os.getenv("YUME_ASSETS_DIR", "./assets")
