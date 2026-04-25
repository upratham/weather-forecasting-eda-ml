import os
import sys

# Ensure project root is on the path so `backend` package resolves
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from mangum import Mangum
from backend.main import app  # noqa: E402

handler = Mangum(app, lifespan="off")
