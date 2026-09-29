import os
import sys

# Mock mode by default so the suite runs with zero API keys.
os.environ.setdefault("LLM_MODE", "mock")
# Offline search mock so tests are deterministic and fast (no real web).
os.environ.setdefault("SENTINEL_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
