# Simple shim so "gunicorn app:app" works when the real app is in api.py
from api import app  # re-export FastAPI instance as app