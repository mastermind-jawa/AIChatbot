import sys
import os
from urllib.parse import parse_qs, urlencode

# Ensure the root project directory is in python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app import app

class VercelPathMiddleware:
    """
    Normalizes PATH_INFO when Vercel rewrites requests to /api/index.py.
    Uses __path query parameter passed by vercel.json rewrite to restore the exact requested route.
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        query = environ.get("QUERY_STRING", "")
        if "__path" in query:
            params = parse_qs(query, keep_blank_values=True)
            if "__path" in params:
                val = params.pop("__path")[0] if params.get("__path") else ""
                target_path = "/" + val.lstrip("/")
                environ["PATH_INFO"] = target_path
                environ["QUERY_STRING"] = urlencode(params, doseq=True)
        else:
            matched_path = (
                environ.get("HTTP_X_MATCHED_PATH")
                or environ.get("HTTP_X_VERCEL_PATH")
                or environ.get("HTTP_X_FORWARDED_URI")
            )
            if matched_path:
                if "?" in matched_path:
                    matched_path = matched_path.split("?", 1)[0]
                environ["PATH_INFO"] = matched_path

        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPathMiddleware(app.wsgi_app)
