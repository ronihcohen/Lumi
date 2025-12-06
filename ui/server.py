import http.server
import socketserver
import sys

PORT = 8000
DIRECTORY = "ui"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def end_headers(self):
        # Explicitly ensure Range support header is sent
        self.send_header('Accept-Ranges', 'bytes')
        
        # Add aggressive caching for large static files
        path = self.translate_path(self.path)
        if path.endswith(('.mp3', '.json')):
            # Cache for 1 year (files don't change often)
            self.send_header('Cache-Control', 'public, max-age=31536000, immutable')
        elif path.endswith(('.html', '.js', '.css')):
            # Cache for 1 hour (code might change during development)
            self.send_header('Cache-Control', 'public, max-age=3600')
        
        super().end_headers()

class QuietTCPServer(socketserver.TCPServer):
    allow_reuse_address = True
    
    def handle_error(self, request, client_address):
        # Suppress benign connection errors (client disconnects/cancels)
        exc_type, exc_value = sys.exc_info()[:2]
        if exc_type in (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            return  # Silently ignore these
        # For other errors, use default handling
        super().handle_error(request, client_address)

try:
    with QuietTCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        print(f"Open your browser and navigate to http://localhost:{PORT}")
        print("Range requests enabled for audio seeking")
        httpd.serve_forever()
except OSError as e:
    if "address already in use" in str(e).lower() or "10048" in str(e):
        PORT = 8001
        print(f"Port 8000 in use, trying {PORT}")
        with QuietTCPServer(("", PORT), Handler) as httpd:
            print(f"Serving at http://localhost:{PORT}")
            print(f"Open your browser and navigate to http://localhost:{PORT}")
            print("Range requests enabled for audio seeking")
            httpd.serve_forever()
    else:
        raise
