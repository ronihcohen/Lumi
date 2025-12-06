import http.server
import socketserver
import sys
import os
import json

PORT = 8000
DIRECTORY = "ui"
DATA_DIRECTORY = "data"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def do_GET(self):
        # API endpoint to list available books
        if self.path == '/api/books':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            
            try:
                # Find all .mp3 files in data directory
                books = []
                data_path = os.path.join(DIRECTORY, DATA_DIRECTORY)
                if os.path.exists(data_path):
                    for file in os.listdir(data_path):
                        if file.endswith('.mp3'):
                            # Remove .mp3 extension to get base name
                            base_name = file[:-4]
                            # Verify that the required files exist
                            words_file = f"{base_name}_transcribed_words.json"
                            sync_file = f"{base_name}_sync_map.json"
                            if (os.path.exists(os.path.join(data_path, words_file)) and 
                                os.path.exists(os.path.join(data_path, sync_file))):
                                books.append(base_name)
                
                books.sort()  # Sort alphabetically
                self.wfile.write(json.dumps(books).encode())
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return
        
        # Default file serving
        return super().do_GET()
    
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
