import http.server
import socketserver
import webbrowser
import os

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving at port {PORT} from directory {DIRECTORY}")
    url = f"http://localhost:{PORT}/index.html"
    print(f"Opening {url} in your default web browser...")
    webbrowser.open_new_tab(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.server_close()
