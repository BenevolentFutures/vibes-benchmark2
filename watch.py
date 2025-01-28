from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import time
import sys

class AppReloader(FileSystemEventHandler):
    def __init__(self):
        self.process = None
        self.start_app()
    
    def start_app(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
        print("\n--- Restarting app.py ---\n")
        self.process = subprocess.Popen([sys.executable, "app.py"])
    
    def on_modified(self, event):
        if event.src_path.endswith('app.py'):
            self.start_app()

if __name__ == "__main__":
    event_handler = AppReloader()
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        if event_handler.process:
            event_handler.process.terminate()
    observer.join() 