#!/usr/bin/env python3
"""voice-input macOS Menu Bar App.

Launches and manages the WebSocket server and Mac client from a menu bar icon.
No Terminal needed — double-click the .app bundle to start everything.

Dependencies (already installed): pyobjc, pyobjc-framework-Cocoa
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# Must be set before AppKit import on some systems
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

from AppKit import (
    NSApplication,
    NSImage,
    NSMenu,
    NSMenuItem,
    NSStatusBar,
    NSObject,
    NSTimer,
    NSWorkspace,
)
from Foundation import NSLog
import AVFoundation

# ---------------------------------------------------------------------------
# Configuration — edit these or set env vars before launching
# ---------------------------------------------------------------------------
REPO_DIR = Path(__file__).resolve().parent
VENV_PYTHON = REPO_DIR / ".venv" / "bin" / "python3"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
ARCH = "/usr/bin/arch"

SERVER_ENV = {
    **os.environ,
    "LLM_MODEL": os.environ.get("LLM_MODEL", "gemma3:4b"),
    "LLM_API_FORMAT": os.environ.get("LLM_API_FORMAT", "ollama"),
    "WHISPER_MODEL": os.environ.get("WHISPER_MODEL", "small"),
    "ARCHPREFERENCE": "arm64",
}
CLIENT_ENV = {
    **os.environ,
    "VOICE_INPUT_SERVER": os.environ.get("VOICE_INPUT_SERVER", "ws://localhost:8991"),
    "ARCHPREFERENCE": "arm64",
}

# Menu bar icon: microphone symbol (Unicode fallback, replaced by SF Symbol if available)
ICON_IDLE    = "🎙"   # server+client running
ICON_STARTING = "⏳"  # starting up
ICON_ERROR   = "⚠️"   # one or both processes died

# ---------------------------------------------------------------------------
# Process manager
# ---------------------------------------------------------------------------

def _find_pid(script_name: str) -> int | None:
    """Find PID of a running python script by name."""
    try:
        r = subprocess.run(
            ["pgrep", "-f", script_name],
            capture_output=True, text=True,
        )
        pids = [int(p) for p in r.stdout.strip().split() if p.isdigit()]
        return pids[0] if pids else None
    except Exception:
        return None


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class ProcessManager:
    def __init__(self):
        self._server: subprocess.Popen | None = None
        self._client_pid: int | None = None
        self._lock = threading.Lock()
        self._client_starting = False

    def start(self):
        with self._lock:
            self._start_server()
            threading.Timer(2.0, self._start_client).start()

    def _start_server(self):
        if self._server and self._server.poll() is None:
            return
        log_path = Path.home() / "Library" / "Logs" / "voice-input-server.log"
        log_file = open(log_path, "a")
        self._server = subprocess.Popen(
            [ARCH, "-arm64", PYTHON, str(REPO_DIR / "ws_server.py")],
            cwd=str(REPO_DIR),
            env=SERVER_ENV,
            stdout=log_file,
            stderr=log_file,
        )

    def _start_client(self):
        with self._lock:
            if self._client_starting:
                return
            if _pid_alive(self._client_pid):
                return
            # Check if mac_client.py is already running from a previous launch
            existing = _find_pid("mac_client.py")
            if existing:
                self._client_pid = existing
                return
            self._client_starting = True

        # Launch via Terminal to inherit its microphone permission
        script = str(REPO_DIR / "run_client.sh")
        applescript = f'''
tell application "Terminal"
    set w to do script "{script}"
    delay 0.5
    set miniaturized of window 1 to true
end tell
'''
        subprocess.run(["osascript", "-e", applescript],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Wait up to 8s for the client process to appear
        for _ in range(16):
            time.sleep(0.5)
            pid = _find_pid("mac_client.py")
            if pid:
                with self._lock:
                    self._client_pid = pid
                    self._client_starting = False
                return

        with self._lock:
            self._client_starting = False

    def stop(self):
        with self._lock:
            if self._server and self._server.poll() is None:
                self._server.terminate()
                try:
                    self._server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._server.kill()
            self._server = None
            if _pid_alive(self._client_pid):
                try:
                    os.kill(self._client_pid, 15)  # SIGTERM
                except OSError:
                    pass
            self._client_pid = None
            self._client_starting = False

    def restart(self):
        self.stop()
        time.sleep(0.5)
        self.start()

    @property
    def server_running(self) -> bool:
        return self._server is not None and self._server.poll() is None

    @property
    def client_running(self) -> bool:
        if self._client_starting:
            return True  # treat as running while starting
        # Refresh PID if we lost track
        if not _pid_alive(self._client_pid):
            self._client_pid = _find_pid("mac_client.py")
        return _pid_alive(self._client_pid)

    @property
    def both_running(self) -> bool:
        return self.server_running and self.client_running

    def open_log(self, which: str):
        log_path = Path.home() / "Library" / "Logs" / f"voice-input-{which}.log"
        log_path.touch(exist_ok=True)
        subprocess.Popen(["open", "-a", "Console", str(log_path)])


# ---------------------------------------------------------------------------
# AppDelegate / menu bar controller
# ---------------------------------------------------------------------------

class AppDelegate(NSObject):
    def applicationDidFinishLaunching_(self, notification):
        self._manager = ProcessManager()
        self._start_time = time.time()

        # Request microphone permission — triggers the system prompt under VoiceInput.app identity
        AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            AVFoundation.AVMediaTypeAudio,
            lambda granted: None,
        )

        # Status bar item
        bar = NSStatusBar.systemStatusBar()
        self._item = bar.statusItemWithLength_(-1)  # NSVariableStatusItemLength
        self._item.setTitle_(ICON_STARTING)
        self._item.setHighlightMode_(True)

        # Build menu
        self._menu = NSMenu.alloc().init()
        self._menu.setAutoenablesItems_(False)
        self._status_item = self._add_item("Starting…", None, enabled=False)
        self._menu.addItem_(NSMenuItem.separatorItem())
        self._add_item("Restart", "restart:", key="r")
        self._menu.addItem_(NSMenuItem.separatorItem())
        self._add_item("Open Server Log", "openServerLog:")
        self._add_item("Open Client Log", "openClientLog:")
        self._menu.addItem_(NSMenuItem.separatorItem())
        self._add_item("Quit voice-input", "quit:", key="q")

        self._item.setMenu_(self._menu)

        # Start processes
        self._manager.start()

        # Poll status every 3 seconds
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            3.0, self, "tick:", None, True
        )

        # Fallback: update icon after 10s regardless of timer
        threading.Timer(10.0, lambda: self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "tick:", None, False
        )).start()

    def _add_item(self, title, action, key="", enabled=True):
        item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            title, action if action else "", key
        )
        item.setEnabled_(enabled)
        if action:
            item.setTarget_(self)
        self._menu.addItem_(item)
        return item

    def tick_(self, _timer=None):
        if self._manager.both_running:
            self._item.setTitle_(ICON_IDLE)
            self._status_item.setTitle_("Server ✓  Client ✓")
        elif self._manager.server_running:
            self._item.setTitle_(ICON_ERROR)
            self._status_item.setTitle_("Server ✓  Client ✗ — restarting…")
            threading.Thread(target=self._manager._start_client, daemon=True).start()
        elif self._manager.client_running:
            self._item.setTitle_(ICON_ERROR)
            self._status_item.setTitle_("Server ✗  Client ✓ — restarting…")
            threading.Thread(target=self._manager.restart, daemon=True).start()
        else:
            self._item.setTitle_(ICON_ERROR)
            self._status_item.setTitle_("Both stopped — restarting…")
            threading.Thread(target=self._manager.restart, daemon=True).start()

    def restart_(self, sender):
        self._item.setTitle_(ICON_STARTING)
        self._status_item.setTitle_("Restarting…")
        threading.Thread(target=self._manager.restart, daemon=True).start()

    def openServerLog_(self, sender):
        self._manager.open_log("server")

    def openClientLog_(self, sender):
        self._manager.open_log("client")

    def quit_(self, sender):
        self._manager.stop()
        NSApplication.sharedApplication().terminate_(None)

    def applicationWillTerminate_(self, notification):
        self._manager.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(1)  # NSApplicationActivationPolicyAccessory — no Dock icon
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    app.run()


if __name__ == "__main__":
    main()
