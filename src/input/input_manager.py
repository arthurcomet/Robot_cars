import threading
import time
import sys
import termios
import tty
import numpy as np

class InputManager:
    def __init__(self, client):
        self.client = client
        self.speed_increment = 0.1
        self.steering_increment = 0.2
        self.current_speed = 0.0
        self.current_steering = 0.0
        self.running = False
        self.thread = None
        self.old_settings = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._input_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def _get_key(self):
        try:
            return sys.stdin.read(1)
        except:
            return None

    def _input_loop(self):
    self.old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        while self.running:
            try:
                key = self._get_key()
                self._handle_key(key)
                self._apply_decay_if_no_input(key)
                self._send_current_actions()
                time.sleep(0.05)
            except Exception as e:
                print(f"Erreur lors de la lecture des touches: {e}")
                self.running = False
                break
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def _handle_key(self, key):
        if not key:
            return

        if key == 'q':
            self.running = False
        elif key == '\x1b':
            sequence = sys.stdin.read(2)
            if sequence == '[A':
                self.current_speed = min(1.0, self.current_speed + self.speed_increment)
            elif sequence == '[B':
                self.current_speed = max(-1.0, self.current_speed - self.speed_increment)
            elif sequence == '[C':
                self.current_steering = min(1.0, self.current_steering + self.steering_increment)
            elif sequence == '[D':
                self.current_steering = max(-1.0, self.current_steering - self.steering_increment)

    def _apply_decay_if_no_input(self, key):
        if not key:
            self.current_speed *= 0.95 if abs(self.current_speed) >= 0.05 else 0.0
            self.current_steering *= 0.85 if abs(self.current_steering) >= 0.05 else 0.0

    def _send_current_actions(self):
        actions = np.array([[self.current_speed, self.current_steering]], dtype=np.float32)
        self.client.send_actions(actions)