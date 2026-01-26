import socket
import json
import time


class HollowKnightEnv:
    """
    Environment wrapper for Hollow Knight.
    Communicates with the C# mod via TCP socket.
    """

    # Action names that match ActionExecutor.cs exactly
    ACTIONS = {
        0: "MOVE_LEFT",
        1: "MOVE_RIGHT",
        2: "UP",
        3: "DOWN",
        4: "JUMP",
        5: "ATTACK",
        6: "DASH",
        7: "SPELL",
    }

    def __init__(self, host="localhost", port=5555, timeout=30.0):
        """
        Initialize the environment.

        Args:
            host: Hostname of the mod server (default: 'localhost')
            port: Port of the mod server (default: 5555)
            timeout: Socket timeout in seconds (default: 30.0)
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.socket_file = None
        self.connected = False
        self._connect()

    def _connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))

            # File wrapper to use readline()
            self.socket_file = self.socket.makefile("r", encoding="utf - 8")
            self.connected = True
            print(
                f"[Env] Connected to {self.host}:{self.port} (timeout: {self.timeout}s)"
            )
        except Exception as e:
            print(f"[Env] Connection failed: {e}")

    def _send_action(self, action_name):
        try:
            if not self.socket:
                return

            # C# expects just the command string + newline
            message = f"{action_name}\n"
            self.socket.sendall(message.encode("utf - 8"))
        except BrokenPipeError:
            print("[Env] Broken pipe sending action. Reconnecting...")
            self.connected = False
            self._connect()
        except Exception as e:
            print(f"[Env] Error sending action: {e}")

    def _receive_state(self):
        try:
            if not self.socket_file:
                return None

            line = self.socket_file.readline()
            if not line:
                # EOF
                self.connected = False
                return None

            line = line.strip()
            if not line:
                return None

            state = json.loads(line)
            return state
        except json.JSONDecodeError as e:
            print(f"[Env] JSON decode error: {e}")
            return None
        except socket.timeout:
            print(f"[Env] Socket timeout (waited {self.timeout}s)")
            return None
        except Exception as e:
            print(f"[Env] Error receiving state: {e}")
            return None

    def reset(self):
        """
        Reset the environment.

        With the instant scene reload system, this method waits for the
        mod to complete the reload sequence:
        1. Death/Victory detected
        2. Terminal state sent to Python
        3. 3.0s death animation delay
        4. Scene reload initiated
        5. Scene loads (~2 - 3s)
        6. Hero restoration
        7. First state sent

        Total: ~5 - 6 seconds expected
        """
        print("[Env] Reset requested - waiting for new episode state...")

        # Send IDLE to confirm reception
        self._send_action("IDLE")

        # Wait for first valid state
        state = self._receive_state()

        # Retry with patience for scene reload
        attempts = 0
        max_attempts = 20  # 20 * 0.5s = 10s max wait

        while state is None and attempts < max_attempts:
            time.sleep(0.5)
            state = self._receive_state()
            attempts += 1

            if attempts % 4 == 0:  # Log every 2 seconds
                print(
                    f"[Env] Still waiting for state... ({attempts * 0.5:.1f}s elapsed)"
                )

        if state is None:
            print(
                "[Env] WARNING: No state received after reset - returning empty state"
            )
            return {}

        print(f"[Env] ✓ Reset complete - received state after {attempts * 0.5:.1f}s")
        return state

    def step(self, action):
        """
        Execute one action and receive the next state.

        Args:
            action: Integer action index (0 - 7)

        Returns:
            state: Dictionary containing game state
            done: Boolean indicating episode termination
            info: Dictionary with additional information
        """
        action_name = self.ACTIONS.get(action, "IDLE")
        self._send_action(action_name)

        state = self._receive_state()

        # Handle disconnection or error
        if state is None:
            done = True
            state = {}  # Empty fallback state
        else:
            done = state.get("isDead", False) or state.get("bossDefeated", False)

        info = {
            "action_name": action_name,
            "timestamp": state.get("timestamp", 0) if state else 0,
        }

        return state, done, info

    def close(self):
        """Close the socket connection."""
        if self.socket_file:
            try:
                self.socket_file.close()
            except Exception:
                pass

        if self.socket:
            try:
                self.socket.close()
                print("[Env] Connection closed")
            except Exception:
                pass

        self.connected = False
        self.socket_file = None
        self.socket = None
