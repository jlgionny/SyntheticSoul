"""Handles communication with Hollow Knight mod"""
import socket
import json
import time
from config import HOST, PORT

class GameCommunicator:
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port
        self.sock = None
        self.connected = False
        # FIX: Persistent buffer to handle partial/multiple JSON messages
        self.buffer = ""

    def connect(self, max_retries=10, retry_delay=2):
        """Connect to the game mod"""
        print(f"Connecting to game on {self.host}:{self.port}...")
        
        # FIX: Close existing socket if any before reconnecting
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        
        # FIX: Clear buffer on new connection
        self.buffer = ""
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # FIX: Set socket timeout to avoid infinite blocking
        self.sock.settimeout(10.0)
        
        for attempt in range(max_retries):
            try:
                self.sock.connect((self.host, self.port))
                self.connected = True
                print(f"✓ Connected to game on attempt {attempt + 1}!")
                return True
            except (ConnectionRefusedError, socket.timeout, OSError) as e:
                print(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"  Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
        
        print("✗ Failed to connect to game")
        self.connected = False
        return False

    def receive_state(self):
        """
        FIX: Receive game state from mod with proper message framing.
        Handles TCP stream coalescing by using a persistent buffer.
        """
        if not self.connected:
            return None

        try:
            # FIX: Loop until we have at least one complete message (terminated by \n)
            while '\n' not in self.buffer:
                # Set timeout for recv
                self.sock.settimeout(5.0)
                
                # Receive new data
                chunk = self.sock.recv(4096)
                
                if not chunk:
                    # Socket closed by remote
                    print("✗ Socket closed by remote (empty chunk)")
                    self.connected = False
                    self.buffer = ""  # Clear buffer on disconnect
                    return None
                
                # FIX: Decode and append to buffer
                self.buffer += chunk.decode('utf-8')
            
            # FIX: Extract first complete message (up to first \n)
            newline_index = self.buffer.index('\n')
            message = self.buffer[:newline_index].strip()
            
            # FIX: Keep remainder in buffer for next call
            self.buffer = self.buffer[newline_index + 1:]
            
            # Parse JSON from the single complete message
            if not message:
                # Empty message, try again
                return self.receive_state()
            
            try:
                state = json.loads(message)
                return state
            except json.JSONDecodeError as e:
                print(f"✗ JSON decode error on message: {message[:100]}...")
                print(f"  Error: {e}")
                # Continue to next message in buffer if available
                if '\n' in self.buffer:
                    return self.receive_state()
                return None
            
        except socket.timeout:
            print("✗ Socket timeout waiting for state")
            self.connected = False
            self.buffer = ""
            return None
        except UnicodeDecodeError as e:
            print(f"✗ Unicode decode error: {e}")
            self.buffer = ""  # Clear corrupted buffer
            return None
        except Exception as e:
            print(f"✗ Error receiving state: {type(e).__name__}: {e}")
            self.connected = False
            self.buffer = ""
            return None

    def send_action(self, action):
        """Send action to game mod"""
        if not self.connected:
            return False

        try:
            message = f"{action}\n".encode('utf-8')
            self.sock.sendall(message)
            return True
        except Exception as e:
            print(f"✗ Error sending action: {e}")
            self.connected = False
            return False

    def close(self):
        """Close connection"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        self.connected = False
        self.buffer = ""  # Clear buffer
        print("Connection closed")
