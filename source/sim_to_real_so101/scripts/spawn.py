import argparse 
import sys 
import termios 
import tty 
from multiprocessing.connection import Client 

OBJECTS  = {
    #"name of the object" : ["path to usd", hotkey to spawn it]
}

parser = argparse.ArgumentParser() 
parser.add_argument("--host", default="localhost")
parser.add_argument("--port", type=int, default=6001)
parser.add_argument("--authkey", default="so101-spawn", help="must match lerobot_record.py's --spawn_authkey")
args = parser.parse_args()
 
HOTKEY_TO_OBJECT = {key: (name, usd_path) for name, (usd_path, key) in OBJECTS.items()}
 
 
def send(msg: dict):
    try:
        conn = Client((args.host, args.port), authkey=args.authkey.encode())
        conn.send(msg)
        conn.close()
    except (ConnectionRefusedError, OSError) as e:
        print(f"[spawn] could not reach lerobot_record.py ({e}). Is it running?")
 
 
def read_key() -> str:
    """Read a single keypress without waiting for Enter (Unix terminals only)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
 
 
def main():
    print("SO-101 sim object spawner. Hotkeys:")
    for name, (usd_path, key) in OBJECTS.items():
        print(f"  [{key}] spawn '{name}' near sim EEF")
    print("  [c] clear spawned objects (see TODO in lerobot_record.py)")
    print("  [q] quit\n")
 
    while True:
        key = read_key()
        if key == "q":
            break
        elif key == "c":
            send({"cmd": "clear"})
            print("[spawn] sent clear")
        elif key in HOTKEY_TO_OBJECT:
            name, usd_path = HOTKEY_TO_OBJECT[key]
            send({"cmd": "spawn", "name": name, "usd_path": usd_path})
            print(f"[spawn] sent spawn('{name}')")
 
 
if __name__ == "__main__":
    main()