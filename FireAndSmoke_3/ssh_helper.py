"""SSH helper script to execute commands on the remote server.

Set YOLO_REMOTE_HOST, YOLO_REMOTE_PORT, YOLO_REMOTE_USER, and
YOLO_REMOTE_PASSWORD in the environment before running this helper.
"""
import os
import paramiko
import sys


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

def run_command(cmd: str, timeout: int = 120) -> str:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=_required_env("YOLO_REMOTE_HOST"),
            port=int(os.getenv("YOLO_REMOTE_PORT", "22")),
            username=_required_env("YOLO_REMOTE_USER"),
            password=_required_env("YOLO_REMOTE_PASSWORD"),
            timeout=30,
        )
        transport = client.get_transport()
        if transport:
            transport.set_keepalive(30)
        stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
        stdout.channel.settimeout(timeout)
        stderr.channel.settimeout(timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        if err and "warning" not in err.lower():
            print(f"STDERR: {err}", file=sys.stderr)
        return out
    finally:
        client.close()

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "echo hello"
    print(run_command(cmd))
