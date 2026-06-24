"""SSH helper script to execute commands on the remote server."""
import paramiko
import sys

def run_command(cmd: str, timeout: int = 120) -> str:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname="221.14.87.239",
            port=6022,
            username="uav",
            password="Hpu@1909",
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
