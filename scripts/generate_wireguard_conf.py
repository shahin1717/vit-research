#!/usr/bin/env python3
"""
Generate WireGuard team1.conf configuration directly from .env.
Ensures private keys remain secured in .env and are not hardcoded in documentation.
"""

import os
from pathlib import Path


def load_env(env_path: Path) -> dict:
    if not env_path.exists():
        raise FileNotFoundError(
            f"Missing '{env_path}'. Please copy .env.example to .env and fill in credentials."
        )
    env_vars = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        env_vars[key.strip()] = val.strip().strip('"').strip("'")
    return env_vars


def main():
    repo_root = Path(__file__).resolve().parent.parent
    env_file = repo_root / ".env"
    env = load_env(env_file)

    required_keys = ["WG_INTERFACE_PRIVATE_KEY", "WG_PEER_PUBLIC_KEY"]
    for k in required_keys:
        if not env.get(k):
            raise ValueError(f"Missing required configuration key in .env: {k}")

    conf_content = f"""[Interface]
PrivateKey = {env.get('WG_INTERFACE_PRIVATE_KEY')}
Address = {env.get('WG_INTERFACE_ADDRESS', '10.8.0.2/32')}
DNS = {env.get('WG_INTERFACE_DNS', '1.1.1.1')}

[Peer]
PublicKey = {env.get('WG_PEER_PUBLIC_KEY')}
Endpoint = {env.get('WG_PEER_ENDPOINT', '69.197.139.29:51820')}
AllowedIPs = {env.get('WG_PEER_ALLOWED_IPS', '10.8.0.0/24')}
PersistentKeepalive = {env.get('WG_PEER_PERSISTENT_KEEPALIVE', '25')}
"""

    out_file = repo_root / "team1.conf"
    out_file.write_text(conf_content, encoding="utf-8")
    os.chmod(out_file, 0o600)
    print(f"✅ Generated WireGuard configuration: {out_file} (permissions: 600)")


if __name__ == "__main__":
    main()
