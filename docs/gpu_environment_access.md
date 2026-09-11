# AI Academy - GPU Environment Access (Team 1)

> [!CAUTION]
> **Security Notice:** Sensitive credentials, WireGuard private keys, and Jupyter authentication tokens must never be committed to source control. They are stored securely in your local `.env` file (which is added to `.gitignore`).

---

## 1. Connect to the VPN

### Option A: Automatic Generation from `.env` (Recommended)

Generate your `team1.conf` configuration file directly from your local `.env` file:

```bash
python scripts/generate_wireguard_conf.py
```

This creates `team1.conf` with restricted read-only permissions (`chmod 600`).

### Option B: Manual Configuration

Populate your `.env` file from the template (`cp .env.example .env`) with your assigned credentials:

```bash
# WireGuard VPN Configuration in .env
WG_INTERFACE_PRIVATE_KEY="<YOUR_PRIVATE_KEY>"
WG_INTERFACE_ADDRESS="10.8.0.2/32"
WG_INTERFACE_DNS="1.1.1.1"

WG_PEER_PUBLIC_KEY="<YOUR_PEER_PUBLIC_KEY>"
WG_PEER_ENDPOINT="69.197.139.29:51820"
WG_PEER_ALLOWED_IPS="10.8.0.0/24"
WG_PEER_PERSISTENT_KEEPALIVE="25"
```

The resulting `team1.conf` template follows this structure:

```ini
[Interface]
PrivateKey = ${WG_INTERFACE_PRIVATE_KEY}
Address = ${WG_INTERFACE_ADDRESS}
DNS = ${WG_INTERFACE_DNS}

[Peer]
PublicKey = ${WG_PEER_PUBLIC_KEY}
Endpoint = ${WG_PEER_ENDPOINT}
AllowedIPs = ${WG_PEER_ALLOWED_IPS}
PersistentKeepalive = ${WG_PEER_PERSISTENT_KEEPALIVE}
```

Import `team1.conf` into the **WireGuard** client (available for Windows, macOS, Linux, iOS, Android).  
Activate the tunnel. You should observe an active handshake within a few seconds.

---

## 2. Open Your Workspace

Once the VPN tunnel is connected, open the JupyterLab instance URL formatted with your team token:

```text
http://${JUPYTER_HOST}:${JUPYTER_PORT}/${JUPYTER_TEAM_PATH}/?token=${JUPYTER_TOKEN}
```

*(Refer to `JUPYTER_URL` in your local `.env` file for the exact authenticated link)*.

You will land directly in your team's remote JupyterLab environment.

---

## 3. Notes & Best Practices

* **Privacy:** Keep your `.env` and `team1.conf` private - they grant direct access to your team's workspace and GPU allocation.
* **Connectivity Issues:** If the Jupyter page does not load, verify that your WireGuard tunnel shows active data transmission (transfer bytes increasing and handshake timestamp recent).
* **Dependencies:** The remote environment ships with standard JupyterLab. Install any project-specific packages directly in your notebook or terminal:
  ```bash
  pip install -r requirements.txt
  ```
