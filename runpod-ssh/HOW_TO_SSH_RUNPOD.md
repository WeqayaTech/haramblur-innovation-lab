# How to SSH into a RunPod Instance from VSCode

This guide walks you through connecting to a RunPod CPU/GPU pod via SSH from your local machine and VSCode — including SCP and SFTP support.

---

## Prerequisites

- A RunPod account with an active pod
- SSH key pair on your local machine (`id_ed25519` recommended)
- VSCode with the **Remote - SSH** extension installed

---

## Step 1 — Find your pod's connection details

In the RunPod dashboard, open your pod and look for **SSH over exposed TCP** (this supports SCP & SFTP, required for VSCode Remote):

```
Host:  <POD_IP>
Port:  <POD_PORT>
User:  root
```

Example format shown in RunPod UI:
```
ssh root@<POD_IP> -p <POD_PORT> -i ~/.ssh/id_ed25519
```

> Note: Do NOT use the standard SSH option (ssh.runpod.io) for VSCode — it does not support SCP/SFTP. Always use the **SSH over exposed TCP** option.

---

## Step 2 — Get your local public key

On your local machine, run:

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the full output (it starts with `ssh-ed25519 AAAA...`).

If you don't have an SSH key yet, generate one:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

---

## Step 3 — Add your public key to the pod

Open the **Web Terminal** in the RunPod dashboard and run the following (paste your key in place of the placeholder):

```bash
mkdir -p ~/.ssh && \
echo "ssh-ed25519 AAAA...your-full-public-key..." >> ~/.ssh/authorized_keys && \
chmod 700 ~/.ssh && \
chmod 600 ~/.ssh/authorized_keys
```

> You must do this every time you start a fresh pod — RunPod does not persist `authorized_keys` between pod restarts unless you use a network volume.

---

## Step 4 — Test the SSH connection from terminal

```bash
ssh root@<POD_IP> -p <POD_PORT> -i ~/.ssh/id_ed25519
```

You should connect without being prompted for a password.

---

## Step 5 — Add the pod to your SSH config

Edit `~/.ssh/config` on your local machine and add:

```
Host runpod
    HostName <POD_IP>
    Port <POD_PORT>
    User root
    IdentityFile ~/.ssh/id_ed25519
```

Now you can connect simply with:

```bash
ssh runpod
```

---

## Step 6 — Connect via VSCode Remote SSH

1. Open VSCode
2. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
3. Select **Remote-SSH: Connect to Host**
4. Choose `runpod` (or type `root@<POD_IP>:<POD_PORT>`)
5. VSCode will open a new window connected to the pod

You can now browse files, open terminals, and run extensions directly on the pod.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Permission denied (publickey)` | Re-run Step 3 — the pod was likely restarted and lost `authorized_keys` |
| VSCode hangs on "Opening Remote" | Make sure you're using the TCP connection (not ssh.runpod.io) |
| Port not available | Check the pod is running and the port is listed under "Direct TCP ports" in RunPod dashboard |
| Key not found | Verify `~/.ssh/id_ed25519` exists locally; regenerate if needed (Step 2) |

---

## Persisting authorized_keys across pod restarts (optional)

Mount a RunPod **Network Volume** and symlink `authorized_keys` to it:

```bash
ln -sf /workspace/.ssh/authorized_keys ~/.ssh/authorized_keys
```

Store your public key in `/workspace/.ssh/authorized_keys` once, and it will persist across restarts as long as the volume is attached.
