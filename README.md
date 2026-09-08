# RunPod Volume — Local Access

Browse and edit the shared RunPod network volume directly from VS Code /
Finder on your Mac, without spinning up a pod. Mounts as a real read-write
folder. Nothing from the volume is downloaded or cached — files stream on
demand, so this works fine even though the volume is hundreds of GB.

Also includes a CLI for one-off uploads/downloads and a `sync` workflow for
pulling a specific project folder locally to work on offline.

---

## 1. One-time setup

```bash
git clone <this-repo>
cd Innovation-lab
pip install -r requirements.txt
cp .env.example .env
```

### Get your RunPod S3 credentials

The volume is accessed through RunPod's S3-compatible API. **This needs a
separate S3 access key — your normal RunPod API key (`rpa_...`) will NOT
work for this.**

1. Go to the RunPod console → **Storage → Network Volumes**
2. Click on the team's volume
3. Find the **S3 Access** / **S3 Credentials** section
4. Copy the **Access Key ID** (looks like `user_xxxxxxxxxxxxxxxxxxxx`) and
   **Secret Access Key** (looks like `rps_xxxxxxxxxxxxxxxxxxxx`)

### Fill in `.env`

```bash
RUNPOD_API_KEY=user_xxxxxxxxxxxxxxxxxxxx       # the S3 Access Key ID
RUNPOD_SECRET_KEY=rps_xxxxxxxxxxxxxxxxxxxx      # the S3 Secret Access Key
RUNPOD_BUCKET=xxxxxxxxxx                        # the Network Volume ID
RUNPOD_REGION=eu-ro-1                           # region shown next to the volume
```

`.env` is gitignored — never commit it.

### Verify it works

```bash
python -m runpod_volume.cli info
python -m runpod_volume.cli ls
```

If this lists files from the volume, credentials are correct.

---

## 2. Mounting the volume (VS Code / Finder access)

```bash
bash mount.sh
```

This starts a small local server and mounts the volume at `mnt/runpod/`
using macOS's **built-in** WebDAV client — no macFUSE, no kernel extension,
no reboot required.

```bash
code mnt/runpod         # open in VS Code
open mnt/runpod          # open in Finder
```

You can browse folders, open files, edit and save — changes write straight
back to the volume. Nothing is cached locally; opening a file streams it
from S3, closing it doesn't leave a copy on your Mac.

When done:

```bash
bash unmount.sh
```

**⚠️ Shared team volume — be careful with deletes.** This mount is
read-write. Deleting a file or folder in Finder/VS Code deletes it from the
volume for everyone. There's no trash/undo. Don't run automated cleanup
scripts against the mount unless you're sure of the path.

### If mounting fails / shows nothing

- Make sure `bash mount.sh` printed "Volume mounted at: ..." without errors
- Check `.env` has the **S3 access key**, not the general RunPod API key
- Run `bash unmount.sh` then `bash mount.sh` again
- Folder listings can take a few seconds the first time you open them
  (subsequent opens within ~20s are cached and instant)

---

## 3. CLI — quick one-off operations

For when you don't want to mount, just need one file:

```bash
python -m runpod_volume.cli ls [PREFIX]              # list files
python -m runpod_volume.cli get <REMOTE_KEY> <LOCAL>  # download one file
python -m runpod_volume.cli put <LOCAL> <REMOTE_KEY>  # upload one file
python -m runpod_volume.cli rm  <REMOTE_KEY>          # delete (asks to confirm)
python -m runpod_volume.cli url <REMOTE_KEY>          # pre-signed download link
```

---

## 4. Syncing a whole project folder locally

If you want a full local copy of one project (e.g. to train/run code with
it, or because the mount feels slow for very large/deep folders — like a
git repo with thousands of small files), use sync instead of the mount:

```bash
python -m runpod_volume.cli sync-pull YOLO-MIT/runs   # downloads to workspace/YOLO-MIT/runs
# ... work on it locally ...
python -m runpod_volume.cli sync-push YOLO-MIT/runs   # uploads only changed files back
```

Sync only transfers files that changed (by size), so re-running it is fast.
Be mindful of folder size before syncing — check first with:

```bash
python -m runpod_volume.cli ls YOLO-MIT/runs
```

---

## How it works (for the curious / troubleshooting)

- `runpod_volume/client.py` — boto3 S3 client wrapper, talks to RunPod's
  S3-compatible API (`https://s3api-<region>.runpod.io`)
- `runpod_volume/webdav_provider.py` — translates WebDAV requests
  (PROPFIND/GET/PUT/DELETE) into S3 calls, read-write, with a short
  in-memory cache for directory listings
- `serve.py` — runs the WebDAV server locally
- `mount.sh` / `unmount.sh` — start the server and mount it via macOS's
  built-in `mount_webdav`

**Known RunPod S3 quirk handled internally:** RunPod's S3 endpoint ignores
the `Prefix` parameter unless `Delimiter` is also set, and listing
`Prefix=<file>/` on an existing file incorrectly echoes back a fake
directory entry. The provider works around both — you don't need to know
this to use the tool, but it explains some of the code if you're reading it.
