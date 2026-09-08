#!/usr/bin/env python3
"""
RunPod volume CLI  (python -m runpod_volume.cli <command> ...)

Commands:
  ls [PREFIX]                          List objects under a prefix
  info                                 Show connection info

  sync-pull <REMOTE_FOLDER> [--dest DIR]   Download folder to local workspace
  sync-push <REMOTE_FOLDER> [--src  DIR]   Upload local changes back to volume

  put  <LOCAL_PATH>  <REMOTE_KEY>      Upload a single file
  get  <REMOTE_KEY>  <LOCAL_PATH>      Download a single file
  rm   <REMOTE_KEY>                    Delete an object
  url  <REMOTE_KEY>  [--expires 3600]  Generate a pre-signed URL
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runpod_volume.client import RunpodVolume

WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"


def _vol() -> RunpodVolume:
    return RunpodVolume()


def cmd_info(args):
    v = _vol()
    print(f"Endpoint : {v.endpoint}")
    print(f"Bucket   : {v.bucket}")
    print(f"Region   : {v.region}")
    print(f"API key  : {v.api_key[:8]}{'*' * max(0, len(v.api_key) - 8)}")


def cmd_ls(args):
    v = _vol()
    total = 0
    for obj in v.list(args.prefix or ""):
        size_kb = obj["Size"] / 1024
        print(f"  {obj['Key']:<60}  {size_kb:>10.1f} KB")
        total += 1
    print(f"\n{total} object(s)")


def cmd_sync_pull(args):
    v = _vol()
    dest = Path(args.dest) if args.dest else WORKSPACE / args.remote_folder
    dest.mkdir(parents=True, exist_ok=True)
    downloaded, skipped = v.sync_pull(args.remote_folder, dest)
    print(f"\nDone — {downloaded} downloaded, {skipped} already up-to-date.")
    print(f"Local path: {dest}")


def cmd_sync_push(args):
    v = _vol()
    src = Path(args.src) if args.src else WORKSPACE / args.remote_folder
    if not src.exists():
        print(f"Error: local directory not found: {src}", file=sys.stderr)
        print(f"Hint: run 'sync-pull {args.remote_folder}' first.", file=sys.stderr)
        sys.exit(1)
    uploaded, skipped = v.sync_push(src, args.remote_folder)
    print(f"\nDone — {uploaded} uploaded, {skipped} already up-to-date.")


def cmd_put(args):
    v = _vol()
    v.upload_file(args.local_path, args.remote_key)
    print(f"\nDone — {args.local_path} → {args.remote_key}")


def cmd_get(args):
    v = _vol()
    v.download_file(args.remote_key, args.local_path)
    print(f"\nDone — {args.remote_key} → {args.local_path}")


def cmd_rm(args):
    answer = input(f"Delete '{args.remote_key}' from volume? [y/N] ")
    if answer.strip().lower() != "y":
        print("Cancelled.")
        return
    v = _vol()
    v.delete(args.remote_key)
    print("Deleted.")


def cmd_url(args):
    v = _vol()
    url = v.generate_url(args.remote_key, expires_in=args.expires)
    print(url)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="runpod-vol", description="RunPod network volume CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info")

    ls = sub.add_parser("ls")
    ls.add_argument("prefix", nargs="?", default="")

    sp = sub.add_parser("sync-pull", help="Download a volume folder to local workspace")
    sp.add_argument("remote_folder", help="Folder name on the volume (e.g. YOLO-MIT)")
    sp.add_argument("--dest", default=None, help="Local destination (default: workspace/<folder>)")

    push = sub.add_parser("sync-push", help="Upload local workspace changes back to volume")
    push.add_argument("remote_folder", help="Folder name on the volume (e.g. YOLO-MIT)")
    push.add_argument("--src", default=None, help="Local source (default: workspace/<folder>)")

    put = sub.add_parser("put")
    put.add_argument("local_path")
    put.add_argument("remote_key")

    get = sub.add_parser("get")
    get.add_argument("remote_key")
    get.add_argument("local_path")

    rm = sub.add_parser("rm")
    rm.add_argument("remote_key")

    url = sub.add_parser("url")
    url.add_argument("remote_key")
    url.add_argument("--expires", type=int, default=3600)

    return p


COMMANDS = {
    "info": cmd_info,
    "ls": cmd_ls,
    "sync-pull": cmd_sync_pull,
    "sync-push": cmd_sync_push,
    "put": cmd_put,
    "get": cmd_get,
    "rm": cmd_rm,
    "url": cmd_url,
}


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        COMMANDS[args.cmd](args)
    except (ValueError, FileNotFoundError, NotADirectoryError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
