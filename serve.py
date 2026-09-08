#!/usr/bin/env python3
"""
Start a read-write WebDAV server backed by the RunPod S3 volume.
Credentials are loaded from .env automatically.

Usage:
    python serve.py            # runs in foreground on port 8765
    python serve.py --port 9000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from runpod_volume.client import RunpodVolume
from runpod_volume.webdav_provider import S3Provider

from wsgidav.wsgidav_app import WsgiDAVApp
from cheroot import wsgi


def main():
    parser = argparse.ArgumentParser(description="Read-write WebDAV server for RunPod S3 volume")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    vol = RunpodVolume()
    provider = S3Provider(vol._client, vol.bucket)

    config = {
        "provider_mapping": {"/": provider},
        "http_authenticator": {"domain_controller": None},  # no auth (local only)
        "simple_dc": {"user_mapping": {"*": True}},
        "verbose": 0,
        "logging": {"enable_loggers": []},
        "property_manager": True,
        "lock_storage": True,  # macOS WebDAV client expects LOCK/UNLOCK to work
    }

    app = WsgiDAVApp(config)
    server = wsgi.Server((args.host, args.port), app)

    print(f"RunPod volume: {vol.bucket}  ({vol.endpoint})")
    print(f"WebDAV server: http://{args.host}:{args.port}  [READ-WRITE]")
    print()
    print(f"Mount command:   bash mount.sh")
    print(f"Stop server:     Ctrl-C")
    print()

    try:
        server.start()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.stop()


if __name__ == "__main__":
    main()
