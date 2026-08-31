#!/usr/bin/env python3

from __future__ import annotations

import argparse

import carla


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    print(f"CARLA OK: map={world.get_map().name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
