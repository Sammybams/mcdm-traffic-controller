#!/usr/bin/env python3
"""Fine-tune a pretrained detector from a versioned experiment config."""

from __future__ import annotations

import argparse

from traffic_vision.training import load_training_config, run_training


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="training experiment JSON file")
    args = parser.parse_args()

    config = load_training_config(args.config)
    run_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

