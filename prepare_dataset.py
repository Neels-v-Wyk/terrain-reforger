#!/usr/bin/env python3
"""Prepare a consolidated dataset from multiple Terraria worlds."""

from src.pipelines.prepare_dataset_cli import main as prepare_main


def main() -> None:
    prepare_main(fixed_mode="consolidated")

if __name__ == "__main__":
    main()
