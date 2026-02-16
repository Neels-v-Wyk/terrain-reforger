#!/usr/bin/env python3
"""Prepare chunked per-world dataset files from Terraria worlds."""

from src.pipelines.prepare_dataset_cli import main as prepare_main


def main() -> None:
    prepare_main(fixed_mode="chunked")

if __name__ == "__main__":
    main()
