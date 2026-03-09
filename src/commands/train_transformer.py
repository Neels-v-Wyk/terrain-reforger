"""
Training script for terrain transformer (now uses MaskGIT).

This is a compatibility wrapper - the actual implementation is in train_maskgit.py.
"""

from src.commands.train_maskgit import run, main, _build_parser

__all__ = ['run', 'main', '_build_parser']

if __name__ == "__main__":
    main()
