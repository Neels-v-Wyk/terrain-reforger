"""
Generate terrain using transformer (now uses MaskGIT).

This is a compatibility wrapper - the actual implementation is in generate_maskgit.py.
"""

from src.commands.generate_maskgit import run, main, _build_parser

__all__ = ['run', 'main', '_build_parser']

if __name__ == "__main__":
    main()
