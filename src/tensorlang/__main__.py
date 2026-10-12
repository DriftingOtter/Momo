"""Allow ``python -m tensorlang run file.tl``."""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
