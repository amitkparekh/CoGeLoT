from cogelot.common.log import setup_logging
from cogelot.entrypoints.__main__ import app

if __name__ == "__main__":
    setup_logging()
    app()
