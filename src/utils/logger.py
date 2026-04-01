import logging
import sys
from pathlib import Path

def setup_logging(config: dict):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(config["logging"]["file"], encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )