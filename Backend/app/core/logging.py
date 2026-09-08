import logging
import sys


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s trace_id=%(trace_id)s %(message)s",
        stream=sys.stdout,
    )
