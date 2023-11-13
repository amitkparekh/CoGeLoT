import inspect
import logging

from loguru import logger


class InterceptHandler(logging.Handler):
    """Logger Handler to intercept logging and send it to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit method for logging."""
        # Get corresponding Loguru level if it exists.
        level: str | int

        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging() -> None:
    """Setup loguru logging for everything."""
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
