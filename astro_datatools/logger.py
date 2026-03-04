import logging
from pathlib import Path


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",      # Cyan
        logging.INFO: "\033[32m",       # Green
        logging.WARNING: "\033[33m",    # Yellow
        logging.ERROR: "\033[31m",      # Red
        logging.CRITICAL: "\033[1;31m", # Bold red
    }
    RESET = "\033[0m"

    def __init__(self, datefmt: str, use_color: bool = True):
        super().__init__(datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        logger_name = record.name
        level_name = record.levelname

        if record.levelno == logging.INFO:
            prefix = f"[{timestamp} {logger_name}]"
        else:
            prefix = f"[{timestamp} | {level_name} | {logger_name}]"

        if self.use_color:
            color = self.COLORS.get(record.levelno)
            if color:
                prefix = f"{color}{prefix}{self.RESET}"

        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        elif record.stack_info:
            message = f"{message}\n{self.formatStack(record.stack_info)}"

        return f"{prefix}: {message}"


def setup_logging(
    name: str = "astro_datatools",
    level=logging.INFO,
    log_file=None,
    datefmt: str = "%m/%d %H:%M:%S",
    use_colors: bool = True,
) -> logging.Logger:
    """
    Configure logging for astro_datatools.

    :param name: Logger name to return
    :param level: Logging level (default: INFO)
    :param log_file: Path to log file. If None, only logs to console.
    :param datefmt: Time format for logs
    :param use_colors: Whether to colorize console logs
    :return: Configured logger instance
    """
    console_handler = logging.StreamHandler()
    stream = getattr(console_handler, "stream", None)
    supports_color = bool(
        use_colors
        and stream is not None
        and hasattr(stream, "isatty")
        and stream.isatty()
    )

    console_handler.setFormatter(
        ColorFormatter(datefmt=datefmt, use_color=supports_color)
    )

    handlers = [console_handler]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(ColorFormatter(datefmt=datefmt, use_color=False))
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,
    )

    return logging.getLogger(name)