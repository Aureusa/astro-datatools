import logging
from pathlib import Path

def setup_logging(
    level=logging.INFO,
    log_file=None,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
):
    """
    Configure logging for astro_datatools.
    
    :param level: Logging level (default: INFO)
    :param log_file: Path to log file. If None, only logs to console.
    :param format: Log message format
    """
    handlers = [logging.StreamHandler()]  # Console output
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format,
        handlers=handlers
    )