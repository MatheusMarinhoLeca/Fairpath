import logging
import os

class WarningFilter(logging.Filter):
    def filter(self, record):
        # Ignore the specific Matplotlib categorical units message
        # It often comes as an INFO level log when using categorical units
        msg = record.getMessage()
        if "categorical units to plot a list of strings" in msg:
            return False
        return True

def setup_logging(log_dir="outputs/logs"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Reset handlers if they exist to avoid duplicate logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Capture standard warnings into the logging system
    logging.captureWarnings(True)

    logging.basicConfig(
        filename=os.path.join(log_dir, 'fairpath.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    
    # Add filter to root logger
    root_logger = logging.getLogger('')
    root_logger.addHandler(console)
    root_logger.addFilter(WarningFilter())

    # Explicitly silence matplotlib categorical unit logger which emits this at INFO level
    logging.getLogger('matplotlib.category').setLevel(logging.WARNING)

def log_action(message):
    logging.info(message)
