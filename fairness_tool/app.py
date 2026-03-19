import warnings
# Suppress RuntimeWarning from aif360 when subgroups are too small for certain metrics
warnings.filterwarnings("ignore", category=RuntimeWarning, module="aif360.metrics.classification_metric")

from core.context import ProjectContext
from core.workflow import WorkflowController
from ui.terminal import TerminalUI
from utils.logging import setup_logging

class FairnessApp:
    def __init__(self):
        setup_logging()
        self.context = ProjectContext()
        self.ui = TerminalUI()
        self.controller = WorkflowController(self.ui, self.context)

    def run(self):
        self.controller.run()

if __name__ == "__main__":
    app = FairnessApp()
    app.run()
