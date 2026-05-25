import warnings
from fairpath.utils.warnings_config import configure_warnings
configure_warnings()

from fairpath.core.context import ProjectContext
from fairpath.core.workflow import WorkflowController
from fairpath.ui.terminal import TerminalUI
from fairpath.utils.logging import setup_logging

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
