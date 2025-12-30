import time
from pipeline_orchestrator import ModuleContext
import logging


# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HelloWorldModule:
    def __init__(self):
        pass

    def run(self, context: ModuleContext):
        module_name = context.module_name
        logger.info(f"Hello from {module_name} module!")

        for i in range(3):
            logger.info(f"Debug message from {module_name} module! {i}")
            time.sleep(1)
        
        return f"Hello from {module_name} module!"