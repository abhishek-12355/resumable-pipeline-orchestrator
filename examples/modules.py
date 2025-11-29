import time
from pipeline_orchestrator import ModuleContext
import logging


# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HelloWorldModule:
    def __init__(self):
        pass

    def run(self, context: ModuleContext):
        logger.info("Hello from hello_world module!")

        for i in range(5):
            logger.info(f"Debug message from hello_world module! {i}")
            time.sleep(3)
        
        return "Hello from hello_world module!"