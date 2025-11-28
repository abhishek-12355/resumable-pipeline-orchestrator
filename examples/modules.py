from pipeline_orchestrator import ModuleContext
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HelloWorldModule:
    def __init__(self):
        pass

    def run(self, context: ModuleContext):
        print(context)
        print("Hello from hello_world module!")
        logger.info("Hello from hello_world module!")
        return "Hello from hello_world module!"