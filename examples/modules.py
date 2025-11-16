from pipeline_orchestrator import ModuleContext


class HelloWorldModule:
    def __init__(self):
        pass

    def run(self, context: ModuleContext):
        print(context)
        print("Hello from hello_world module!")
        return "Hello from hello_world module!"