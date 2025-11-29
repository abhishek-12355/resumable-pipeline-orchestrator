import logging
from pipeline_orchestrator import PipelineOrchestrator

logging.basicConfig(level=logging.DEBUG)


def main():
    print("Hello from sample-pipeline!")
    # Initialize orchestrator with pipeline configuration
    orchestrator = PipelineOrchestrator(config_path="examples/pipeline.yaml")

    # Execute pipeline
    results = orchestrator.execute()

    # Access results
    # for module_name, result in results.items():
    #     if isinstance(result, Exception):
    #         print(f"Module {module_name} failed: {result}")
    #     else:
    #         print(f"Module {module_name} succeeded: {result}")


if __name__ == "__main__":
    main()
