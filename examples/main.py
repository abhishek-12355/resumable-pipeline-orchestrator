import argparse
import logging
from pipeline_orchestrator import PipelineOrchestrator

logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sample pipeline.")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart the pipeline and re-execute all modules regardless of previous state.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Hello from sample-pipeline!")
    # Initialize orchestrator with pipeline configuration
    orchestrator = PipelineOrchestrator(config_path="examples/pipeline.yaml")

    # Execute pipeline
    if args.restart:
        logging.info("CLI flag --restart set: forcing full pipeline restart")
        results = orchestrator.execute(force_restart=True)
    else:
        results = orchestrator.execute(force_restart=False)

    # Access results
    # for module_name, result in results.items():
    #     if isinstance(result, Exception):
    #         print(f"Module {module_name} failed: {result}")
    #     else:
    #         print(f"Module {module_name} succeeded: {result}")


if __name__ == "__main__":
    main()
