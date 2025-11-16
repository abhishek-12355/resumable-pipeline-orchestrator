# Pipeline Orchestrator

A Python library for YAML-driven pipeline orchestration with support for sequential/parallel execution, resource management (CPU/GPU), checkpointing, and nested parallel task execution.

## Features

- **YAML Configuration**: Define pipeline modules and dependencies in YAML
- **Flexible Execution**: Sequential or parallel execution modes
- **Resource Management**: Automatic CPU and GPU allocation and monitoring
- **Worker Types**: Support for threads or processes
- **Checkpointing**: Automatic state persistence for pipeline resumption
- **Nested Execution**: Modules can request parallel sub-task execution
- **Failure Policies**: Fail-fast or collect-all error handling
- **Resumption**: Automatically resume from checkpoints on restart

## Installation

### Basic Installation

```bash
pip install pipeline-orchestrator
```

### With GPU Support

```bash
pip install pipeline-orchestrator[gpu]
```

### From Local Checkpoint Manager

If using the local checkpoint-manager:

```bash
# Install checkpoint-manager first from local path
cd /path/to/checkpoint-manager
pip install -e .

# Then install pipeline-orchestrator
cd /path/to/pipeline-orchestrator
pip install -e .
```

## Quick Start

### 1. Create a YAML Configuration File

Create `pipeline.yaml`:

```yaml
pipeline:
  name: "example_pipeline"
  execution:
    mode: "parallel"  # parallel (default) or sequential
    worker_type: "process"  # thread or process
    failure_policy: "fail_fast"  # fail_fast or collect_all
  resources:
    max_cpus: 4
    max_gpus: 2
  checkpoint:
    enabled: true
    directory: "./checkpoints"
  modules:
    - name: "module1"
      path: "mymodules.module1:MyModule"  # Class with run() method
      depends_on: []
      resources:
        cpus: 1
        gpus: 0
    - name: "module2"
      path: "mymodules.module2:MyModule"
      depends_on: ["module1"]
      resources:
        cpus: 2
        gpus: 1
```

### 2. Create Modules

Create your modules as Python classes:

```python
from pipeline_orchestrator import BaseModule, ModuleContext

class MyModule(BaseModule):
    def run(self, context: ModuleContext):
        # Access dependency results
        data = context.get_result("module1")
        
        # Process data
        result = process_data(data)
        
        # Optionally request nested parallel execution
        sub_tasks = [task1, task2, task3]
        sub_results = context.execute_tasks(sub_tasks)
        
        return result
```

### 3. Run the Pipeline

```python
from pipeline_orchestrator import PipelineOrchestrator

# Initialize orchestrator
orchestrator = PipelineOrchestrator(config_path="pipeline.yaml")

# Execute pipeline
results = orchestrator.execute()

# Access results
for module_name, result in results.items():
    if isinstance(result, Exception):
        print(f"Module {module_name} failed: {result}")
    else:
        print(f"Module {module_name} succeeded: {result}")
```

## YAML Configuration

### Pipeline Structure

```yaml
pipeline:
  name: "pipeline_name"
  execution:
    mode: "parallel"  # "parallel" (default) or "sequential"
    worker_type: "process"  # "thread" or "process"
    failure_policy: "fail_fast"  # "fail_fast" or "collect_all"
    max_nested_depth: 3  # Optional: limit nested execution depth
  resources:
    max_cpus: 4  # Optional: defaults to system detection
    max_gpus: 2  # Optional: defaults to system detection
  checkpoint:
    enabled: true  # Enable checkpointing
    directory: "./checkpoints"  # Checkpoint directory
  modules:
    - name: "module_name"
      path: "module.path:ClassName"  # Class-based module
      # OR
      script: "path/to/script.py:function_name"  # Function-based module
      depends_on: ["dependency1", "dependency2"]
      resources:
        cpus: 1
        gpus: 0
```

### Module Configuration

**Class-based Module:**
```yaml
- name: "module1"
  path: "mymodules.module1:MyModule"
  depends_on: []
  resources:
    cpus: 1
    gpus: 0
```

**Function-based Module:**
```yaml
- name: "module1"
  script: "path/to/script.py:my_function"
  depends_on: []
  resources:
    cpus: 1
    gpus: 0
```

## Module Development

### BaseModule Interface

All modules must inherit from `BaseModule` and implement the `run()` method:

```python
from pipeline_orchestrator import BaseModule, ModuleContext

class MyModule(BaseModule):
    def run(self, context: ModuleContext):
        # Access dependency results
        data1 = context.get_result("dependency1")
        data2 = context.dependency_results.get("dependency2")
        
        # Get all dependency results
        all_deps = context.dependency_results
        
        # Process data
        result = process(data1, data2)
        
        # Return result (automatically checkpointed)
        return result
```

### Accessing Dependency Results

**Option 1: Get specific dependency**
```python
data = context.get_result("module1")  # Raises DependencyError if not found
```

**Option 2: Access via dictionary**
```python
data = context.dependency_results.get("module1")  # Returns None if not found
```

**Option 3: Iterate over dependencies**
```python
for module_name, result in context.dependency_results.items():
    process(result)
```

### Nested Parallel Execution

Modules can request parallel execution of sub-tasks:

```python
def run(self, context: ModuleContext):
    # Request nested parallel execution
    sub_tasks = [
        lambda: process_item(item1),
        lambda: process_item(item2),
        lambda: process_item(item3),
    ]
    
    # Execute tasks in parallel via orchestrator
    results = context.execute_tasks(sub_tasks)
    
    # Results are in same order as tasks
    # Each result can be success (data) or error (Exception)
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Task {i} failed: {result}")
        else:
            print(f"Task {i} succeeded: {result}")
    
    return combined_results
```

### Standalone Function Modules

You can also use standalone functions as modules:

```python
# script.py
def my_function(context):
    data = context.get_result("dependency1")
    return process(data)
```

YAML:
```yaml
- name: "module1"
  script: "script.py:my_function"
  depends_on: ["dependency1"]
```

## Resource Management

### CPU Allocation

Modules can specify CPU requirements:

```yaml
resources:
  cpus: 2  # Number of CPUs required
```

### GPU Allocation

Modules can specify GPU requirements:

```yaml
resources:
  gpus: 1  # Number of GPUs required
```

The orchestrator automatically:
- Detects available GPUs
- Assigns specific GPU devices to modules
- Sets `CUDA_VISIBLE_DEVICES` for process mode
- Tracks GPU usage across modules

### Resource Limits

Global resource limits can be set:

```yaml
resources:
  max_cpus: 8  # Optional: defaults to system detection
  max_gpus: 4  # Optional: defaults to system detection
```

## Checkpointing

### Automatic Checkpointing

Results are automatically checkpointed after each module completes:

```python
# Module result is automatically saved
result = orchestrator.execute()

# On restart, pipeline resumes from checkpoints
orchestrator = PipelineOrchestrator(config_path="pipeline.yaml")
results = orchestrator.execute()  # Completed modules are skipped
```

### Checkpoint Directory

Checkpoints are stored in the configured directory:

```yaml
checkpoint:
  enabled: true
  directory: "./checkpoints"  # Default: "./.checkpoints"
```

### Resumption

The pipeline automatically:
- Detects completed modules from checkpoints
- Loads results for dependent modules
- Skips execution of completed modules
- Continues with pending modules

## Execution Modes

### Sequential Mode

Execute modules one at a time:

```yaml
execution:
  mode: "sequential"
```

### Parallel Mode

Execute modules in parallel (when dependencies allow):

```yaml
execution:
  mode: "parallel"
  worker_type: "process"  # or "thread"
```

## Worker Types

### Thread Mode

Use threads for execution (shared memory):

```yaml
execution:
  worker_type: "thread"
```

**Pros:**
- Shared memory (faster communication)
- Lower overhead
- Simpler nested execution

**Cons:**
- GIL limitations (CPU-bound tasks)
- Less isolation

### Process Mode

Use processes for execution (isolated memory):

```yaml
execution:
  worker_type: "process"
```

**Pros:**
- True parallelism (CPU-bound tasks)
- Better isolation
- GPU management via CUDA_VISIBLE_DEVICES

**Cons:**
- Higher overhead
- Requires picklable modules/contexts
- IPC for nested execution

## Failure Policies

### Fail-Fast Policy

Stop execution on first failure:

```yaml
execution:
  failure_policy: "fail_fast"
```

### Collect-All Policy

Continue execution regardless of failures:

```yaml
execution:
  failure_policy: "collect_all"
```

All results (success or error) are collected and returned.

## API Reference

### PipelineOrchestrator

Main orchestrator class:

```python
from pipeline_orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator(config_path="pipeline.yaml")
results = orchestrator.execute()
```

**Methods:**
- `execute()`: Execute the pipeline
- `get_results()`: Get execution results
- `get_module_result(module_name)`: Get result for specific module
- `cleanup()`: Cleanup resources

### BaseModule

Base class for modules:

```python
from pipeline_orchestrator import BaseModule, ModuleContext

class MyModule(BaseModule):
    def run(self, context: ModuleContext):
        return result
```

### ModuleContext

Execution context provided to modules:

**Properties:**
- `module_name`: Name of the current module
- `pipeline_name`: Name of the pipeline
- `resources`: Resource allocation (cpus, gpus, gpu_ids)
- `dependency_results`: Dictionary of dependency results

**Methods:**
- `get_result(module_name)`: Get result from dependency
- `get_all_results()`: Get all completed module results
- `execute_tasks(tasks)`: Execute nested tasks in parallel

## Examples

### Example 1: Simple Sequential Pipeline

```yaml
pipeline:
  name: "simple_pipeline"
  execution:
    mode: "sequential"
  modules:
    - name: "data_loader"
      path: "modules.data:DataLoader"
      depends_on: []
    - name: "processor"
      path: "modules.process:Processor"
      depends_on: ["data_loader"]
    - name: "saver"
      path: "modules.save:Saver"
      depends_on: ["processor"]
```

### Example 2: Parallel Pipeline with GPU

```yaml
pipeline:
  name: "gpu_pipeline"
  execution:
    mode: "parallel"
    worker_type: "process"
  resources:
    max_gpus: 2
  modules:
    - name: "data_loader"
      path: "modules.data:DataLoader"
      depends_on: []
      resources:
        cpus: 1
        gpus: 0
    - name: "trainer1"
      path: "modules.train:Trainer"
      depends_on: ["data_loader"]
      resources:
        cpus: 2
        gpus: 1
    - name: "trainer2"
      path: "modules.train:Trainer"
      depends_on: ["data_loader"]
      resources:
        cpus: 2
        gpus: 1
```

### Example 3: Module with Nested Execution

```python
class DataProcessor(BaseModule):
    def run(self, context: ModuleContext):
        # Get input data
        data = context.get_result("data_loader")
        
        # Process items in parallel
        items = data.items
        tasks = [lambda item=item: process_item(item) for item in items]
        results = context.execute_tasks(tasks)
        
        # Combine results
        return combine_results(results)
```

## Dependencies

- `checkpoint-manager`: Checkpointing support
- `pyyaml`: YAML configuration parsing
- `psutil`: CPU detection and system information
- `pynvml` (optional): GPU detection

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

