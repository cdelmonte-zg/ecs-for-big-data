# Entity Component System for Big Data - Performance Testing of Different Software Solutions


This project aims to evaluate and compare the performance of different software solutions using various performance metrics such as execution time and memory usage. The solutions are implemented using different design patterns and methodologies, and their performance is tested under varying data sizes.

## Solutions Tested
1. **ECS (Entity-Component-System) Solution**
2. **Functional Solution**
3. **Visitor + DOM (Document Object Model) Solution**

## Features
- Performance metrics: execution time and memory usage.
- Cyclomatic complexity analysis using Radon.
- Detailed reports and visualizations of the results.
- Automatic cache clearing after each test to ensure clean testing conditions.

## Prerequisites
Make sure you have the following libraries installed in your Python environment:
- numpy
- pandas
- matplotlib
- psutil
- torch
- radon

You can install them using `pip`:
```bash
pip install numpy pandas matplotlib psutil torch radon
```

## Directory Structure
- `modules_in_test/`: Contains the implementation files for the solutions to be tested (`ecs_solution.py`, `functional_solution.py`, `visitor_dom_solution.py`).
- `images/`: Directory where the generated images (histograms and tables) will be saved.
- `reports/`: Directory where the detailed reports will be saved.

## Running the Experiment
To run the experiment, use the following command:

```bash
python exec_experiment.py --repetitions <number_of_repetitions> --test_data_sizes <data_size1> <data_size2> <data_sizeN> --memory_limit <memory_limit_in_MB> --max_workers <number_of_workers> --verbose
```

### Arguments
- `--repetitions`: Number of repetitions for each test (default: 5).
- `--test_data_sizes`: List of test data sizes (default: [1000, 10000, 100000]).
- `--memory_limit`: Memory limit in MB (default: 40000).
- `--max_workers`: Maximum number of concurrent workers (default: 4).
- `--verbose`: Enable verbose logging.

### Example
```bash
python exec_experiment.py --repetitions 5 --test_data_sizes 1000 10000 100000 --memory_limit 40000 --max_workers 4 --verbose
```

## Output
The experiment generates the following outputs:
1. **Images**:
   - Histograms and tables showing the average execution time and memory usage for each solution at different data sizes.
   - Saved in the `images/` directory.

2. **Reports**:
   - A detailed text file containing system information, test parameters, cyclomatic complexity analysis, and the results of the tests.
   - Saved in the `reports/` directory with a timestamp in the filename.

## System Information
The script also logs detailed system information, including:
- System details (OS, node name, release, version, machine type, processor).
- CPU details (number of cores and logical CPUs).
- Memory details (total and available memory, memory usage percentage).
- Disk details (total, used, and free disk space, disk usage percentage).
- RAM speed.
- GPU information.

## Cyclomatic Complexity Analysis
The script uses Radon to perform a cyclomatic complexity analysis on the solution files. The results are included in the final report.

## Clear Cache Function
To ensure clean testing conditions, the script includes a `clear_cache` function that clears the memory cache after each test run.

## Test Framework Architecture
The test framework is architected to ensure robust and efficient testing of multiple solutions under varying conditions. Here's an overview of its architecture:

### Main Components
1. **Data Generation**: Generates random transaction data and user profiles for testing.
2. **Module Loading**: Dynamically loads the solution modules to be tested.
3. **Test Execution**: Uses `concurrent.futures.ProcessPoolExecutor` to run tests in parallel, leveraging multiple CPU cores for efficiency.
4. **Result Collection**: Collects and aggregates results (execution time and memory usage) from each test run.
5. **Cache Clearing**: Ensures that memory cache is cleared after each test run to prevent cross-test contamination.
6. **Cyclomatic Complexity Analysis**: Analyzes the complexity of each solution using Radon before the tests begin.
7. **Reporting**: Generates and saves detailed reports and visualizations of the test results.

### Workflow
1. **Initialization**: Set up logging, create necessary directories, and print system information.
2. **Cyclomatic Complexity Analysis**: Run Radon analysis on the solution files and store the results.
3. **Test Execution**: For each data size:
   - Generate test data.
   - Execute tests for each solution using a process pool.
   - Collect and aggregate results.
   - Clear cache after each test.
4. **Result Aggregation**: Compute average execution time and memory usage for each solution and data size.
5. **Visualization**: Generate histograms and tables for the results.
6. **Reporting**: Save a detailed report with system information, test parameters, complexity analysis, and results.

### Clear Cache Function
The `clear_cache` function is essential for ensuring that each test run starts with a clean slate:
```python
def clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### Example Functions
Here’s an example of how the `run_test_in_process` function works:
```python
def run_test_in_process(test_function, transaction_data, user_profiles, verbose, memory_limit):
    process_name = multiprocessing.current_process().name
    if verbose:
        logging.debug(f'Starting process {process_name}')
    try:
        start_time = perf_counter()
        result = test_function(transaction_data, user_profiles)
        end_time = perf_counter()
        memory_usage = psutil.Process().memory_info().rss / (1024 ** 2)  # In MB
        if memory_usage > memory_limit:
            logging.warning(f'Memory usage exceeded: {memory_usage}MB > {memory_limit}MB in process {process_name}')
        if verbose:
            logging.debug(f'Completed process {process_name} with result: {result}, Memory Usage: {memory_usage}MB, Time: {end_time - start_time}s')
        return (result, memory_usage, start_time, end_time)
    except Exception as e:
        logging.error(f'Error in process {process_name}: {e}')
        return (None, None, None, None)
    finally:
        clear_cache()
```

## Contributing
Feel free to contribute to this project by adding new solutions or improving the existing ones. Make sure to update the README and provide clear documentation for any changes made.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
