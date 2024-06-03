import logging
import numpy as np
import pandas as pd
import subprocess
import concurrent.futures
import gc
import random
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend to avoid issues with Tkinter
import matplotlib.pyplot as plt
import argparse
import psutil
import torch
import platform
import os
from time import perf_counter, strftime
import multiprocessing
import importlib.util

# Create necessary directories if they do not exist
os.makedirs('images', exist_ok=True)
os.makedirs('reports', exist_ok=True)

def get_ram_speed():
    try:
        result = subprocess.run(['sudo', 'dmidecode', '--type', '17'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        speeds = [line.split(': ')[1] for line in lines if 'Speed' in line and 'Configured Clock Speed' not in line]
        if speeds:
            return speeds[0]
    except Exception as e:
        logging.warning(f"Could not get RAM speed: {e}")
    return "Unknown"

def get_gpu_info():
    if torch.cuda.is_available():
        gpus = []
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpus.append(gpu_name)
        return ', '.join(gpus)
    else:
        return "No GPU found"

def get_system_info():
    system_info = {
        "System": platform.system(),
        "Node Name": platform.node(),
        "Release": platform.release(),
        "Version": platform.version(),
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "CPU Cores": psutil.cpu_count(logical=False),
        "Logical CPUs": psutil.cpu_count(logical=True),
        "Total Memory (GB)": psutil.virtual_memory().total / (1024 ** 3),
        "Available Memory (GB)": psutil.virtual_memory().available / (1024 ** 3),
        "Memory Usage (%)": psutil.virtual_memory().percent,
        "Disk Total (GB)": psutil.disk_usage('/').total / (1024 ** 3),
        "Disk Used (GB)": psutil.disk_usage('/').used / (1024 ** 3),
        "Disk Free (GB)": psutil.disk_usage('/').free / (1024 ** 3),
        "Disk Usage (%)": psutil.disk_usage('/').percent,
        "RAM Speed": get_ram_speed(),
        "GPU": get_gpu_info()
    }
    return system_info

def print_system_info():
    system_info = get_system_info()
    for key, value in system_info.items():
        print(f"{key}: {value}")

def generate_test_data(num_transactions):
    # Generate random transaction data and user profiles
    transaction_data = [
        (i, random.randint(1, 1000000), f"2024-05-31T10:{i % 60:02d}:00Z", f"M{i % 10}")
        for i in range(1, num_transactions + 1)
    ]
    user_profiles = [
        (i, [random.randint(1, 1000000) for _ in range(random.randint(0, 10))])
        for i in range(1, num_transactions + 1)
    ]
    return transaction_data, user_profiles

def load_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_ecs_test(transaction_data, user_profiles):
    try:
        ecs_solution = load_module('ecs_solution', 'modules_in_test/ecs_solution.py').ecs_solution
        logging.debug("Starting ECS Solution test")
        result = ecs_solution(transaction_data, user_profiles)
        logging.debug(f"ECS Solution result: {result}")
        assert result is not None, "ECS Solution returned None"
        return result
    except Exception as e:
        logging.error(f"Error in ECS Solution test: {e}")
        return None

def run_functional_test(transaction_data, user_profiles):
    try:
        functional_solution = load_module('functional_solution', 'modules_in_test/functional_solution.py').functional_solution
        logging.debug("Starting Functional Solution test")
        result = functional_solution(transaction_data, user_profiles)
        logging.debug(f"Functional Solution result: {result}")
        assert result is not None, "Functional Solution returned None"
        return result
    except Exception as e:
        logging.error(f"Error in Functional Solution test: {e}")
        return None

def run_visitordom_test(transaction_data, user_profiles):
    try:
        VisitorDom = load_module('visitor_dom_solution', 'modules_in_test/visitor_dom_solution.py').VisitorDom
        logging.debug("Starting VisitorDom Solution test")
        visitor_dom = VisitorDom(transaction_data, user_profiles)
        result = visitor_dom.combined_solution()
        logging.debug(f"VisitorDom Solution result: {result}")
        assert result is not None, "VisitorDom Solution returned None"
        return result
    except Exception as e:
        logging.error(f"Error in VisitorDom Solution test: {e}")
        return None

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
        return (result, memory_usage, end_time - start_time)
    except Exception as e:
        logging.error(f'Error in process {process_name}: {e}')
        return (None, None, None)
    finally:
        clear_cache()

def clear_cache():
    # Clear cache memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_radon_analysis(file_path):
    logging.info(f'Running radon analysis for {file_path}')
    cc_output = subprocess.run(['radon', 'cc', file_path], capture_output=True, text=True).stdout
    cc_output_avg = subprocess.run(['radon', 'cc', '-a', file_path], capture_output=True, text=True).stdout
    logging.info(f'Cyclomatic Complexity for {file_path}:\n{cc_output}')
    logging.info(f'Average Cyclomatic Complexity for {file_path}:\n{cc_output_avg}')
    return cc_output, cc_output_avg

def configure_logging(verbose):
    logging_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=logging_level, format='%(asctime)s %(levelname)s %(message)s')

def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    logging.info(f"Memory Usage: RSS={memory_info.rss / (1024 ** 2):.2f}MB, VMS={memory_info.vms / (1024 ** 2):.2f}MB")

def main(repetitions=5, verbose=True, test_data_sizes=[1000, 10000, 100000], memory_limit=40000, max_workers=4):
    system_info = get_system_info()
    configure_logging(verbose)

    all_results_agg = []
    memory_usages_agg = []
    complexity_results = {}

    # Run radon analysis once for each solution
    solutions = [
        ("ECS Solution", 'modules_in_test/ecs_solution.py'),
        ("Functional Solution", 'modules_in_test/functional_solution.py'),
        ("VisitorDom Solution", 'modules_in_test/visitor_dom_solution.py')
    ]

    timestamp = strftime("%Y%m%d-%H%M%S")
    
    for label, file_path in solutions:
        complexity_results[label] = run_radon_analysis(file_path)

    for size in test_data_sizes:
        logging.info(f'Running tests for data size: {size}')
        log_memory_usage()
        transaction_data, user_profiles = generate_test_data(size)

        tests = [
            (run_ecs_test, transaction_data, user_profiles, "ECS Solution"),
            (run_functional_test, transaction_data, user_profiles, "Functional Solution"),
            (run_visitordom_test, transaction_data, user_profiles, "VisitorDom Solution")
        ]

        all_results = {label: [] for _, _, _, label in tests}
        memory_usages = {label: [] for _, _, _, label in tests}

        for _ in range(repetitions):
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for test_function, t_data, u_profiles, label in tests:
                    future = executor.submit(run_test_in_process, test_function, t_data, u_profiles, verbose, memory_limit)
                    futures[future] = label

                for future in concurrent.futures.as_completed(futures):
                    label = futures[future]
                    try:
                        result, memory_usage, exec_time = future.result()
                        if result is not None and memory_usage is not None:
                            all_results[label].append(exec_time)
                            memory_usages[label].append(memory_usage)
                            if verbose:
                                logging.debug(f"Test {label}: {result} - Execution Time: {exec_time}s, Memory Usage: {memory_usage}MB")
                        else:
                            logging.error(f"Error in test {label}: Result is None or memory usage is None")
                            all_results[label].append(float('inf'))
                            memory_usages[label].append(float('inf'))
                    except Exception as e:
                        logging.error(f'Error while processing future for {label}: {e}')
                        all_results[label].append(float('inf'))
                        memory_usages[label].append(float('inf'))

        avg_results = {label: np.mean(times) if len(times) > 0 and not np.isnan(times).all() else float('nan') for label, times in all_results.items()}
        avg_memory_usages = {label: np.mean(usages) if len(usages) > 0 and not np.isnan(usages).all() else float('nan') for label, usages in memory_usages.items()}

        df_results = pd.DataFrame.from_dict(avg_results, orient='index', columns=[f'Average Execution Time (s) - {size}'])
        df_memory = pd.DataFrame.from_dict(avg_memory_usages, orient='index', columns=[f'Average Memory Usage (MB) - {size}'])
        print(df_results)
        print(df_memory)

        all_results_agg.append((size, avg_results))
        memory_usages_agg.append((size, avg_memory_usages))

        # Save histograms
        labels, times = zip(*avg_results.items())
        plt.figure(figsize=(8, 5))
        plt.bar(labels, times, color=['blue', 'green', 'red'])
        plt.xlabel('Solutions')
        plt.ylabel('Average Execution Time (s)')
        plt.title(f'Comparison of Average Execution Time of Solutions for Data Size {size}')
        plt.savefig(f'images/histogram_{timestamp}_{size}.png')
        plt.close()

        labels, usages = zip(*avg_memory_usages.items())
        plt.figure(figsize=(8, 5))
        plt.bar(labels, usages, color=['blue', 'green', 'red'])
        plt.xlabel('Solutions')
        plt.ylabel('Average Memory Usage (MB)')
        plt.title(f'Comparison of Average Memory Usage of Solutions for Data Size {size}')
        plt.savefig(f'images/memory_usage_{timestamp}_{size}.png')
        plt.close()

    # Create bar chart for all tests (execution times)
    bar_width = 0.2
    positions = np.arange(len(all_results_agg[0][1]))
    plt.figure(figsize=(10, 6))

    for i, (size, results) in enumerate(all_results_agg):
        labels, times = zip(*results.items())
        plt.bar(positions + i * bar_width, times, width=bar_width, label=f'Data Size {size}')

    plt.xlabel('Solutions')
    plt.ylabel('Average Execution Time (s)')
    plt.title('Comparison of Average Execution Time of Solutions for Different Data Sizes')
    plt.xticks(positions + bar_width * (len(all_results_agg) - 1) / 2, labels)
    plt.legend()
    plt.savefig('images/histogram_all_{timestamp}.png')
    plt.close()

    # Create bar chart for all tests (memory usage)
    positions = np.arange(len(memory_usages_agg[0][1]))
    plt.figure(figsize=(10, 6))

    for i, (size, results) in enumerate(memory_usages_agg):
        labels, usages = zip(*results.items())
        plt.bar(positions + i * bar_width, usages, width=bar_width, label=f'Data Size {size}')

    plt.xlabel('Solutions')
    plt.ylabel('Average Memory Usage (MB)')
    plt.title('Comparison of Average Memory Usage of Solutions for Different Data Sizes')
    plt.xticks(positions + bar_width * (len(memory_usages_agg) - 1) / 2, labels)
    plt.legend()
    plt.savefig('images/memory_usage_all_{timestamp}.png')
    plt.close()

    # Save results and system info to a text file with timestamp
    report_filename = f'reports/test_results_{timestamp}.txt'
    with open(report_filename, 'w') as f:
        f.write("System Information:\n")
        for key, value in system_info.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write("Test Parameters:\n")
        f.write(f"Repetitions: {repetitions}\n")
        f.write(f"Test Data Sizes: {test_data_sizes}\n")
        f.write(f"Memory Limit (MB): {memory_limit}\n")
        f.write(f"Max Workers: {max_workers}\n")
        f.write("\n")

        for label, (cc_output, cc_output_avg) in complexity_results.items():
            f.write(f"Complexity results for {label}:\n")
            f.write(f"Cyclomatic Complexity:\n{cc_output}\n")
            f.write(f"Average Cyclomatic Complexity:\n{cc_output_avg}\n")
            f.write("\n")

        f.write("Average Execution Time (s) for each data size:\n")
        for size, results in all_results_agg:
            f.write(f"Data Size {size}:\n")
            for label, time in results.items():
                f.write(f"{label}: {time}\n")
            f.write("\n")

        f.write("Average Memory Usage (MB) for each data size:\n")
        for size, memory in memory_usages_agg:
            f.write(f"Data Size {size}:\n")
            for label, mem in memory.items():
                f.write(f"{label}: {mem}\n")
            f.write("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run performance tests on different solutions.')
    parser.add_argument('--repetitions', type=int, default=5, help='Number of repetitions for each test.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging.')
    parser.add_argument('--test_data_sizes', nargs='+', type=int, default=[1000, 10000, 100000], help='List of test data sizes.')
    parser.add_argument('--memory_limit', type=int, default=40000, help='Memory limit in MB.')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of concurrent workers.')

    args = parser.parse_args()
    main(repetitions=args.repetitions, verbose=args.verbose, test_data_sizes=args.test_data_sizes, memory_limit=args.memory_limit, max_workers=args.max_workers)
