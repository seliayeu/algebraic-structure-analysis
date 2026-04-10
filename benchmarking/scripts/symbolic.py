import subprocess
import os
import re
import csv
import statistics
import argparse

def generate_mlir(k, filename, mode="dia"):
    """Generates an MLIR file with a chain of k matmul operations (linalg or dia)."""
    with open(filename, 'w') as f:
        f.write("module {\n")
        f.write("  func.func @chained_matmul() -> f32 {\n")
        
        meta = '{metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}}'
        
        if mode == "dia":
            f.write("    %c0 = arith.constant 0 : index\n")

        f.write(f"    %input = arith.constant {meta} dense<1.0> : tensor<2048x2048xf32>\n")
        
        for i in range(1, k + 1):
            f.write(f"    %W{i} = arith.constant {meta} dense<1.0> : tensor<2048x2048xf32>\n")
            if mode == "dia":
                f.write(f"    %e{i} = tensor.empty(%c0) : tensor<?x2048xf32>\n")
            else:
                f.write(f"    %e{i} = tensor.empty() : tensor<2048x2048xf32>\n")
        
        f.write("\n")
        
        prev = "%input"
        prev_type = "tensor<2048x2048xf32>"
        out_type = "tensor<?x2048xf32>" if mode == "dia" else "tensor<2048x2048xf32>"
        op_name = "dia.matmul" if mode == "dia" else "linalg.matmul"

        for i in range(1, k + 1):
            f.write(f"    %m{i} = {op_name} ins({prev}, %W{i} : {prev_type}, tensor<2048x2048xf32>) outs(%e{i} : {out_type}) -> {out_type}\n")
            prev = f"%m{i}"
            prev_type = out_type
            
        f.write("\n")
        
        f.write("    %index = arith.constant 5 : index\n")
        f.write(f"    %result = tensor.extract {prev}[%index, %index] : {prev_type}\n")
        f.write("    return %result : f32\n")
        f.write("  }\n")
        f.write("}\n")

def run_benchmark():
    ks = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    REPETITIONS = 3
    modes = ["linalg", "dia"]
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("benchmarking/programs", exist_ok=True)
    
    for mode in modes:
        csv_filename = f"results/symbolic_chained_{mode}.csv"
        print(f"\nRunning benchmark for mode: {mode.upper()}")
        print(f"{'K':<6} | {'Avg Time (ms)':<15} | {'Trials (ms)':<30}")
        print("-" * 60)
        
        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['k', 'avg_time_ms', 'trial_1', 'trial_2', 'trial_3'])
            
            for k in ks:
                filename = f"benchmarking/programs/chained_{k}_{mode}.mlir"
                generate_mlir(k, filename, mode)
                
                trials = []
                for i in range(REPETITIONS):
                    cmd = ["build/tools/alg-opt", filename, "--banded-analysis"]
                    
                    try:
                        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
                        match = re.search(r"BandedAnalysis time: ([\d.eE+-]+) ms", output)
                        
                        if match:
                            trials.append(float(match.group(1)))
                        else:
                            print(f"Error: Could not find timing in output for k={k} trial {i+1}")
                    except subprocess.CalledProcessError as e:
                        print(f"Error running benchmark for k={k} trial {i+1}:\n{e.output.decode()}")
                
                if trials:
                    avg_time = statistics.mean(trials)
                    trials_str = ", ".join([f"{t:.2f}" for t in trials])
                    print(f"{k:<6} | {avg_time:<15.2f} | {trials_str}")
                    
                    csv_writer.writerow([k, avg_time] + trials)
                    csvfile.flush()

                # if os.path.exists(filename):
                #     os.remove(filename)
                
        print(f"Results saved to {csv_filename}")

if __name__ == "__main__":
    run_benchmark()
