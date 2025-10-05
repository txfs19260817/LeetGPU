How to run ncu on WSL?

1. Follow the Windows section in https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters to allow access to the GPU performance counter to all users.
2. Restart wsl in powershell by running `wsl --shutdown`
3. Run `ncu` (without sudo):
  ```bash
  ncu \
    --set=full \
    -f \
    --kernel-name-base demangled \
    --kernel-name 'regex:vector_add' \
    -o vector_add \
    ./001_vector_addition_benchmark \
    --profile \
    --axis "N=67108864"
  ```

### NCU Command Flags Explained

- `--set=full`: Collect the complete set of performance metrics (most comprehensive profiling)
- `-f`: Force overwrite output files if they already exist
- `--kernel-name-base demangled`: Use demangled (human-readable) kernel names in output
- `--kernel-name 'regex:vector_add'`: Only profile kernels matching the regex pattern "vector_add"
- `-o vector_add`: Output results to files with "vector_add" prefix (creates .ncu-rep files)
- `./001_vector_addition_benchmark`: The executable to profile
- `--profile`: Enable detailed profiling (as opposed to just metrics collection)
- `--axis "N=67108864"`: Pass this parameter to nvbench to run only the 67M element test case

### Output Files
This will generate `vector_add.ncu-rep` which can be opened in:
- **Nsight Compute GUI** (Windows): For interactive analysis with charts and recommendations
- **Command line**: `ncu -i vector_add.ncu-rep` for text-based analysis

Flags for nvbench program can be found in https://github.com/NVIDIA/nvbench/blob/main/docs/cli_help.md

