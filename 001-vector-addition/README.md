# Devices

## [0] `NVIDIA GeForce RTX 3070 Laptop GPU`
* SM Version: 860 (PTX Version: 750)
* Number of SMs: 40
* SM Default Clock Rate: 1560 MHz
* Global Memory: 7116 MiB Free / 8191 MiB Total
* Global Memory Bus Peak: 448 GB/sec (256-bit DDR @7001MHz)
* Max Shared Memory: 100 KiB/SM, 48 KiB/Block
* L2 Cache Size: 4096 KiB
* Maximum Active Blocks: 16/SM
* Maximum Active Threads: 1536/SM, 1024/Block
* Available Registers: 65536/SM, 65536/Block
* ECC Enabled: No

# Log

```
Run:  [1/28] vector_add [Device=0 N=1024]
Pass: Cold: 0.106528ms GPU, 0.132959ms CPU, 0.50s total GPU, 0.91s total wall, 4704x 
Run:  [2/28] vector_add [Device=0 N=2051]
Pass: Cold: 0.105741ms GPU, 0.134849ms CPU, 0.50s total GPU, 0.93s total wall, 4736x 
Run:  [3/28] vector_add [Device=0 N=1024]
Pass: Cold: 0.099877ms GPU, 0.124105ms CPU, 0.50s total GPU, 0.94s total wall, 5024x 
Run:  [4/28] vector_add [Device=0 N=65536]
Pass: Cold: 0.103046ms GPU, 0.131951ms CPU, 0.50s total GPU, 0.95s total wall, 4864x 
Run:  [5/28] vector_add [Device=0 N=1048576]
Pass: Cold: 0.147182ms GPU, 0.169212ms CPU, 0.50s total GPU, 0.82s total wall, 3408x 
Run:  [6/28] vector_add [Device=0 N=67108864]
Pass: Cold: 2.620982ms GPU, 2.432510ms CPU, 1.43s total GPU, 1.38s total wall, 544x 
Run:  [7/28] vector_add [Device=0 N=268435456]
Pass: Cold: 10.037197ms GPU, 9.227202ms CPU, 2.41s total GPU, 2.25s total wall, 240x 
Run:  [8/28] vector_add_stride [Device=0 N=1024]
Pass: Cold: 0.098728ms GPU, 0.124133ms CPU, 0.50s total GPU, 0.95s total wall, 5072x 
Run:  [9/28] vector_add_stride [Device=0 N=2051]
Pass: Cold: 0.104085ms GPU, 0.132358ms CPU, 0.50s total GPU, 0.95s total wall, 4832x 
Run:  [10/28] vector_add_stride [Device=0 N=1024]
Pass: Cold: 0.106747ms GPU, 0.135513ms CPU, 0.52s total GPU, 0.97s total wall, 4832x 
Run:  [11/28] vector_add_stride [Device=0 N=65536]
Pass: Cold: 0.118539ms GPU, 0.149085ms CPU, 0.50s total GPU, 0.93s total wall, 4240x 
Run:  [12/28] vector_add_stride [Device=0 N=1048576]
Pass: Cold: 0.149957ms GPU, 0.182275ms CPU, 0.50s total GPU, 0.84s total wall, 3344x 
Run:  [13/28] vector_add_stride [Device=0 N=67108864]
Pass: Cold: 2.641022ms GPU, 2.468521ms CPU, 1.44s total GPU, 1.40s total wall, 544x 
Run:  [14/28] vector_add_stride [Device=0 N=268435456]
Pass: Cold: 10.082720ms GPU, 9.264539ms CPU, 2.58s total GPU, 2.40s total wall, 256x 
Run:  [15/28] vector_add_vectorized [Device=0 N=1024]
Pass: Cold: 0.111830ms GPU, 0.136970ms CPU, 0.50s total GPU, 0.91s total wall, 4480x 
Run:  [16/28] vector_add_vectorized [Device=0 N=2051]
Pass: Cold: 0.104813ms GPU, 0.128923ms CPU, 0.50s total GPU, 0.91s total wall, 4784x 
Run:  [17/28] vector_add_vectorized [Device=0 N=1024]
Pass: Cold: 0.105768ms GPU, 0.137743ms CPU, 0.50s total GPU, 0.95s total wall, 4736x 
Run:  [18/28] vector_add_vectorized [Device=0 N=65536]
Pass: Cold: 0.108490ms GPU, 0.139446ms CPU, 0.50s total GPU, 0.95s total wall, 4624x 
Run:  [19/28] vector_add_vectorized [Device=0 N=1048576]
Pass: Cold: 0.140792ms GPU, 0.169119ms CPU, 0.50s total GPU, 0.83s total wall, 3552x 
Run:  [20/28] vector_add_vectorized [Device=0 N=67108864]
Pass: Cold: 2.801392ms GPU, 2.608016ms CPU, 0.54s total GPU, 0.52s total wall, 192x 
Run:  [21/28] vector_add_vectorized [Device=0 N=268435456]
Pass: Cold: 10.574250ms GPU, 13.191706ms CPU, 9.14s total GPU, 11.50s total wall, 864x 
Run:  [22/28] vector_add_vectorized2 [Device=0 N=1024]
Pass: Cold: 0.107597ms GPU, 0.130818ms CPU, 0.50s total GPU, 0.92s total wall, 4656x 
Run:  [23/28] vector_add_vectorized2 [Device=0 N=2051]
Pass: Cold: 0.104231ms GPU, 0.126383ms CPU, 0.50s total GPU, 0.89s total wall, 4800x 
Run:  [24/28] vector_add_vectorized2 [Device=0 N=1024]
Pass: Cold: 0.101888ms GPU, 0.134572ms CPU, 0.50s total GPU, 0.96s total wall, 4912x 
Run:  [25/28] vector_add_vectorized2 [Device=0 N=65536]
Pass: Cold: 0.106769ms GPU, 0.136183ms CPU, 0.50s total GPU, 0.95s total wall, 4688x 
Run:  [26/28] vector_add_vectorized2 [Device=0 N=1048576]
Pass: Cold: 0.145021ms GPU, 0.182490ms CPU, 0.50s total GPU, 0.88s total wall, 3456x 
Run:  [27/28] vector_add_vectorized2 [Device=0 N=67108864]
Pass: Cold: 2.814197ms GPU, 2.608454ms CPU, 0.54s total GPU, 0.53s total wall, 192x 
Run:  [28/28] vector_add_vectorized2 [Device=0 N=268435456]
Pass: Cold: 10.800547ms GPU, 9.926324ms CPU, 15.55s total GPU, 14.50s total wall, 1440x 
```

# Benchmark Results

## vector_add

### [0] NVIDIA GeForce RTX 3070 Laptop GPU

|     N     | Samples |  CPU Time  |  Noise  |  GPU Time  |  Noise  |
|-----------|---------|------------|---------|------------|---------|
|      1024 |   4704x | 132.959 us | 209.32% | 106.528 us | 232.46% |
|      2051 |   4736x | 134.849 us | 207.95% | 105.741 us | 227.63% |
|      1024 |   5024x | 124.105 us | 194.14% |  99.877 us | 212.02% |
|     65536 |   4864x | 131.951 us | 199.64% | 103.046 us | 210.67% |
|   1048576 |   3408x | 169.212 us | 177.20% | 147.182 us | 191.27% |
|  67108864 |    544x |   2.433 ms |  44.95% |   2.621 ms |  45.29% |
| 268435456 |    240x |   9.227 ms |  12.56% |  10.037 ms |  12.70% |

## vector_add_stride

### [0] NVIDIA GeForce RTX 3070 Laptop GPU

|     N     | Samples |  CPU Time  |  Noise  |  GPU Time  |  Noise  |
|-----------|---------|------------|---------|------------|---------|
|      1024 |   5072x | 124.133 us | 191.64% |  98.728 us | 208.66% |
|      2051 |   4832x | 132.358 us | 203.10% | 104.085 us | 208.78% |
|      1024 |   4832x | 135.513 us | 214.67% | 106.747 us | 237.84% |
|     65536 |   4240x | 149.085 us | 233.54% | 118.539 us | 266.09% |
|   1048576 |   3344x | 182.275 us | 183.21% | 149.957 us | 189.31% |
|  67108864 |    544x |   2.469 ms |  44.45% |   2.641 ms |  44.80% |
| 268435456 |    256x |   9.265 ms |  12.75% |  10.083 ms |  12.87% |

## vector_add_vectorized

### [0] NVIDIA GeForce RTX 3070 Laptop GPU

|     N     | Samples |  CPU Time  |  Noise  |  GPU Time  |  Noise  |
|-----------|---------|------------|---------|------------|---------|
|      1024 |   4480x | 136.970 us | 211.21% | 111.830 us | 237.69% |
|      2051 |   4784x | 128.923 us | 196.41% | 104.813 us | 223.81% |
|      1024 |   4736x | 137.743 us | 202.41% | 105.768 us | 211.70% |
|     65536 |   4624x | 139.446 us | 204.82% | 108.490 us | 209.59% |
|   1048576 |   3552x | 169.119 us | 177.67% | 140.792 us | 182.25% |
|  67108864 |    192x |   2.608 ms |  40.96% |   2.801 ms |  41.17% |
| 268435456 |    864x |  13.192 ms | 776.32% |  10.574 ms |  13.89% |

## vector_add_vectorized2

### [0] NVIDIA GeForce RTX 3070 Laptop GPU

|     N     | Samples |  CPU Time  |  Noise  |  GPU Time  |  Noise  |
|-----------|---------|------------|---------|------------|---------|
|      1024 |   4656x | 130.818 us | 197.37% | 107.597 us | 231.46% |
|      2051 |   4800x | 126.383 us | 197.38% | 104.231 us | 223.62% |
|      1024 |   4912x | 134.572 us | 213.68% | 101.888 us | 218.91% |
|     65536 |   4688x | 136.183 us | 198.14% | 106.769 us | 216.07% |
|   1048576 |   3456x | 182.490 us | 175.63% | 145.021 us | 167.62% |
|  67108864 |    192x |   2.608 ms |  41.12% |   2.814 ms |  41.62% |
| 268435456 |   1440x |   9.926 ms |  14.13% |  10.801 ms |  14.08% |

---

# NCU results

ncu -i vector_add.ncu-rep
[16925] 001_vector_addition_benchmark@127.0.0.1
  vector_add(const float *, const float *, float *, int) (262144, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         6.99
    SM Frequency                    Ghz         1.11
    Elapsed Cycles                cycle      2107384
    Memory Throughput                 %        95.02
    DRAM Throughput                   %        95.02
    Duration                         ms         1.90
    L1/TEX Cache Throughput           %        24.98
    L2 Cache Throughput               %        39.01
    SM Active Cycles              cycle   2139959.80
    Compute (SM) Throughput           %        19.26
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of the device.   
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.   
          Start by analyzing DRAM in the Memory Workload Analysis section.                                              

    Section: GPU Speed Of Light Roofline Chart
    INF   The ratio of peak float (FP32) to double (FP64) performance on this device is 64:1. The workload achieved     
          close to 0% of this device's FP32 peak performance and 0% of its FP64 peak performance. See the Profiling     
          Guide (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline) for more details on         
          roofline analysis.                                                                                            

    Section: PM Sampling
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Maximum Buffer Size             Mbyte        14.61
    Maximum Sampling Interval          us            2
    # Pass Groups                                    2
    ------------------------- ----------- ------------

    Section: Compute Workload Analysis
    -------------------- ----------- ------------
    Metric Name          Metric Unit Metric Value
    -------------------- ----------- ------------
    Executed Ipc Active   inst/cycle         0.39
    Executed Ipc Elapsed  inst/cycle         0.39
    Issue Slots Busy               %         9.63
    Issued Ipc Active     inst/cycle         0.39
    SM Busy                        %         9.63
    -------------------- ----------- ------------

    OPT   Est. Local Speedup: 95.18%                                                                                    
          All compute pipelines are under-utilized. Either this workload is very small or it doesn't issue enough warps 
          per scheduler. Check the Launch Statistics and Scheduler Statistics sections for further details.             

    Section: Memory Workload Analysis
    -------------------------------------- ----------- ------------
    Metric Name                            Metric Unit Metric Value
    -------------------------------------- ----------- ------------
    Local Memory Spilling Requests                                0
    Local Memory Spilling Request Overhead           %            0
    Memory Throughput                          Gbyte/s       425.29
    Mem Busy                                         %        39.01
    Max Bandwidth                                    %        95.02
    L1/TEX Hit Rate                                  %            0
    L2 Persisting Size                           Kbyte       786.43
    L2 Compression Success Rate                      %            0
    L2 Compression Ratio                                          0
    L2 Compression Input Sectors                sector            0
    L2 Hit Rate                                      %        33.25
    Mem Pipes Busy                                   %        19.26
    -------------------------------------- ----------- ------------

    Section: Scheduler Statistics
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    One or More Eligible                   %         9.73
    Issued Warp Per Scheduler                        0.10
    No Eligible                            %        90.27
    Active Warps Per Scheduler          warp         8.86
    Eligible Warps Per Scheduler        warp         0.11
    ---------------------------- ----------- ------------

    OPT   Est. Local Speedup: 4.983%                                                                                    
          Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only    
          issues an instruction every 10.3 cycles. This might leave hardware resources underutilized and may lead to    
          less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average    
          of 8.86 active warps per scheduler, but only an average of 0.11 warps were eligible per cycle. Eligible       
          warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no      
          eligible warp results in no instruction being issued and the issue slot remains unused. To increase the       
          number of eligible warps, reduce the time the active warps are stalled by inspecting the top stall reasons    
          on the Warp State Statistics and Source Counters sections.                                                    

    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle        91.13
    Warp Cycles Per Executed Instruction           cycle        91.15
    Avg. Active Threads Per Warp                                   32
    Avg. Not Predicated Off Threads Per Warp                       30
    ---------------------------------------- ----------- ------------

    OPT   Est. Speedup: 4.983%                                                                                          
          On average, each warp of this workload spends 82.1 cycles being stalled waiting for a scoreboard dependency   
          on a L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited  
          upon to identify the culprit. To reduce the number of cycles waiting on L1TEX data accesses verify the        
          memory access patterns are optimal for the target architecture, attempt to increase cache hit rates by        
          increasing data locality (coalescing), or by changing the cache configuration. Consider moving frequently     
          used data to shared memory. This stall type represents about 90.1% of the total average of 91.1 cycles        
          between issuing two instructions.                                                                             
    ----- --------------------------------------------------------------------------------------------------------------
    INF   Check the Warp Stall Sampling (All Samples) table for the top stall locations in your source based on         
          sampling data. The Profiling Guide                                                                            
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference) provides more details    
          on each stall reason.                                                                                         

    Section: Instruction Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Avg. Executed Instructions Per Scheduler        inst    209715.20
    Executed Instructions                           inst     33554432
    Avg. Issued Instructions Per Scheduler          inst    209766.10
    Issued Instructions                             inst     33562576
    ---------------------------------------- ----------- ------------

    OPT   Est. Speedup: 2.408%                                                                                          
          This kernel executes 0 fused and 2097152 non-fused FP32 instructions. By converting pairs of non-fused        
          instructions to their fused (https://docs.nvidia.com/cuda/floating-point/#cuda-and-floating-point),           
          higher-throughput equivalent, the achieved FP32 performance could be increased by up to 50% (relative to its  
          current performance).                                                                                         

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                 262144
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte            8.19
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              40
    Stack Size                                                  1024
    Threads                                   thread        67108864
    # TPCs                                                        20
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                             1092.27
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           16
    Block Limit Shared Mem                block            8
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        77.02
    Achieved Active Warps Per SM           warp        36.97
    ------------------------------- ----------- ------------

    OPT   Est. Speedup: 4.983%                                                                                          
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (77.0%) can be the     
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                                                                         

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle     12617098
    Total DRAM Elapsed Cycles        cycle    106230784
    Average L1 Active Cycles         cycle   2139959.80
    Total L1 Elapsed Cycles          cycle     87092634
    Average L2 Active Cycles         cycle   2079115.12
    Total L2 Elapsed Cycles          cycle     64703008
    Average SM Active Cycles         cycle   2139959.80
    Total SM Elapsed Cycles          cycle     87092634
    Average SMSP Active Cycles       cycle   2156478.14
    Total SMSP Elapsed Cycles        cycle    348370536
    -------------------------- ----------- ------------

    Section: Source Counters
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Branch Instructions Ratio           %         0.12
    Branch Instructions              inst      4194304
    Branch Efficiency                   %            0
    Avg. Divergent Branches      branches            0
    ------------------------- ----------- ------------

  vector_add_stride(const float *, const float *, float *, int) (262144, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         6.99
    SM Frequency                    Ghz         1.11
    Elapsed Cycles                cycle      2105738
    Memory Throughput                 %        95.11
    DRAM Throughput                   %        95.11
    Duration                         ms         1.90
    L1/TEX Cache Throughput           %        24.34
    L2 Cache Throughput               %        39.06
    SM Active Cycles              cycle   2131611.23
    Compute (SM) Throughput           %        18.67
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of the device.   
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.   
          Start by analyzing DRAM in the Memory Workload Analysis section.                                              

    Section: GPU Speed Of Light Roofline Chart
    INF   The ratio of peak float (FP32) to double (FP64) performance on this device is 64:1. The workload achieved     
          close to 0% of this device's FP32 peak performance and 0% of its FP64 peak performance. See the Profiling     
          Guide (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline) for more details on         
          roofline analysis.                                                                                            

    Section: PM Sampling
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Maximum Buffer Size             Mbyte        14.61
    Maximum Sampling Interval          us            2
    # Pass Groups                                    2
    ------------------------- ----------- ------------

    Section: Compute Workload Analysis
    -------------------- ----------- ------------
    Metric Name          Metric Unit Metric Value
    -------------------- ----------- ------------
    Executed Ipc Active   inst/cycle         0.76
    Executed Ipc Elapsed  inst/cycle         0.72
    Issue Slots Busy               %        18.09
    Issued Ipc Active     inst/cycle         0.76
    SM Busy                        %        18.09
    -------------------- ----------- ------------

    OPT   Est. Local Speedup: 85.99%                                                                                    
          All compute pipelines are under-utilized. Either this workload is very small or it doesn't issue enough warps 
          per scheduler. Check the Launch Statistics and Scheduler Statistics sections for further details.             

    Section: Memory Workload Analysis
    -------------------------------------- ----------- ------------
    Metric Name                            Metric Unit Metric Value
    -------------------------------------- ----------- ------------
    Local Memory Spilling Requests                                0
    Local Memory Spilling Request Overhead           %            0
    Memory Throughput                          Gbyte/s       425.69
    Mem Busy                                         %        39.06
    Max Bandwidth                                    %        95.11
    L1/TEX Hit Rate                                  %            0
    L2 Persisting Size                           Kbyte       786.43
    L2 Compression Success Rate                      %            0
    L2 Compression Ratio                                          0
    L2 Compression Input Sectors                sector            0
    L2 Hit Rate                                      %        33.24
    Mem Pipes Busy                                   %        18.67
    -------------------------------------- ----------- ------------

    Section: Scheduler Statistics
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    One or More Eligible                   %        18.90
    Issued Warp Per Scheduler                        0.19
    No Eligible                            %        81.10
    Active Warps Per Scheduler          warp         9.10
    Eligible Warps Per Scheduler        warp         0.27
    ---------------------------- ----------- ------------

    OPT   Est. Local Speedup: 4.894%                                                                                    
          Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only    
          issues an instruction every 5.3 cycles. This might leave hardware resources underutilized and may lead to     
          less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average    
          of 9.10 active warps per scheduler, but only an average of 0.27 warps were eligible per cycle. Eligible       
          warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no      
          eligible warp results in no instruction being issued and the issue slot remains unused. To increase the       
          number of eligible warps, reduce the time the active warps are stalled by inspecting the top stall reasons    
          on the Warp State Statistics and Source Counters sections.                                                    

    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle        48.14
    Warp Cycles Per Executed Instruction           cycle        48.15
    Avg. Active Threads Per Warp                                   32
    Avg. Not Predicated Off Threads Per Warp                    29.94
    ---------------------------------------- ----------- ------------

    OPT   Est. Speedup: 4.894%                                                                                          
          On average, each warp of this workload spends 42.6 cycles being stalled waiting for a scoreboard dependency   
          on a L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited  
          upon to identify the culprit. To reduce the number of cycles waiting on L1TEX data accesses verify the        
          memory access patterns are optimal for the target architecture, attempt to increase cache hit rates by        
          increasing data locality (coalescing), or by changing the cache configuration. Consider moving frequently     
          used data to shared memory. This stall type represents about 88.4% of the total average of 48.1 cycles        
          between issuing two instructions.                                                                             
    ----- --------------------------------------------------------------------------------------------------------------
    INF   Check the Warp Stall Sampling (All Samples) table for the top stall locations in your source based on         
          sampling data. The Profiling Guide                                                                            
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference) provides more details    
          on each stall reason.                                                                                         

    Section: Instruction Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Avg. Executed Instructions Per Scheduler        inst    406323.20
    Executed Instructions                           inst     65011712
    Avg. Issued Instructions Per Scheduler          inst    406418.06
    Issued Instructions                             inst     65026889
    ---------------------------------------- ----------- ------------

    OPT   Est. Speedup: 2.334%                                                                                          
          This kernel executes 0 fused and 2097152 non-fused FP32 instructions. By converting pairs of non-fused        
          instructions to their fused (https://docs.nvidia.com/cuda/floating-point/#cuda-and-floating-point),           
          higher-throughput equivalent, the achieved FP32 performance could be increased by up to 50% (relative to its  
          current performance).                                                                                         

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                 262144
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte            8.19
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              40
    Stack Size                                                  1024
    Threads                                   thread        67108864
    # TPCs                                                        20
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                             1092.27
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           16
    Block Limit Shared Mem                block            8
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        77.01
    Achieved Active Warps Per SM           warp        36.96
    ------------------------------- ----------- ------------

    OPT   Est. Speedup: 4.894%                                                                                          
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (77.0%) can be the     
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                                                                         

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle     12619060
    Total DRAM Elapsed Cycles        cycle    106147840
    Average L1 Active Cycles         cycle   2131611.23
    Total L1 Elapsed Cycles          cycle     89845004
    Average L2 Active Cycles         cycle      2045864
    Total L2 Elapsed Cycles          cycle     64652352
    Average SM Active Cycles         cycle   2131611.23
    Total SM Elapsed Cycles          cycle     89845004
    Average SMSP Active Cycles       cycle   2150336.96
    Total SMSP Elapsed Cycles        cycle    359380016
    -------------------------- ----------- ------------

    Section: Source Counters
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Branch Instructions Ratio           %         0.10
    Branch Instructions              inst      6291456
    Branch Efficiency                   %          100
    Avg. Divergent Branches      branches            0
    ------------------------- ----------- ------------

  vector_add_vectorized(const float *, const float *, float *, int) (262144, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         6.99
    SM Frequency                    Ghz         1.11
    Elapsed Cycles                cycle      2451425
    Memory Throughput                 %        82.23
    DRAM Throughput                   %        82.23
    Duration                         ms         2.21
    L1/TEX Cache Throughput           %        22.24
    L2 Cache Throughput               %        33.76
    SM Active Cycles              cycle   2419362.95
    Compute (SM) Throughput           %        11.66
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of the device.   
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.   
          Start by analyzing DRAM in the Memory Workload Analysis section.                                              

    Section: GPU Speed Of Light Roofline Chart
    INF   The ratio of peak float (FP32) to double (FP64) performance on this device is 64:1. The workload achieved     
          close to 0% of this device's FP32 peak performance and 0% of its FP64 peak performance. See the Profiling     
          Guide (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline) for more details on         
          roofline analysis.                                                                                            

    Section: PM Sampling
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Maximum Buffer Size             Mbyte        15.14
    Maximum Sampling Interval          us            2
    # Pass Groups                                    2
    ------------------------- ----------- ------------

    Section: Compute Workload Analysis
    -------------------- ----------- ------------
    Metric Name          Metric Unit Metric Value
    -------------------- ----------- ------------
    Executed Ipc Active   inst/cycle         0.48
    Executed Ipc Elapsed  inst/cycle         0.47
    Issue Slots Busy               %        11.66
    Issued Ipc Active     inst/cycle         0.48
    SM Busy                        %        11.66
    -------------------- ----------- ------------

    OPT   Est. Local Speedup: 91%                                                                                       
          All compute pipelines are under-utilized. Either this workload is very small or it doesn't issue enough warps 
          per scheduler. Check the Launch Statistics and Scheduler Statistics sections for further details.             

    Section: Memory Workload Analysis
    -------------------------------------- ----------- ------------
    Metric Name                            Metric Unit Metric Value
    -------------------------------------- ----------- ------------
    Local Memory Spilling Requests                                0
    Local Memory Spilling Request Overhead           %            0
    Memory Throughput                          Gbyte/s       368.08
    Mem Busy                                         %        33.76
    Max Bandwidth                                    %        82.23
    L1/TEX Hit Rate                                  %            0
    L2 Persisting Size                           Kbyte       786.43
    L2 Compression Success Rate                      %            0
    L2 Compression Ratio                                          0
    L2 Compression Input Sectors                sector            0
    L2 Hit Rate                                      %        33.03
    Mem Pipes Busy                                   %         7.41
    -------------------------------------- ----------- ------------

    Section: Scheduler Statistics
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    One or More Eligible                   %        11.57
    Issued Warp Per Scheduler                        0.12
    No Eligible                            %        88.43
    Active Warps Per Scheduler          warp         8.89
    Eligible Warps Per Scheduler        warp         0.22
    ---------------------------- ----------- ------------

    OPT   Est. Local Speedup: 17.77%                                                                                    
          Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only    
          issues an instruction every 8.6 cycles. This might leave hardware resources underutilized and may lead to     
          less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average    
          of 8.89 active warps per scheduler, but only an average of 0.22 warps were eligible per cycle. Eligible       
          warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no      
          eligible warp results in no instruction being issued and the issue slot remains unused. To increase the       
          number of eligible warps, reduce the time the active warps are stalled by inspecting the top stall reasons    
          on the Warp State Statistics and Source Counters sections.                                                    

    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle        76.83
    Warp Cycles Per Executed Instruction           cycle        76.89
    Avg. Active Threads Per Warp                                   32
    Avg. Not Predicated Off Threads Per Warp                    31.27
    ---------------------------------------- ----------- ------------

    OPT   Est. Speedup: 17.77%                                                                                          
          On average, each warp of this workload spends 62.9 cycles being stalled waiting for a scoreboard dependency   
          on a L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited  
          upon to identify the culprit. To reduce the number of cycles waiting on L1TEX data accesses verify the        
          memory access patterns are optimal for the target architecture, attempt to increase cache hit rates by        
          increasing data locality (coalescing), or by changing the cache configuration. Consider moving frequently     
          used data to shared memory. This stall type represents about 81.9% of the total average of 76.8 cycles        
          between issuing two instructions.                                                                             
    ----- --------------------------------------------------------------------------------------------------------------
    INF   Check the Warp Stall Sampling (All Samples) table for the top stall locations in your source based on         
          sampling data. The Profiling Guide                                                                            
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference) provides more details    
          on each stall reason.                                                                                         

    Section: Instruction Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Avg. Executed Instructions Per Scheduler        inst    288358.40
    Executed Instructions                           inst     46137344
    Avg. Issued Instructions Per Scheduler          inst    288558.53
    Issued Instructions                             inst     46169365
    ---------------------------------------- ----------- ------------

    OPT   Est. Speedup: 1.655%                                                                                          
          This kernel executes 0 fused and 2097152 non-fused FP32 instructions. By converting pairs of non-fused        
          instructions to their fused (https://docs.nvidia.com/cuda/floating-point/#cuda-and-floating-point),           
          higher-throughput equivalent, the achieved FP32 performance could be increased by up to 50% (relative to its  
          current performance).                                                                                         

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                 262144
    Registers Per Thread             register/thread              20
    Shared Memory Configuration Size           Kbyte            8.19
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              40
    Stack Size                                                  1024
    Threads                                   thread        67108864
    # TPCs                                                        20
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                             1092.27
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           10
    Block Limit Shared Mem                block            8
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        78.57
    Achieved Active Warps Per SM           warp        37.71
    ------------------------------- ----------- ------------

    OPT   Est. Speedup: 17.77%                                                                                          
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (78.6%) can be the     
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                                                                         

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle     12702768
    Total DRAM Elapsed Cycles        cycle    123578368
    Average L1 Active Cycles         cycle   2419362.95
    Total L1 Elapsed Cycles          cycle     99015216
    Average L2 Active Cycles         cycle   2246530.50
    Total L2 Elapsed Cycles          cycle     75268896
    Average SM Active Cycles         cycle   2419362.95
    Total SM Elapsed Cycles          cycle     99015216
    Average SMSP Active Cycles       cycle   2494452.58
    Total SMSP Elapsed Cycles        cycle    396060864
    -------------------------- ----------- ------------

    Section: Source Counters
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Branch Instructions Ratio           %         0.15
    Branch Instructions              inst      6817134
    Branch Efficiency                   %          100
    Avg. Divergent Branches      branches            0
    ------------------------- ----------- ------------

  vector_add_vectorized2(const float *, const float *, float *, int) (262144, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         6.99
    SM Frequency                    Ghz         1.11
    Elapsed Cycles                cycle      2466784
    Memory Throughput                 %        82.34
    DRAM Throughput                   %        82.34
    Duration                         ms         2.22
    L1/TEX Cache Throughput           %        22.00
    L2 Cache Throughput               %        33.81
    SM Active Cycles              cycle   2380912.35
    Compute (SM) Throughput           %         7.33
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of the device.   
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.   
          Start by analyzing DRAM in the Memory Workload Analysis section.                                              

    Section: GPU Speed Of Light Roofline Chart
    INF   The ratio of peak float (FP32) to double (FP64) performance on this device is 64:1. The workload achieved     
          close to 0% of this device's FP32 peak performance and 0% of its FP64 peak performance. See the Profiling     
          Guide (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline) for more details on         
          roofline analysis.                                                                                            

    Section: PM Sampling
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Maximum Buffer Size             Mbyte        14.61
    Maximum Sampling Interval          us            2
    # Pass Groups                                    2
    ------------------------- ----------- ------------

    Section: Compute Workload Analysis
    -------------------- ----------- ------------
    Metric Name          Metric Unit Metric Value
    -------------------- ----------- ------------
    Executed Ipc Active   inst/cycle         0.23
    Executed Ipc Elapsed  inst/cycle         0.21
    Issue Slots Busy               %         5.37
    Issued Ipc Active     inst/cycle         0.23
    SM Busy                        %         5.37
    -------------------- ----------- ------------

    OPT   Est. Local Speedup: 96.6%                                                                                     
          All compute pipelines are under-utilized. Either this workload is very small or it doesn't issue enough warps 
          per scheduler. Check the Launch Statistics and Scheduler Statistics sections for further details.             

    Section: Memory Workload Analysis
    -------------------------------------- ----------- ------------
    Metric Name                            Metric Unit Metric Value
    -------------------------------------- ----------- ------------
    Local Memory Spilling Requests                                0
    Local Memory Spilling Request Overhead           %            0
    Memory Throughput                          Gbyte/s       368.56
    Mem Busy                                         %        33.81
    Max Bandwidth                                    %        82.34
    L1/TEX Hit Rate                                  %            0
    L2 Persisting Size                           Kbyte       786.43
    L2 Compression Success Rate                      %            0
    L2 Compression Ratio                                          0
    L2 Compression Input Sectors                sector            0
    L2 Hit Rate                                      %        32.78
    Mem Pipes Busy                                   %         7.33
    -------------------------------------- ----------- ------------

    Section: Scheduler Statistics
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    One or More Eligible                   %         5.77
    Issued Warp Per Scheduler                        0.06
    No Eligible                            %        94.23
    Active Warps Per Scheduler          warp         9.51
    Eligible Warps Per Scheduler        warp         0.07
    ---------------------------- ----------- ------------

    OPT   Est. Local Speedup: 17.66%                                                                                    
          Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only    
          issues an instruction every 17.3 cycles. This might leave hardware resources underutilized and may lead to    
          less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average    
          of 9.51 active warps per scheduler, but only an average of 0.07 warps were eligible per cycle. Eligible       
          warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no      
          eligible warp results in no instruction being issued and the issue slot remains unused. To increase the       
          number of eligible warps, reduce the time the active warps are stalled by inspecting the top stall reasons    
          on the Warp State Statistics and Source Counters sections.                                                    

    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle       164.94
    Warp Cycles Per Executed Instruction           cycle       165.05
    Avg. Active Threads Per Warp                                   32
    Avg. Not Predicated Off Threads Per Warp                    31.22
    ---------------------------------------- ----------- ------------

    OPT   Est. Speedup: 17.66%                                                                                          
          On average, each warp of this workload spends 119.7 cycles being stalled waiting for a scoreboard dependency  
          on a L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited  
          upon to identify the culprit. To reduce the number of cycles waiting on L1TEX data accesses verify the        
          memory access patterns are optimal for the target architecture, attempt to increase cache hit rates by        
          increasing data locality (coalescing), or by changing the cache configuration. Consider moving frequently     
          used data to shared memory. This stall type represents about 72.6% of the total average of 164.9 cycles       
          between issuing two instructions.                                                                             
    ----- --------------------------------------------------------------------------------------------------------------
    INF   Check the Warp Stall Sampling (All Samples) table for the top stall locations in your source based on         
          sampling data. The Profiling Guide                                                                            
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference) provides more details    
          on each stall reason.                                                                                         

    Section: Instruction Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Avg. Executed Instructions Per Scheduler        inst    134348.80
    Executed Instructions                           inst     21495808
    Avg. Issued Instructions Per Scheduler          inst    134440.03
    Issued Instructions                             inst     21510405
    ---------------------------------------- ----------- ------------

    OPT   Est. Speedup: 0.9161%                                                                                         
          This kernel executes 0 fused and 2097152 non-fused FP32 instructions. By converting pairs of non-fused        
          instructions to their fused (https://docs.nvidia.com/cuda/floating-point/#cuda-and-floating-point),           
          higher-throughput equivalent, the achieved FP32 performance could be increased by up to 50% (relative to its  
          current performance).                                                                                         

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                 262144
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte            8.19
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              40
    Stack Size                                                  1024
    Threads                                   thread        67108864
    # TPCs                                                        20
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                             1092.27
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           16
    Block Limit Shared Mem                block            8
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        79.21
    Achieved Active Warps Per SM           warp        38.02
    ------------------------------- ----------- ------------

    OPT   Est. Speedup: 17.66%                                                                                          
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (79.2%) can be the     
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                                                                         

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle     12798746
    Total DRAM Elapsed Cycles        cycle    124348416
    Average L1 Active Cycles         cycle   2380912.35
    Total L1 Elapsed Cycles          cycle    100157226
    Average L2 Active Cycles         cycle   2207581.75
    Total L2 Elapsed Cycles          cycle     75739040
    Average SM Active Cycles         cycle   2380912.35
    Total SM Elapsed Cycles          cycle    100157226
    Average SMSP Active Cycles       cycle   2331780.86
    Total SMSP Elapsed Cycles        cycle    400628904
    -------------------------- ----------- ------------

    Section: Source Counters
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Branch Instructions Ratio           %         0.12
    Branch Instructions              inst      2621440
    Branch Efficiency                   %            0
    Avg. Divergent Branches      branches            0
    ------------------------- ----------- ------------