# AMD GPU test
AMD GPU backend test

## Installation
Steps to test Julia and `AMDGPU.jl` on CSCS's Piz Ault and MIT's Satori R&D system. Using a [local Julia install](#local-julia-install)

### Ault
Access one of ault's AMD Vega partitions (`ault07`, `ault08`, `ault20`):
```sh
srun -p amdvega -w ault20 --gres=gpu:1 -A c23 --time=01:00:00 --pty bash
```
Load AMD-related modules:
```sh
module load rocm hip-rocclr hip hsa-rocr-dev hsakmt-roct llvm-amdgpu rocm-cmake rocminfo roctracer-dev-api rocprofiler-dev rocm-smi-lib
```

### Satori
Access the AMD partition
```sh
srun --gres=gpu:4 -N 1 --partition=sched_system_penguin --mem=0 --time 6:00:00 --pty /bin/bash
```
ROCm libraries should be located in default `/opt/rocm/*`.

### Launch Julia
```sh
julia
```
and install `AMDGPU.jl`
```julia-repl
] add AMDGPU
build AMDGPU
test AMDGPU
```
Note that testing may still fail.

## The scripts
* [`test_amdgpu.jl`](scripts/test_amdgpu.jl) The [Quick Start example](https://amdgpu.juliagpu.org/stable/quickstart/) from AMDGPU.jl GitHub's
* [`memcopy3D_amdgpu.jl`](scripts/memcopy3D_amdgpu.jl) The memcopy tool (adapted from [here](https://github.com/luraess/parallel-gpu-workshop-JuliaCon21/blob/main/extras/memcopy3D.jl))
* [`diffusion_2D_perf_amdgpu.jl`](scripts/diffusion_2D_perf_amdgpu.jl) A "simpler" 2D diffusion script
* [`diffusion_2D_damp_perf_amdgpu.jl`](scripts/diffusion_2D_damp_perf_amdgpu.jl) The 2D diffusion solver [from JuliaCon21](https://github.com/luraess/parallel-gpu-workshop-JuliaCon21#gpu-implementation) workshop that delivers 92% of T_peak on Nvidia Tesla V100 PCIe 16GB GPUs.

## Results

### Running on Satori (Vega20 and V100 SXM2)

On Vega20 `T_peak = 1024 GB/s` ([scripts](scripts)):
```sh
[luraess@node2004 amdgpu-test]$ juliamdp -O3 --check-bounds=no memcopy3D_amdgpu.jl 
A_eff = (((2 * 1 + 1) * 1) / 1.0e9) * nx * ny * nz * sizeof(Float64) = 6.442450944
time_s=0.7967000007629395 T_eff=727.7778139886401
[luraess@node2004 amdgpu-test]$ juliamdp -O3 --check-bounds=no diffusion_2D_perf_amdgpu.jl 
Time = 2.741 sec, T_eff = 538.00 GB/s (niter = 610)
```

On V100 SXM2 `T_peak = 900 GB/s` ([scripts_cuda](scripts_cuda)):
```sh
[luraess@node0049 amdgpu-test-cudaref]$ juliap -O3 --check-bounds=no memcopy3D.jl 
A_eff = (((2 * 1 + 1) * 1) / 1.0e9) * nx * ny * nz * sizeof(Float64) = 6.442450944
time_s=0.7179579734802246 T_eff=807.5968320950342
[luraess@node0049 amdgpu-test-cudaref]$ juliap -O3 --check-bounds=no diffusion_2D_perf_gpu.jl 
Time = 2.017 sec, T_eff = 730.00 GB/s (niter = 610)
```

> Note that the results on the Radeon VII seem to be in-line with the results reported [in the Julia BabelStream bench](https://github.com/UoB-HPC/BabelStream/pull/106#issuecomment-897621652).



## Current issues/challenges
- 30% difference between measured `T_peak` and vendor announced `T_peak_vendor`. On Nvidia Tesla V100 GPU, the difference is about 7% only.
- 77% of `T_peak` for simple 2D diffusion code (Nvidia counterpart runs at 88%)
- Low perf of current naive AMD version [`diffusion_2D_damp_perf_amdgpu.jl`](scripts/diffusion_2D_damp_perf_amdgpu.jl). The Nvidia counterpart runs at 92% of `T_peak`


## Local Julia install

To nstall a local copy of Julia:
```sh
mkdir -p ~/julia_local
cd ~/julia_local
wget https://julialang-s3.julialang.org/bin/linux/x64/1.7/julia-1.7.0-linux-x86_64.tar.gz
tar -xzf julia-1.7.0-linux-x86_64.tar.gz
echo 'export PATH=~/julia_local/julia-1.7.0/bin/:$PATH' >> ~/.bashrc
echo 'export JULIA_AMDGPU_DISABLE_ARTIFACTS=1' >> ~/.bashrc
bash
```
Ensure to set `JULIA_AMDGPU_DISABLE_ARTIFACTS=1` in order not to use the artifacts.
