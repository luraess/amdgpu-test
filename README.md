# AMD GPU test
AMD GPU backend test

## Installation
👉 Steps to get Julia and `AMDGPU.jl` on CSCS's `ault` R&D system

Log in to the system:
```sh
ssh <user>@ault.cscs.ch
```
Then, install a local copy of Julia:
```sh
mkdir -p ~/julia_local
cd ~/julia_local
wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.2-linux-x86_64.tar.gz
tar -xzf julia-1.6.2-linux-x86_64.tar.gz
echo 'export PATH=~/julia_local/julia-1.6.2/bin/:$PATH' >> ~/.bashrc
echo 'export JULIA_AMDGPU_DISABLE_ARTIFACTS=1' >> ~/.bashrc
bash
```
Ensure to set `JULIA_AMDGPU_DISABLE_ARTIFACTS=1` in order not to use the artifacts.

Access one of ault's AMD Vega partitions (`ault07`, `ault08`, `ault20`):
```sh
srun -p amdvega -w ault07 --gres=gpu:1 -A c23 --time=01:00:00 --pty bash
```

Load AMD-related modules:
```sh
module load rocm hip-rocclr hip hsa-rocr-dev hsakmt-roct llvm-amdgpu rocm-cmake rocminfo roctracer-dev-api
```
Launch Julia:
```sh
julia
```
and install `AMDGPU.jl`
```julia-repl
] add AMDGPU
build AMDGPU
test AMDGPU
```
Note that testing failed for me.


## The scripts
* [`test_amdgpu.jl`](scripts/test_amdgpu.jl) The [Quick Start example](https://amdgpu.juliagpu.org/stable/quickstart/) from AMDGPU.jl GitHub's
* [`memcopy3D_amdgpu.jl`](scripts/memcopy3D_amdgpu.jl) The memcopy tool (adapted from [here](https://github.com/luraess/parallel-gpu-workshop-JuliaCon21/blob/main/extras/memcopy3D.jl))
* [`diffusion_2D_damp_perf_amdgpu.jl`](scripts/diffusion_2D_damp_perf_amdgpu.jl) The 2D diffusion solver [from JuliaCon21](https://github.com/luraess/parallel-gpu-workshop-JuliaCon21#gpu-implementation) workshop that delivers 92% of T_peak on Nvidia Tesla V100 PCIe 16GB GPUs.
* [`diffusion_2D_perf_amdgpu.jl`](scripts/diffusion_2D_perf_amdgpu.jl) Another "simpler" 2D diffusion script

## Results
### The Quick Start example pass

### The memcopy3D produce following output (tested on 2 different GPUs):

On Vega 10
```
Vega 10 XT [Radeon PRO WX 9100]
T_peak_vendor 484 GB/s
ault08: time_s=1.75 T_eff=330.86 (68% of T_peak_vendor)
```

On Vega 20
```
Vega 20 WKS GL-XE [Radeon Pro VII]
T_peak_vendor 1024 GB/s
ault20: time_s=0.79 T_eff=726.34 (70% of T_peak_vendor)
```
> Note that the results on the Radeon VII seem to be in-line with the results reported [in the Julia BabelStream bench](https://github.com/UoB-HPC/BabelStream/pull/106#issuecomment-897621652).

### Diffusion 2D

WIP 🚧

The 2D diffusion code runs at `120GB/s` (bad) on Vega 10 and I need to test if the the ouput is correct.

## Current issues/challenges
- 30% difference between measured `T_peak` and vendor announced `T_peak_vendor`. On Nvidia Tesla V100 GPU, the difference is about 7% only.
- Very low perf of current naive AMD version [`diffusion_2D_damp_perf_amdgpu.jl`](scripts/diffusion_2D_damp_perf_amdgpu.jl). The Nvidia counterpart runs at 92% of `T_peak`
