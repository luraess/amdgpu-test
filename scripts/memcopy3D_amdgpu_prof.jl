# Memory copy 3D to return T_peak
using AMDGPU, Profile

function copy3D!(T2, T, Ci)
    ix = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x  # only workgroupIdx starts at 1
    iy = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y  # only workgroupIdx starts at 1
    iz = (workgroupIdx().z - 1) * workgroupDim().z + workitemIdx().z  # only workgroupIdx starts at 1
    T2[ix,iy,iz] = T[ix,iy,iz] + Ci[ix,iy,iz]
    return
end

function memcopy3D()
    # Numerics
    nx, ny, nz = 1024, 512, 512                            # Number of grid points in dimensions x, y and z
    TBSize = (32, 8, 1)
    GSize  = (nx, ny, nz)

    nt  = 100                                              # Number of time steps

    # Array initializations
    T   = ROCArray(zeros(Float64, nx, ny, nz))
    T2  = ROCArray(zeros(Float64, nx, ny, nz))
    Ci  = ROCArray(zeros(Float64, nx, ny, nz))

    # Initial conditions
    Ci .= 0.5
    T  .= 1.7
    T2 .= T

    wait( @roc groupsize = TBSize gridsize = GSize copy3D!(T2, T, Ci) )
    
    t_tic = 0.0
    # Time loop
    for it = 1:nt
        if (it == 11) t_tic=time() end  # Start measuring time.
        Profile.@profile wait( @roc groupsize = TBSize gridsize = GSize copy3D!(T2, T, Ci) )
        T, T2 = T2, T
    end
    t_toc=time()-t_tic

    # Performance
    @show A_eff = (2*1+1)*1/1e9*nx*ny*nz*sizeof(Float64)      # Effective main memory access per iteration [GB] (Lower bound of required memory access: T has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
    t_it  = t_toc/(nt-10)                               # Execution time per iteration [s]
    T_eff = A_eff/t_it                                  # Effective memory throughput [GB/s]
    println("time_s=$t_toc T_eff=$T_eff")
end

memcopy3D()
# Profile.@profile memcopy3D()
Profile.print()
