using AMDGPU

function main()
    nx,ny = 3,4
    buf   = AMDGPU.Mem.alloc(nx*ny*sizeof(Float64); coherent=true)
    AMDGPU.Mem.set!(buf,UInt32(0),nx*ny*sizeof(Float64))
    A_h   = unsafe_wrap(Array{Float64,2},Ptr{Float64}(buf.ptr),(nx,ny))
    A_d   = AMDGPU.ROCArray{Float64,2}(buf,(nx,ny))
    return A_h,A_d
end

A_h,A_d = main()