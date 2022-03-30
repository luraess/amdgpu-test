using AMDGPU

function main()
    Dst   = zeros(Float64,32)
    Dst_w = unsafe_wrap(AMDGPU.ROCArray,pointer(Dst),size(Dst))
    Src   = AMDGPU.rand(Float64,size(Dst))

    srcPos    = (1,1,1)
    dstPos    = (1,1,1)
    srcPitch  = Base.elsize(Src)*size(Src,1)
    dstPitch  = Base.elsize(Src)*size(Dst,1)
    srcSlice  = Base.elsize(Src)*size(Src,1)
    dstSlice  = Base.elsize(Src)*size(Dst,1)
    width     = Base.elsize(Src)*size(Src,1)
    srcOffset = ((srcPos[1]-1)*Base.elsize(Src), srcPos[2]-1, srcPos[3]-1)
    dstOffset = ((dstPos[1]-1)*Base.elsize(Src), dstPos[2]-1, dstPos[3]-1)

    hsaCopyDir = AMDGPU.HSA.LibHSARuntime.hsaDeviceToHost

    srcPtr       = Base.unsafe_convert(Ptr{AMDGPU.HSA.PitchedPtr}, Ref(AMDGPU.HSA.PitchedPtr(pointer(Src), srcPitch, srcSlice)))
    dstPtr       = Base.unsafe_convert(Ptr{AMDGPU.HSA.PitchedPtr}, Ref(AMDGPU.HSA.PitchedPtr(pointer(Dst_w), dstPitch, dstSlice)))
    srcOffsetPtr = Base.unsafe_convert(Ptr{AMDGPU.HSA.Dim3},       Ref(AMDGPU.HSA.Dim3(srcOffset...)))
    dstOffsetPtr = Base.unsafe_convert(Ptr{AMDGPU.HSA.Dim3},       Ref(AMDGPU.HSA.Dim3(dstOffset...)))
    rangePtr     = Base.unsafe_convert(Ptr{AMDGPU.HSA.Dim3},       Ref(AMDGPU.HSA.Dim3(width,size(Src,2),size(Src,3))))

    signal = HSASignal(1)
    AMDGPU.HSA.amd_memory_async_copy_rect(dstPtr,dstOffsetPtr,srcPtr,srcOffsetPtr,rangePtr,
                                        get_default_agent().agent,hsaCopyDir,UInt32(0),C_NULL,signal.signal[]) |> AMDGPU.check
    wait(signal)
    @assert all(Dst .== Array(Src))
    return
end

main()
