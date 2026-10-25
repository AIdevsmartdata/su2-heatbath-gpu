// hip2cuda.h — mechanical HIP -> CUDA mapping used by the CUDA port of su2_hip.cpp.
// No semantic change: every HIP call used by su2_hip.cpp has a direct CUDA counterpart;
// hipLaunchKernelGGL(K, grid, block, shmem, stream, args...) -> K<<<grid, block, shmem, stream>>>(args...).
#pragma once
#include <cuda_runtime.h>
#define hipError_t cudaError_t
#define hipSuccess cudaSuccess
#define hipGetErrorString cudaGetErrorString
#define hipMalloc cudaMalloc
#define hipMemset cudaMemset
#define hipMemcpy cudaMemcpy
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipFree cudaFree
#define hipEvent_t cudaEvent_t
#define hipEventCreate cudaEventCreate
#define hipEventDestroy cudaEventDestroy
#define hipEventRecord cudaEventRecord
#define hipEventSynchronize cudaEventSynchronize
#define hipEventElapsedTime cudaEventElapsedTime
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipGetLastError cudaGetLastError
#define hipGetDeviceCount cudaGetDeviceCount
#define hipDeviceProp_t cudaDeviceProp
#define hipGetDeviceProperties cudaGetDeviceProperties
#define hipLaunchKernelGGL(K, G, B, S, ST, ...) K<<<(G), (B), (S), (ST)>>>(__VA_ARGS__)
