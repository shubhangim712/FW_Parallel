//version 8.8

#include <unordered_set>
#include <vector>
#include <algorithm>
#include <iostream>
#include <utility>
#include "ECLgraph.h"
#include <cuda.h>
#include <sys/time.h>
#include <limits>

static const int warpsize = 32;
static const int ThreadsPerBlock = warpsize * warpsize;

//initialize adj matrix
static __global__ void init1(const int nodes, float* AdjMat, const int upper)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int i = idx / upper;
  if (i < upper) {
    const int j = idx % upper;
    AdjMat[idx] = ((i == j) && (i < nodes)) ? 0 : INFINITY;
  }
}

//add edges to adj matrix
static __global__ void init2(const ECLgraph g, float* AdjMat, const int upper)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < g.nodes) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      const int nei = g.nlist[j];
      AdjMat[i * upper + nei] = g.eweight[j];
    }
  }
}

static __global__ __launch_bounds__(ThreadsPerBlock, 1)
void FW0(float* AdjMat, const int nodes, float* krows, float* kcols)
{
  __shared__ float temp[warpsize * warpsize];

  const int warp = threadIdx.x / warpsize;
  const int i = warp;
  const int lane = threadIdx.x % warpsize;
  const int j = lane;

  float ij = AdjMat[i * nodes + j];
  for (int k = 0; k < warpsize; ++k) {
    const float ik = __shfl_sync(~(-2 << (warpsize - 1)), ij, k);
    __shared__ float krow[warpsize];
    if (i == k) krow[lane] = ij;
    __syncthreads();
    ij = min(ij, ik + krow[lane]);
    __syncthreads();
    if (i == k) krows[i * nodes + j] = ij;
    if (j == k) temp[lane * warpsize + warp] = ij;
  }
  __syncthreads();
  kcols[i * nodes + j] = temp[warp * warpsize + lane];
  AdjMat[i * nodes + j] = ij;
}

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void FWrowcol(float* AdjMat, const int nodes, float* krows, float* kcols, const int x, const int sub)
{
  if (blockIdx.x < sub) {
    const int y = blockIdx.x;
    if (x != y) {
      const int warp = threadIdx.x / warpsize;
      const int i = warp + x * warpsize;
      const int lane = threadIdx.x % warpsize;
      const int j = lane + y * warpsize;

      __shared__ float temp[warpsize * warpsize];

      temp[warp * warpsize + lane] = kcols[i * nodes + lane + x * warpsize];
      __syncthreads();
      float ij = AdjMat[i * nodes + j];
      for (int k = x * warpsize; k < (x + 1) * warpsize; ++k) {
        const float ik = temp[(k % warpsize) * warpsize + warp];
        __shared__ float krow[warpsize];
        if (i == k) krow[lane] = ij;
        __syncthreads();
        ij = min(ij, ik + krow[lane]);
        __syncthreads();
        if (i == k) krows[i * nodes + j] = ij;
      }
      AdjMat[i * nodes + j] = ij;
    }
  }
  else {
    const int y = blockIdx.x - sub;
    if (x != y) {
      const int warp = threadIdx.x / warpsize;
      const int i = warp + y * warpsize;
      const int lane = threadIdx.x % warpsize;
      const int j = lane + x * warpsize;

      __shared__ float temp[warpsize * warpsize];

      float ij = AdjMat[i * nodes + j];
      for (int k = x * warpsize; k < (x + 1) * warpsize; ++k) {
        const float ik = __shfl_sync(~(-2 << (warpsize - 1)), ij, k % warpsize);
        ij = min(ij, ik + krows[k * nodes + j]);
        if (j == k) temp[lane * warpsize + warp] = ij;
      }
      __syncthreads();
      kcols[i * nodes + j] = temp[warp * warpsize + lane];
      AdjMat[i * nodes + j] = ij;
    }
  }
}

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void FWrem(float* AdjMat, const int nodes, float* krows, float* kcols, const int x, const int sub)
{
  const int y = blockIdx.x / sub;
  if (x != y) {
    const int z = blockIdx.x % sub;

    __shared__ float temp[warpsize * warpsize];

    const int warp = threadIdx.x / warpsize;
    const int i = warp + y * warpsize;
    const int lane = threadIdx.x % warpsize;
    const int j = lane + z * warpsize;

    temp[warp * warpsize + lane] = kcols[i * nodes + lane + x * warpsize];
    float ij = AdjMat[i * nodes + j];
    __syncthreads();
    /*
    for (int k = x * warpsize; k < (x + 1) * warpsize; ++k) {
      float ik = temp[(k % warpsize) * warpsize + warp];
      ij = min(ij, ik + krows[k * nodes + j]);
    }
    */
    int idx1 = warp;
    int idx2 = (x * warpsize) * nodes + j;
    #pragma unroll 32
    for (int k1 = 0; k1 < warpsize; k1++) {
      const float ik = temp[idx1];
      ij = min(ij, ik + krows[idx2]);
      idx1 += warpsize;
      idx2 += nodes;
    }

    if ((y == z) && (y == x + 1) && (x != sub - 1)) {
      __shared__ float temp2[warpsize * warpsize];
      for (int k = y * warpsize; k < (y + 1) * warpsize; ++k) {
        const float ik = __shfl_sync(~(-2 << (warpsize - 1)), ij, k % warpsize);
        __shared__ float krow[warpsize];
        if (i == k) krow[lane] = ij;
        __syncthreads();
        ij = min(ij, ik + krow[lane]);
        __syncthreads();
        if (i == k) krows[i * nodes + j] = ij;
        if (j == k) temp2[lane * warpsize + warp] = ij;
      }
      __syncthreads();
      kcols[i * nodes + j] = temp2[warp * warpsize + lane];
    }

    AdjMat[i * nodes + j] = ij;
  }
}

static void CheckCuda()
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d: %s\n", e, cudaGetErrorString(e));
    exit(-1);
  }
}

static void FW_cuda(ECLgraph g, float* AdjMat)
{
  const int sub = (g.nodes + warpsize - 1) / warpsize;
  const int upper = sub * warpsize;
  float* d_AdjMat;
  if (cudaSuccess != cudaMalloc((void **)&d_AdjMat, sizeof(float) * upper * upper)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}

  ECLgraph d_g = g;
  cudaMalloc((void **)&d_g.nindex, sizeof(int) * (g.nodes + 1));
  cudaMalloc((void **)&d_g.nlist, sizeof(int) * g.edges);
  cudaMalloc((void **)&d_g.eweight, sizeof(int) * g.edges);
  cudaMemcpy(d_g.nindex, g.nindex, sizeof(int) * (g.nodes + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_g.nlist, g.nlist, sizeof(int) * g.edges, cudaMemcpyHostToDevice);
  cudaMemcpy(d_g.eweight, g.eweight, sizeof(int) * g.edges, cudaMemcpyHostToDevice);

  timeval start, end;
  gettimeofday(&start, NULL);
  init1<<<(upper * upper + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(g.nodes, d_AdjMat, upper);
  init2<<<(g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_g, d_AdjMat, upper);
  cudaDeviceSynchronize();
  gettimeofday(&end, NULL);
  const double inittime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("time to initialize (parallel): %.5f s\n", inittime);
  CheckCuda();

  float* d_krows;
  if (cudaSuccess != cudaMalloc((void **)&d_krows, sizeof(float) * upper * upper)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  float* d_kcols;
  if (cudaSuccess != cudaMalloc((void **)&d_kcols, sizeof(float) * upper * upper)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}

  gettimeofday(&start, NULL);

  FW0<<<1, warpsize * warpsize>>>(d_AdjMat, upper, d_krows, d_kcols);
  for (int x = 0; x < sub; ++x) {
    FWrowcol<<<2 * sub, warpsize * warpsize>>>(d_AdjMat, upper, d_krows, d_kcols, x, sub);
    FWrem<<<sub * sub, warpsize * warpsize>>>(d_AdjMat, upper, d_krows, d_kcols, x, sub);
  }

  cudaDeviceSynchronize();
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time (parallel): %.5f s\n", runtime);
  CheckCuda();

  if (cudaSuccess != cudaMemcpy(AdjMat, d_AdjMat, sizeof(float) * upper * upper, cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying from device failed\n"); exit(-1);}

  cudaFree(d_g.nindex);
  cudaFree(d_g.nlist);
  cudaFree(d_g.eweight);
  cudaFree(d_AdjMat);
  cudaFree(d_krows);
  cudaFree(d_kcols);
}
