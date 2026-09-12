#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

struct Job { uint32_t words[13]; uint32_t difficulty; uint64_t start; uint64_t stride; };

__device__ __constant__ uint32_t K[64] = {
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__device__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
__device__ __forceinline__ uint32_t ch(uint32_t x,uint32_t y,uint32_t z) { return (x&y) ^ (~x&z); }
__device__ __forceinline__ uint32_t maj(uint32_t x,uint32_t y,uint32_t z) { return (x&y) ^ (x&z) ^ (y&z); }
__device__ __forceinline__ uint32_t e0(uint32_t x) { return rotr(x,2)^rotr(x,13)^rotr(x,22); }
__device__ __forceinline__ uint32_t e1(uint32_t x) { return rotr(x,6)^rotr(x,11)^rotr(x,25); }
__device__ __forceinline__ uint32_t s0(uint32_t x) { return rotr(x,7)^rotr(x,18)^(x>>3); }
__device__ __forceinline__ uint32_t s1(uint32_t x) { return rotr(x,17)^rotr(x,19)^(x>>10); }

__device__ void compress(uint32_t st[8], const uint32_t block[16]) {
  uint32_t w[64];
  #pragma unroll
  for (int i=0;i<16;i++) w[i]=block[i];
  for (int i=16;i<64;i++) w[i]=s1(w[i-2])+w[i-7]+s0(w[i-15])+w[i-16];
  uint32_t a=st[0],b=st[1],c=st[2],d=st[3],e=st[4],f=st[5],g=st[6],h=st[7];
  #pragma unroll
  for (int i=0;i<64;i++) { uint32_t t1=h+e1(e)+ch(e,f,g)+K[i]+w[i]; uint32_t t2=e0(a)+maj(a,b,c); h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2; }
  st[0]+=a;st[1]+=b;st[2]+=c;st[3]+=d;st[4]+=e;st[5]+=f;st[6]+=g;st[7]+=h;
}

__device__ void hash_job(const Job &job, uint64_t nonce, uint32_t out[8]) {
  uint32_t first[16]={0};
  #pragma unroll
  for (int i=0;i<5;i++) first[i]=job.words[i];
  for (int i=5;i<8;i++) first[i+8]=job.words[i];
  first[11]=(uint32_t)(nonce >> 32); first[12]=(uint32_t)nonce;
  uint32_t st[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  compress(st,first);
  uint32_t second[16]={0};
  second[0]=job.words[8]; second[1]=job.words[9]; second[2]=job.words[10]; second[3]=job.words[11]; second[4]=job.words[12];
  second[5]=0x80000000; second[15]=672;
  compress(st,second);
  #pragma unroll
  for (int i=0;i<8;i++) out[i]=st[i];
}

__device__ bool meets(const uint32_t h[8], uint32_t bits) {
  uint32_t full=bits/32, rem=bits%32;
  for (uint32_t i=0;i<full;i++) if (h[i]!=0) return false;
  return rem==0 || (h[full] >> (32-rem)) == 0;
}

__global__ void mine_kernel(Job job, unsigned long long *found, unsigned long long *count) {
  uint64_t id=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
  uint64_t nonce=job.start+id*job.stride;
  uint32_t h[8]; hash_job(job,nonce,h);
  atomicAdd(count,1ULL);
  if (meets(h,job.difficulty)) atomicCAS(found,0xffffffffffffffffULL,nonce);
}

static uint32_t word(const char *p) { return (uint32_t)strtoul(p,NULL,16); }

int main(int argc,char **argv) {
  if (argc != 5) { fprintf(stderr,"usage: hashbroker_cuda <device> <difficulty> <start> <stride>\n"); return 2; }
  int device=atoi(argv[1]); Job job{}; job.difficulty=(uint32_t)atoi(argv[2]); job.start=strtoull(argv[3],NULL,10); job.stride=strtoull(argv[4],NULL,10);
  if (job.difficulty == 0 || job.difficulty > 256 || job.stride == 0) { fprintf(stderr,"invalid difficulty or stride\n"); return 2; }
  // Job words are supplied as 13 hexadecimal words: address(5), challenge(8).
  for (int i=0;i<13;i++) { char buf[32]; if (scanf("%31s",buf)!=1) return 3; job.words[i]=word(buf); }
  cudaError_t err = cudaSetDevice(device);
  if (err != cudaSuccess) { fprintf(stderr,"cudaSetDevice failed: %s\n", cudaGetErrorString(err)); return 4; }
  cudaDeviceProp props{};
  if ((err = cudaGetDeviceProperties(&props, device)) != cudaSuccess) { fprintf(stderr,"cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err)); return 4; }
  fprintf(stderr,"CUDA device %d: %s (compute %d.%d)\n", device, props.name, props.major, props.minor);
  unsigned long long *dfound,*dcount;
  if ((err = cudaMalloc(&dfound,8)) != cudaSuccess || (err = cudaMalloc(&dcount,8)) != cudaSuccess) {
    fprintf(stderr,"cudaMalloc failed: %s\n", cudaGetErrorString(err)); return 5;
  }
  unsigned long long missing=0xffffffffffffffffULL, zero=0;
  cudaMemcpy(dfound,&missing,8,cudaMemcpyHostToDevice); cudaMemcpy(dcount,&zero,8,cudaMemcpyHostToDevice);
  const int threads=256, blocks=4096; const uint64_t batch=(uint64_t)threads*blocks;
  while (true) {
    mine_kernel<<<blocks,threads>>>(job,dfound,dcount);
    if ((err = cudaGetLastError()) != cudaSuccess || (err = cudaDeviceSynchronize()) != cudaSuccess) { fprintf(stderr,"CUDA kernel failed: %s\n", cudaGetErrorString(err)); return 6; }
    unsigned long long found; cudaMemcpy(&found,dfound,8,cudaMemcpyDeviceToHost);
    unsigned long long count; cudaMemcpy(&count,dcount,8,cudaMemcpyDeviceToHost);
    if(found!=missing){printf("FOUND %llu\n",found); fflush(stdout); printf("HASHES %llu\n",count); fflush(stdout); break;}
    printf("PROGRESS %llu\n",count); fflush(stdout);
    job.start += batch * job.stride;
  }
  cudaFree(dfound);cudaFree(dcount); return 0;
}
