// Taken from Bonsai -- A GPU gravitational [BH]-tree code
// Copyright [2010-2012] Jeroen Bédorf bedorf@strw.leidenuniv.nl Evghenii Gaburov egaburov.work@gmail.com
// Licensed under the Apache License, Version 2.0.


// Inline PTX to retrieve lane, warp
__device__ __forceinline__ unsigned int __laneid() { unsigned int laneid; asm volatile ("mov.u32 %0, %laneid;" : "=r"(laneid)); return laneid; }
__device__ __forceinline__ unsigned int __warpid() { unsigned int warpid; asm volatile ("mov.u32 %0, %warpid;" : "=r"(warpid)); return warpid; }

// Inline PTX to return number of leading zeroes (from MSB) in a unsigned int
__device__ __forceinline__ unsigned int __clz(unsigned int word) { unsigned int ret; asm volatile ("clz.b32 %0, %1;" : "=r"(ret) : "r"(word)); return ret; }
__device__ __forceinline__ unsigned int __clz(unsigned long long dword) { unsigned int ret; asm volatile ("clz.b64 %0, %1;" : "=r"(ret) : "l"(dword)); return ret; }
