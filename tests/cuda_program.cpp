#include <cuda.h>
#include <cuda_runtime_api.h>

#include "cuda_initialise.h"
#include "cuda_kernel_ptx.h"
#include "cuda_kernel_runtime.h"

int main (int argc, char** argv)
{
  cuda_initialise   cuda;

  // loading, compiling and executing kernel at runtime
  {
      // simple kernel to compile and run
      const char *cuda_kernel_src = "                                       \n\
      extern \"C\" __global__ void test_fn (int *a, int *b)                 \n\
      {                                                                     \n\
          *a = 11;                                                          \n\
          *b = 12;                                                          \n\
          return;                                                           \n\
      }                                                                     \n";

      cuda_kernel_runtime       k (cuda_kernel_src, { "test_fn" });

      if (k == false)
      {
          return 1;
      }

      // allocate some memory on the device and copy data into it
      int               h_A = 2;
      int               h_B = 3;
      CUdeviceptr       d_A, d_B;

      cuMemAlloc (&d_A, sizeof(int));
      cuMemAlloc (&d_B, sizeof(int));;
      cuMemcpyHtoD (d_A, &h_A, sizeof(int));
      cuMemcpyHtoD (d_B, &h_B, sizeof(int));
      void *args[] = { &d_A, &d_B, };

      // run the kernel
      k.apply ({ d_A, d_B }, { 1 });

      if (k == false)
      {
          return 2;
      }

      // copy the device variables back into host memory
      cuMemcpyDtoH (&h_A, d_A, sizeof(int));
      cuMemcpyDtoH (&h_B, d_B, sizeof(int));

      cuMemFree (d_A);
      cuMemFree (d_B);

      auto result_match = ((h_A == 11) && (h_B == 12));

      if (result_match == false)
      {
          return 3;
      }
  }

  // loading precompiled ptx and executing kernel at runtime
  {
      cuda_kernel_ptx   ptx_k ("..\\tests\\shdr\\test.ptx\0", { "test_fn_ptx" });

      // allocate some memory on the device and copy data into it
      int               h_A = 2;
      int               h_B = 3;
      CUdeviceptr       d_A, d_B;

      cuMemAlloc (&d_A, sizeof(int));
      cuMemAlloc (&d_B, sizeof(int));;
      cuMemcpyHtoD (d_A, &h_A, sizeof(int));
      cuMemcpyHtoD (d_B, &h_B, sizeof(int));
      void *args[] = { &d_A, &d_B, };

      // run the kernel
      ptx_k.apply ({ d_A, d_B }, { 1 });

      if (ptx_k == false)
      {
          return 4;
      }

      // copy the device variables back into host memory
      cuMemcpyDtoH (&h_A, d_A, sizeof(int));
      cuMemcpyDtoH (&h_B, d_B, sizeof(int));

      cuMemFree (d_A);
      cuMemFree (d_B);

      auto result_match = ((h_A == 11) && (h_B == 21));

      if (result_match == false)
      {
          return 5;
      }
  }

	return 0;
}