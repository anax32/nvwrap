#include <iostream>
#include <map>

#include <cuda.h>
#include <cudnn.h>

#include "cuda_initialise.h"
#include "cudnn_initialise.h"

int main(int argc, char** argv)
{
	cuda_initialise	cuda;
	cudnn_initialise cudnn;

	if (cuda.result () != CUDA_SUCCESS)
	{
		return 1;
	}

	if (cudnn.result() != CUDNN_STATUS_SUCCESS)
	{
		return 2;
	}

	return 0;
}