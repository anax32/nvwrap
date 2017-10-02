#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include "cuda_initialise.h"
#include "cudnn_initialise.h"
#include "cublas_initialise.h"
#include "cudnn_tensor.h"
#include "cudnn_matrix_scale.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <array>

int main(int argc, char** argv)
{
	cuda_initialise		cuda;
	cudnn_initialise	cudnn;
	cublas_initialise	cublas;

	{
		cudnn_matrix_scale<float>	scale(cudnn);
		cudnn_tensor<float>			A(cudnn, { 1,1,3,4 }, 1.0f);
		std::vector<float>			h_A;

		// preconditions
		A.get(h_A);

		auto all_values_are_one = std::all_of(
			std::begin(h_A),
			std::end(h_A),
			[](const float x) {return x == 1.0f; });

		scale.apply(A, 3.0f);

		// postconditions
		A.get(h_A);

		auto all_values_are_three = std::all_of(
			std::begin(h_A),
			std::end(h_A),
			[](const float x) {return x == 3.0f; });

		if ((all_values_are_one == false) ||
			(all_values_are_three == false))
		{
			return 1;
		}
	}

	return 0;
}