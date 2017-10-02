#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include "cuda_initialise.h"
#include "cudnn_initialise.h"
#include "cublas_initialise.h"

#include "cudnn_tensor.h"
#include "cudnn_tensor_operation.h"
#include "cudnn_matrix_subtraction.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <array>

int main(int argc, char** argv)
{
	cuda_initialise		cuda;
	cudnn_initialise	cudnn;

	// cudnn matrix subtraction (3.0f - 1.0f == 2.0f)
	{
		cudnn_tensor<float>		A(cudnn, { 1, 1, 8, 4 }, 3.0f);
		cudnn_tensor<float>		B(cudnn, { 1, 1, 8, 4 }, 1.0f);
		cudnn_tensor<float>		C(cudnn, { 1, 1, 8, 4 }, 0.0f);
		std::vector<float>		h_C;

		cudnn_matrix_subtraction<float>		sub(cudnn);
		sub.apply(A, B, C);
		C.get(h_C);

		auto add_correct = std::all_of(
			std::begin(h_C),
			std::end(h_C),
			[](const float x) { return x == 2.0f; });

		if (add_correct == false)
		{
			return 1;
		}
	}

	// cudnn matrix subtraction (0.0f - 4.0f == -4.0f)
	{
		cudnn_tensor<float>		A(cudnn, { 1, 1, 8, 4 }, 0.0f);
		cudnn_tensor<float>		B(cudnn, { 1, 1, 8, 4 }, 4.0f);
		cudnn_tensor<float>		C(cudnn, { 1, 1, 8, 4 }, 0.0f);
		std::vector<float>		h_C;

		cudnn_matrix_subtraction<float>		sub(cudnn);
		sub.apply(A, B, C);
		C.get(h_C);

		auto add_correct = std::all_of(
			std::begin(h_C),
			std::end(h_C),
			[](const float x) { return x == -4.0f; });

		if (add_correct == false)
		{
			return 2;
		}
	}

	return 0;
}