#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include "cuda_initialise.h"
#include "cudnn_initialise.h"
#include "cublas_initialise.h"

#include "cudnn_tensor.h"
#include "cudnn_tensor_operation.h"
#include "cudnn_matrix_addition.h"
#include "cublas_matrix_addition.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <array>

int main(int argc, char** argv)
{
	cuda_initialise		cuda;
	cudnn_initialise	cudnn;

	// cudnn matrix addition
	{
		cudnn_tensor<float>		A(cudnn,  { 1, 1, 8, 4 }, 1.0f);
		cudnn_tensor<float>		B(cudnn, { 1, 1, 8, 4 }, 3.0f);
		cudnn_tensor<float>		C(cudnn, { 1, 1, 8, 4 }, 0.0f);
		std::vector<float>		h_C;

		cudnn_matrix_addition<float>		add(cudnn);
		add.apply(A, B, C);
		C.get(h_C);

		auto add_correct = std::all_of(
			std::begin(h_C),
			std::end(h_C),
			[](const float x) { return x == 4.0f; });

		if (add_correct == false)
		{
			return 1;
		}
	}

	{
		cublas_initialise	cublas;

		// cublas matrix addition
		{
			cudnn_tensor<float>		A(cudnn, { 1, 1, 8, 4 }, 1.0f);
			cudnn_tensor<float>		B(cudnn, { 1, 1, 8, 4 }, 3.0f);
			cudnn_tensor<float>		C(cudnn, { 1, 1, 8, 4 }, 0.0f);
			std::vector<float>		h_C;

			cublas_matrix_addition	add(cublas);
			add.apply(A, B, C);
			C.get(h_C);

			auto add_correct = std::all_of(
				std::begin(h_C),
				std::end(h_C),
				[](const float x) { return x == 4.0f; });

			if (add_correct == false)
			{
				return 1;
			}
		}

		// cublas matrix addition in-place
		{
			cudnn_tensor<float>		A(cudnn, { 1, 1, 8, 4 }, 1.0f);
			cudnn_tensor<float>		B(cudnn, { 1, 1, 8, 4 }, 3.0f);
			std::vector<float>		h_C;

			cublas_matrix_addition	add(cublas);
			add.apply(A, B, A);
			A.get(h_C);

			auto add_correct = std::all_of(
				std::begin(h_C),
				std::end(h_C),
				[](const float x) { return x == 4.0f; });

			if (add_correct == false)
			{
				return 1;
			}
		}
	}

	return 0;
}