#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include "cuda_initialise.h"
#include "cudnn_initialise.h"
#include "cublas_initialise.h"
#include "cudnn_tensor.h"
#include "cudnn_tensor_operation.h"
#include "cudnn_matrix_hadamard.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <array>

// FIXME: cudnn OpTensor multiplication does not work

int main(int argc, char** argv)
{
	cuda_initialise		cuda;
	cudnn_initialise	cudnn;

#if 0
	// there is no cublas hadamard product
	{
		cublas_initialise	cublas;

		// cublas hadamard product of square matrices
		{
			cublas_matrix_hadamard		had (cublas);
			cudnn_tensor<float>			A(cudnn, { 1, 1, 2, 2 }, 1.0f);
			cudnn_tensor<float>			B(cudnn, { 1, 1, 2, 2 }, 2.0f);
			cudnn_tensor<float>			C(cudnn, { 1, 1, 2, 2 }, 0.0f);
			std::vector<float>			h_C;

			had.apply<float>(A, B, C);
			C.get(h_C);

			auto square_mult_correct = std::all_of(
				std::begin(h_C),
				std::end(h_C),
				[](const float x) {return x == 2.0f; });

			if (square_mult_correct == false)
			{
				return 1;
			}
		}

		// cublas hadamard product of non-square matrices
		{
			cublas_matrix_hadamard		had (cublas);
			cudnn_tensor<float>			A(cudnn, { 1, 1, 3, 4 }, 1.0f);
			cudnn_tensor<float>			B(cudnn, { 1, 1, 3, 4 }, 2.0f);
			cudnn_tensor<float>			C(cudnn, { 1, 1, 3, 4 }, 0.0f);
			std::vector<float>			h_C;

			had.apply<float>(A, B, C);
			C.get(h_C);

			auto wonky_mult_correct = std::all_of(
				std::begin(h_C),
				std::end(h_C),
				[](const float x) { return x == 2.0f; });

			if (wonky_mult_correct == false)
			{
				return 2;
			}
		}
	}
#endif

	// cudnn matrix multiplication using cudnnOpTensor
	{
		// cudnn matrix multiplication of square matrices
		{
			cudnn_matrix_hadamard<float>		had(cudnn);
			cudnn_tensor<float>			A(cudnn, { 1, 1, 2, 2 }, 1.0f);
			cudnn_tensor<float>			B(cudnn, { 1, 1, 2, 2 }, 1.0f);
			cudnn_tensor<float>			C(cudnn, { 1, 1, 2, 2 }, 0.0f);
			std::vector<float>			h_C;

			had.apply(A, B, C);
			C.get(h_C);

			auto square_mult_correct = std::all_of(
				std::begin(h_C),
				std::end(h_C),
				[](const float x) {return x == 1.0f; });

			if (square_mult_correct == false)
			{
				return 1;
			}
		}

		// cudnn matrix multiplcation of non-square matrices
		{
			cudnn_matrix_hadamard<float>	had(cudnn);
			cudnn_tensor<float>		A(cudnn, { 1, 1, 3, 4 }, 1.0f);
			cudnn_tensor<float>		B(cudnn, { 1, 1, 3, 4 }, 2.0f);
			cudnn_tensor<float>		C(cudnn, { 1, 1, 3, 4 }, 0.0f);
			std::vector<float>		h_C;

			had.apply(A, B, C);
			C.get(h_C);

			auto wonky_mult_correct = std::all_of(
				std::begin(h_C),
				std::end(h_C),
				[](const float x) { return x == 2.0f; });

			if (wonky_mult_correct == false)
			{
				return 2;
			}
		}
	}

	return 0;
}