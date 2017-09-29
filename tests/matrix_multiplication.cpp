#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include "cuda_initialise.h"
#include "cudnn_initialise.h"
#include "cublas_initialise.h"
#include "cudnn_tensor.h"
#include "cublas_matrix_multiply.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <array>

int main(int argc, char** argv)
{
	cuda_initialise		cuda;
	cudnn_initialise	cudnn;
	
	// cublas matrix multiplication using cublasSgemm
	{
		cublas_initialise	cublas;

		// cublas matrix multiplication of square matrices
		{
			cublas_matrix_multiply		matmul(cublas);
			cudnn_tensor<float>			A(cudnn, { 1, 1, 2, 2 }, 1.0f);
			cudnn_tensor<float>			B(cudnn, { 1, 1, 2, 2 }, 1.0f);
			cudnn_tensor<float>			C(cudnn, matmul.get_output_size(A.dimensions(), B.dimensions()), 0.0f);
			std::vector<float>			h_C;

			matmul.apply<float>(A, B, C);
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

		// cublas matrix multiplcation of non-square matrices
		{
			cublas_matrix_multiply		matmul(cublas);
			cudnn_tensor<float>			A(cudnn, { 1, 1, 4, 2 }, 1.0f);
			cudnn_tensor<float>			B(cudnn, { 1, 1, 2, 3 }, 1.0f);
			cudnn_tensor<float>			C(cudnn, matmul.get_output_size(A.dimensions(), B.dimensions()), 0.0f);
			std::vector<float>			h_C;

			matmul.apply<float>(A, B, C);
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