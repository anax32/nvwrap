#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include "cuda_initialise.h"
#include "cudnn_initialise.h"
#include "cublas_initialise.h"
#include "cudnn_tensor.h"
#include "cublas_matrix_transpose.h"
#include "cudnn_matrix_transpose.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <array>

int main(int argc, char** argv)
{
	cuda_initialise		cuda;
	cudnn_initialise	cudnn;
	cublas_initialise	cublas;

	// cublas matrix transpose unit value matrix
	{
		cudnn_tensor<float>		A(cudnn, { 1, 1, 6, 3 }, 1.0f);
		cudnn_tensor<float>		B(cudnn, { 1, 1, 3, 6 }, 0.0f);
		std::vector<float>		h_B;

		cublas_matrix_transpose	transpose(cublas);
		transpose.apply(A, B);
		B.get(h_B);

		auto transpose_correct = std::all_of(
			std::begin(h_B),
			std::end(h_B),
			[](const float x) {return x == 1.0f; });

		if (transpose_correct == false)
		{
			return 1;
		}
	}

	// cublas matrix transpose ascending value matrix
	{
		int i = 0;
		auto counter = [&i]() mutable {return i++; };
		cudnn_tensor<float>		A(cudnn, { 1,1,3,3 }, counter);
		cudnn_tensor<float>		B(cudnn, { 1,1,3,3 }, 0.0f);
		std::vector<float>		h_A, h_B;
		std::vector<float>		h_exp{0,3,6,1,4,7,2,5,8 };

		// check preconditions are valid
		A.get(h_A);
		B.get(h_B);

		i = 0;
		auto A_is_incrementing = std::all_of(
			std::begin(h_A),
			std::end(h_A),
			[&counter](const float x) {return x == counter(); });
		auto B_is_zero = std::all_of(
			std::begin(h_B),
			std::begin(h_B),
			[](const float x) {return x == 0.0f; });

		if ((A_is_incrementing == false) ||
			(B_is_zero == false))
		{
			return 2;
		}

		cublas_matrix_transpose	transpose(cublas);
		transpose.apply(A, B);

		// check post-conditions
		A.get(h_A);
		B.get(h_B);

		i = 0;
		auto A_still_incrementing = std::all_of(
			std::begin(h_A),
			std::end(h_A),
			[&counter](const float x) {return x == counter(); });
		auto B_is_exp = std::equal(
			std::begin(h_B),
			std::end(h_B),
			std::begin(h_exp));

		if ((A_still_incrementing == false) ||
			(B_is_exp == false))
		{
			return 3;
		}
	}

	// cudnn matrix transpose unit value matrix
	{
		cudnn_tensor<float>		A(cudnn, { 1, 1, 6, 3 }, 1.0f);
		cudnn_tensor<float>		B(cudnn, { 1, 1, 3, 6 }, 0.0f);
		std::vector<float>		h_B;

		cudnn_matrix_transpose<float>	transpose(cudnn, cublas);
		transpose.apply(A, B);
		B.get(h_B);

		auto transpose_correct = std::all_of(
			std::begin(h_B),
			std::end(h_B),
			[](const float x) {return x == 1.0f; });

		if (transpose_correct == false)
		{
			return 4;
		}
	}

	// cudnn matrix transpose ascending value matrix
	{
		int i = 0;
		auto counter = [&i]() mutable {return i++; };
		cudnn_tensor<float>		A(cudnn, { 1,1,3,3 }, counter);
		cudnn_tensor<float>		B(cudnn, { 1,1,3,3 }, 0.0f);
		std::vector<float>		h_A, h_B;
		std::vector<float>		h_exp{ 0,3,6,1,4,7,2,5,8 };

		// check preconditions are valid
		A.get(h_A);
		B.get(h_B);

		i = 0;
		auto A_is_incrementing = std::all_of(
			std::begin(h_A),
			std::end(h_A),
			[&counter](const float x) {return x == counter(); });
		auto B_is_zero = std::all_of(
			std::begin(h_B),
			std::begin(h_B),
			[](const float x) {return x == 0.0f; });

		if ((A_is_incrementing == false) ||
			(B_is_zero == false))
		{
			return 5;
		}
	
		cudnn_matrix_transpose<float>	transpose(cudnn, cublas);
		transpose.apply(A, B);

		// check post-conditions
		A.get(h_A);
		B.get(h_B);

		i = 0;
		auto A_still_incrementing = std::all_of(
			std::begin(h_A),
			std::end(h_A),
			[&counter](const float x) {return x == counter(); });
		auto B_is_exp = std::equal(
			std::begin(h_B),
			std::end(h_B),
			std::begin(h_exp));

		if ((A_still_incrementing == false) ||
			(B_is_exp == false))
		{
			return 6;
		}
	}

	return 0;
}