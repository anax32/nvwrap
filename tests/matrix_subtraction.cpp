#include <cuda.h>
#include <cudnn.h>

#include <iostream>

#include "cuda_initialise.h"
#include "cuda_matrix_subtraction.h"

#include "cudnn_initialise.h"
#include "cudnn_tensor.h"

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
		std::vector<float>		h_A, h_B, h_C;

		A.get(h_A);
		B.get(h_B);
		C.get(h_C);

		cuda_matrix_subtraction<float>		sub(cudnn);
		sub.apply(A, B, C);

		A.get(h_A);
		B.get(h_B);
		C.get(h_C);

		auto sub_correct = std::all_of(
			std::begin(h_C),
			std::end(h_C),
			[](const float x) { return x == 2.0f; });

		if (sub_correct == false)
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

		cuda_matrix_subtraction<float>		sub(cudnn);
		sub.apply(A, B, C);
		C.get(h_C);

		auto sub_correct = std::all_of(
			std::begin(h_C),
			std::end(h_C),
			[](const float x) { return x == -4.0f; });

		if (sub_correct == false)
		{
			return 2;
		}
	}

	// cudnn matrix subtraction (counter - counter == 0.0f)
	{
		auto counter = [i = 0.0f]() mutable {return i+=1.0f; };

		cudnn_tensor<float>		A(cudnn, { 1, 1, 8, 4 }, counter);
		cudnn_tensor<float>		B(cudnn, { 1, 1, 8, 4 }, counter);
		cudnn_tensor<float>		C(cudnn, { 1, 1, 8, 4 }, 0.0f);
		std::vector<float>		h_C;

		cuda_matrix_subtraction<float>		sub(cudnn);
		sub.apply(A, B, C);
		C.get(h_C);

		auto sub_correct = std::all_of(
			std::begin(h_C),
			std::end(h_C),
			[](const float x) { return x == 0.0f; });

		if (sub_correct == false)
		{
			return 3;
		}
	}

	// cudnn matrix subtraction (counter - counter == 0.0f)
	{
		auto counter = [i = 0]() mutable {return i++; };

		cudnn_tensor<float>		A(cudnn, { 1, 1, 1, 4 }, 0.0f);
		cudnn_tensor<float>		B(cudnn, { 1, 1, 1, 4 }, 0.0f);
		cudnn_tensor<float>		C(cudnn, { 1, 1, 1, 4 }, 0.0f);
		cudnn_tensor<float>		exp(cudnn, { 1, 1, 1, 4 }, 0.0f);
		std::vector<float>		h_C, h_exp;

		A.set<float>({ 0.0f, 1.0f, 1.0f, 0.0f });
		B.set<float>({ 1.0f, 1.0f, 1.0f, 1.0f });
		exp.set<float>({ -1.0f, 0.0f, 0.0f, -1.0f });

		cuda_matrix_subtraction<float>		sub(cudnn);
		sub.apply(A, B, C);
		C.get(h_C);
    exp.get(h_exp);

		auto sub_correct = std::equal(
			std::begin(h_C),
			std::end(h_C),
			std::begin(h_exp));

		if (sub_correct == false)
		{
			return 4;
		}
	}

	return 0;
}