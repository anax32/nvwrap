#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include "cuda_initialise.h"
#include "cudnn_initialise.h"
#include "cublas_initialise.h"
#include "cudnn_tensor.h"
#include "cublas_matrix_multiply.h"
#include "cublas_matrix_transpose.h"
#include "cudnn_activation.h"
#include "cudnn_matrix_transpose.h"
#include "cudnn_matrix_addition.h"
#include "cudnn_matrix_subtraction.h"
#include "cudnn_matrix_scale.h"
#include "cudnn_matrix_hadamard.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <array>

int main(int argc, char** argv)
{
	cuda_initialise		cuda;
	cudnn_initialise	cudnn;
	cublas_initialise	cublas;

	// NN learning XOR function
	{
		// data
		auto input = std::array <float, 3 * 4 >
		{{
				0.0f, 0.0f, 1.0f,
				0.0f, 1.0f, 1.0f,
				1.0f, 0.0f, 1.0f,
				1.0f, 1.0f, 1.0f
		}};

		auto output = std::array<float, 4>
		{{
				0.0f,
				1.0f,
				1.0f,
				0.0f
		}};

		// host vars
		std::vector<float>	h_X, h_Y, h_syn0, h_syn1, h_l1, h_l2, h_error;

		auto rnd = []() {return ((2.0f*((float)rand() / (float)RAND_MAX)) - 1.0f); };

		// tensor storage for data
		cudnn_tensor<float>	X(
			cudnn,
			{ 1, 1, 4, 3 },
			0.0f);
		X.set(input);
		X.get(h_X);

		cudnn_tensor<float> X_T(cudnn, cudnn_matrix_transpose<float>::get_output_size(X.dimensions()), 0.0f);

		cudnn_tensor<float>	Y(
			cudnn,
			{ 1, 1, 1, 4 },
			0.0f);
		Y.set(output);
		Y.get(h_Y);

		cudnn_tensor<float> Y_T(cudnn, cudnn_matrix_transpose<float>::get_output_size(Y.dimensions()), 0.0f);

		cudnn_tensor<float> syn0(cudnn, { 1, 1, 3, 4 }, rnd);
		cudnn_tensor<float>	syn0_W(cudnn, syn0.dimensions(), 0.0f);

		cudnn_tensor<float> syn1(cudnn, { 1, 1, 4, 1 }, rnd);
		cudnn_tensor<float> syn1_T(cudnn, cudnn_matrix_transpose<float>::get_output_size(syn1.dimensions()), 0.0f);
		cudnn_tensor<float> syn1_W(cudnn, syn1.dimensions(), 0.0f);

		cudnn_tensor<float> l1(
			cudnn,
			cublas_matrix_multiply::get_output_size(X.dimensions(), syn0.dimensions()),
			0.0f);

		cudnn_tensor<float>	l1_delta(cudnn, l1.dimensions(), 0.0f);
		cudnn_tensor<float>	l1_inv(cudnn, l1.dimensions(), 0.0f);
		cudnn_tensor<float>	l1_one(cudnn, l1.dimensions(), 1.0f);
		cudnn_tensor<float> l1_gradient(cudnn, l1.dimensions(), 0.0f);
		cudnn_tensor<float>	l1_T(cudnn, cudnn_matrix_transpose<float>::get_output_size(l1.dimensions()), 0.0f);

		cudnn_tensor<float> l2(
			cudnn,
			cublas_matrix_multiply::get_output_size(l1.dimensions(), syn1.dimensions()),
			0.0f);

		cudnn_tensor<float>	l2_delta(cudnn, l2.dimensions(), 0.0f);
		cudnn_tensor<float>	l2_inv (cudnn, l2.dimensions(), 0.0f);
		cudnn_tensor<float>	l2_one(cudnn, l2.dimensions(), 1.0f);
		cudnn_tensor<float> l2_gradient(cudnn, l2.dimensions(), 0.0f);
		cudnn_tensor<float>	l2_T(cudnn, cudnn_matrix_transpose<float>::get_output_size(l2.dimensions()), 0.0f);

		cudnn_tensor<float>	error(cudnn, l2.dimensions(), 0.0f);

		// functions
		cublas_matrix_multiply			matmul(cublas);
		cudnn_matrix_scale<float>		scale(cudnn);
		cudnn_matrix_addition<float>	add(cudnn);
		cudnn_matrix_transpose<float>	transpose(cudnn, cublas);
		cudnn_matrix_subtraction<float>	subtract(cudnn);
		cudnn_matrix_hadamard<float>	hadamard(cudnn);

		// layers
		cudnn_activation_layer<float>	l1_activation_layer(
			cudnn,
			l1.dimensions(),
			CUDNN_ACTIVATION_SIGMOID,
			CUDNN_PROPAGATE_NAN,
			1.0f);

		cudnn_activation_layer<float>	l2_activation_layer(
			cudnn,
			l2.dimensions(),
			CUDNN_ACTIVATION_SIGMOID,
			CUDNN_PROPAGATE_NAN,
			1.0f);

		// adapt the weights
		for (auto i = 0; i < 10000; i++)
		{
			matmul.apply(X, syn0, l1);
			l1_activation_layer.forward(l1, l1);

			matmul.apply(l1, syn1, l2);
			l2_activation_layer.forward(l2, l2);

			// compute the error
			subtract.apply(Y, l2, error);

			Y.get(h_Y);
			l1.get(h_l1);
			l2.get(h_l2);
			error.get(h_error);

			for (auto j = 0; j < h_error.size(); j++)
			{
				if (h_error[j] != h_Y[j] - h_l2[j])
				{
					break;
				}
			}

			// compute the error sum
			error.get(h_error);
			auto error_sum = std::accumulate(
				std::begin(h_error),
				std::end(h_error),
				0.0f,
				[](const float x, const float y) {return x + std::abs(y); });

			if (i % 1000 == 0)
			{
				std::cout << i << " : " << error_sum << std::endl;
			}

			// get the derivative of l2
			subtract.apply(l2_one, l2, l2_inv);			// backprop of activation function
			hadamard.apply(l2, l2_inv, l2_gradient);	// backprop of activation function
			hadamard.apply(error, l2_gradient, l2_delta);

			// get the derivative of l1
			subtract.apply(l1_one, l1, l1_inv);			// backprop of activation function
			hadamard.apply(l1, l1_inv, l1_gradient);	// backprop of activation function
			transpose.apply(syn1, syn1_T);
			matmul.apply(l2_delta, syn1_T, l1_delta);
			hadamard.apply(l1_delta, l1_gradient, l1_delta);	// weight according to gradient

			// update the weights
			transpose.apply(l1, l1_T);
			matmul.apply(l1_T, l2_delta, syn1_W);
			add.apply(syn1, syn1_W, syn1);

			transpose.apply(X, X_T);
			matmul.apply(X_T, l1_delta, syn0_W);
			add.apply(syn0, syn0_W, syn0);
		}

		// output the output...
		Y.get(h_Y);
		l1.get(h_l1);
		l2.get(h_l2);
		error.get(h_error);

		std::copy(
			std::begin(h_l2),
			std::end(h_l2),
			std::ostream_iterator<float>(std::cout));
	}

	return 0;
}