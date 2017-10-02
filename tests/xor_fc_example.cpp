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

	// NN learning XOR function
	{
		// read the data
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
		std::vector<float>	h_X, h_Y, h_syn0, h_syn1, h_l1, h_l2;

		// create the tensors
		cudnn_tensor<float>	X(
			cudnn,
			{ 1, 1, 4, 3 },
			0.0f);
		X.set(input);
		X.get(h_X);

		cudnn_tensor<float>	Y(
			cudnn,
			{ 1, 1, 1, 4 },
			0.0f);
		Y.set(output);
		Y.get(h_Y);
		cudnn_tensor<float> Y_T(
			cudnn,
			{ 1, 1, 4, 1 },
			0.0f);

		cudnn_tensor<float> syn0(
			cudnn,
			{ 1, 1, 3, 4 },
			[]() {return ((2.0f*((float)rand() / (float)RAND_MAX)) - 1.0f); });

		cudnn_tensor<float> syn1(
			cudnn,
			{ 1, 1, 4, 1 },
			[](){return (2.0f*((float)rand() / (float)RAND_MAX)) - 1.0f; });

		cudnn_tensor<float> l1(
			cudnn,
			cublas_matrix_multiply::get_output_size(X.dimensions(), syn0.dimensions()),
			0.0f);
		cudnn_tensor<float> l2(
			cudnn,
			cublas_matrix_multiply::get_output_size(l1.dimensions(), syn1.dimensions()),
			0.0f);

		cudnn_tensor<float>	error(
			cudnn,
			l2.dimensions(),
			0.0f);

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

		cublas_matrix_multiply			matmul(cublas);
		cudnn_matrix_scale<float>		scale(cudnn);
		cudnn_matrix_addition<float>	add(cudnn);
		cudnn_matrix_transpose<float>	transpose(cudnn, cublas);


		//for (auto i = 0; i < 60000; i++)
		{
			matmul.apply(X, syn0, l1);
			l1_activation_layer.forward(l1, l1);

			matmul.apply(l1, syn1, l2);
			l2_activation_layer.forward(l2, l2);

			// compute the error
			error.fill(0.0f);				// zero the error
			add.apply(error, l2, error);	// copy l2 into error
			scale.apply (error, -1.0f);		// make error -l2
			transpose.apply(Y, Y_T);			// transpose Y
			add.apply(Y_T, error, error);	// add Y_T and -l2

			
			// get the derivative of l2



			// get the derivative of l1


		}
	}
	
	return 0;
}