#include <cuda.h>
#include <cudnn.h>

#include "cuda_initialise.h"
#include "cudnn_initialise.h"
#include "cudnn_tensor.h"
#include "cudnn_layer.h"
#include "cudnn_activation.h"

int main(int argc, char** argv)
{
	cuda_initialise	cuda;
	cudnn_initialise cudnn;

	cudnn_tensor<float>	input(
		cudnn,
		{ 1, 8, 64, 64 },
		0.0f);

	cudnn_tensor<float>	input_gradient(
		cudnn,
		{ 1, 8, 64, 64 },
		0.0f);

	cudnn_tensor<float> output(
		cudnn,
		{ 1, 8, 64, 64 },
		0.0f);

	cudnn_tensor<float> error(
		cudnn,
		{ 1, 8, 64, 64 },
		0.0f);

	cudnn_tensor<float> error_gradient(
		cudnn,
		{ 1, 8, 64, 64 },
		0.0f);
	
	// layer primitives
	{
		cudnn_activation	acc(
			cudnn,
			CUDNN_ACTIVATION_SIGMOID,
			CUDNN_PROPAGATE_NAN,
			0.0f);
	}

	// layer constructor
	{
		cudnn_activation_layer<float>	act_layer(
			cudnn,
			{ 1, 8, 64, 64 },
			CUDNN_ACTIVATION_SIGMOID,
			CUDNN_PROPAGATE_NAN,
			1.0f);

		act_layer.forward(input, output);
		act_layer.backward(input, input_gradient, error, error_gradient);
	}

	return 0;
}