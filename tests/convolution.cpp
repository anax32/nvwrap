#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "cudnn.lib")

#include <cuda.h>
#include <cudnn.h>

#include "cuda_initialise.h"
#include "cudnn_initialise.h"
#include "cudnn_filter.h"
#include "cudnn_tensor.h"
#include "cudnn_convolution.h"

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

	cudnn_tensor<float> output_gradient(
		cudnn,
		{ 1, 8, 64, 64 },
		0.0f);

	cudnn_filter<float>	filter(
		{ 1, 8, 5, 5 },
		[]() {return ((float)rand() / (float)RAND_MAX); });

	// layer primitives
	{
		cudnn_convolution<float>	conv(
			cudnn,
			{ 2, 2 },
			{ 1, 1 },
			{ 1, 1 });	// stride and dilation must be non-zero

		cudnn_filter<float>			filter_gradient(filter.dimensions());
		cudnn_tensor<float>			output(cudnn, conv.get_output_size(input, filter));
		cudnn_tensor<float>			gradient(cudnn, conv.get_output_size(input, filter));

		// get the best algorithm
		cudnn_convolution_forward_algorithm	forward(
			cudnn,
			input,
			filter,
			conv,
			output);

		cudnn_convolution_backward_filter_algorithm back_filter(
			cudnn,
			input,
			filter,
			conv,
			gradient);

		cudnn_convolution_backward_data_algorithm back_data(
			cudnn,
			output,
			filter,
			conv,
			gradient);

		// apply a forward algorithm
		float one = 1.0f;
		float zero = 0.0f;
		forward.apply(input, output, filter, conv, &one, &zero);
		back_filter.apply(input, gradient, filter_gradient, conv, &one, &zero);
		back_data.apply(input, gradient, filter, conv, output, &one, &zero);
	}
	
	// layer constructor
	{
		cudnn_convolution_layer<float>	cnn_layer(
			cudnn,
			{ 2, 2 }, { 1, 1 }, { 1, 1 },
			{ 1, 8, 5, 5 },
			input,
			output);

		cnn_layer.forward(input, output);
		cnn_layer.backward(input, input_gradient, output, output_gradient);
	}
	
	return 0;
}