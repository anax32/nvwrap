#include <cuda.h>
#include <cudnn.h>

#include "cuda_initialise.h"
#include "cudnn_initialise.h"
#include "cudnn_tensor.h"
#include "cudnn_layer.h"

class cudnn_pooling : public cudnn_result
{
protected:
	cudnnPoolingDescriptor_t	descriptor_;
public:
	cudnn_pooling()
		: descriptor_(NULL)
	{
		result_ = cudnnCreatePoolingDescriptor(&descriptor_);
	}
	cudnn_pooling(const std::vector<int>& size,
				  const std::vector<int>& padding,
				  const std::vector<int>& stride,
				  cudnnPoolingMode_t mode,
				  cudnnNanPropagation_t nan_propagation)
		: cudnn_pooling ()
	{ 
#if 0
		result_ = cudnnSetPooling2dDescriptor(
			descriptor_,
			mode,
			nan_propagation,
			size[0],
			size[1],
			padding[0],
			padding[1],
			stride[0],
			stride[1]);
#else
		result_ = cudnnSetPoolingNdDescriptor(
			descriptor_,
			mode,
			nan_propagation,
			size.size(),
			size.data(),
			padding.data(),
			stride.data());
#endif
	}
	virtual ~cudnn_pooling()
	{
		cudnnDestroyPoolingDescriptor(descriptor_);
	}
	operator cudnnPoolingDescriptor_t() const { return descriptor_; }
	auto get_output_size(const cudnnTensorDescriptor_t& input_tensor)->std::vector < int >
	{
		std::vector<int>		output_dimensions;
#if 0
		int	n, c, h, w;

		result = cudnnGetPooling2dForwardOutputDim(
			descriptor_,
			input_tensor,
			&n, &c, &h, &w);

		output_dimensions[0] = n;
		output_dimensions[1] = c;
		output_dimensions[2] = h;
		output_dimensions[3] = w;
#else
		int					dummy;
		cudnnDataType_t		input_type;
		int					input_dims;

		result_ = cudnnGetTensorNdDescriptor(
			input_tensor,
			1,
			&input_type,
			&input_dims,
			&dummy,
			&dummy);

		output_dimensions.resize(input_dims);

		result_ = cudnnGetPoolingNdForwardOutputDim(
			descriptor_,
			input_tensor,
			input_dims,
			output_dimensions.data());
#endif
		return output_dimensions;
	}
};

template<typename T>
class cudnn_pooling_layer : public cudnn_layer<T>
{
protected:
	cudnn_pooling			pooling;
public:
	cudnn_pooling_layer(cudnnHandle_t cudnn_context,
						const std::vector<int>& size,
						const std::vector<int>& padding,
						const std::vector<int>& stride,
						const cudnnPoolingMode_t mode,
						const cudnnNanPropagation_t nan_propagation,
						const cudnn_tensor<T>& input_tensor)
		: cudnn_layer(cudnn_context),
		  pooling(size, padding, stride, mode, nan_propagation)
	{}
	virtual ~cudnn_pooling_layer()
	{}

	void forward(const cudnn_tensor<T>& input, const cudnn_tensor<T>& output)
	{
		result_ = cudnnPoolingForward(
			cudnn_context_,
			pooling,
			alpha(),
			input,
			input.device_storage(),
			beta(),
			output,
			output.device_storage());
	}
	void backward(const cudnn_tensor<T>& input, const cudnn_tensor<T>& input_gradient,
				  const cudnn_tensor<T>& error, const cudnn_tensor<T>& error_gradient)
	{
		result_ = cudnnPoolingBackward(
			cudnn_context_,
			pooling,
			alpha(),
			input,
			input.device_storage(),
			input_gradient,
			input_gradient.device_storage(),
			error,
			error.device_storage(),
			beta(),
			error_gradient,
			error_gradient.device_storage());
	}
};


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
		{ 1, 8, 32, 32 },
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
		cudnn_pooling	acc(
		{ 2, 2 },
		{ 1, 1 },
		{ 1, 1 },
			CUDNN_POOLING_MAX,
			CUDNN_PROPAGATE_NAN);
	}

	// layer constructor
	{
		cudnn_pooling_layer<float>	pool_layer(
			cudnn,
			{ 2, 2 },
			{ 1, 1 },
			{ 1, 1 },
			CUDNN_POOLING_MAX,
			CUDNN_PROPAGATE_NAN,
			input);

		pool_layer.forward(input, output);
		pool_layer.backward(input, input_gradient, error, error_gradient);
	}

	return 0;
}