#include "cudnn_result.h"
#include "cudnn_layer.h"

#ifndef CUDNN_ACTIVATION_H
#define CUDNN_ACTIVATION_H
class cudnn_activation : public cudnn_result
{
protected:
	cudnnHandle_t				cudnn_context_;
	cudnnActivationDescriptor_t	descriptor_;

public:
	cudnn_activation(cudnnHandle_t cudnn_context)
		: cudnn_context_(cudnn_context)
	{
		result_ = cudnnCreateActivationDescriptor(&descriptor_);
	}
	cudnn_activation(cudnnHandle_t cudnn_context,
		cudnnActivationMode_t mode,
		cudnnNanPropagation_t nan_propagation,
		double coefficient)
		: cudnn_activation(cudnn_context)
	{
		result_ = cudnnSetActivationDescriptor(
			descriptor_,
			mode,
			nan_propagation,
			coefficient);
	}
	virtual ~cudnn_activation()
	{
		cudnnDestroyActivationDescriptor(descriptor_);
	}
	operator cudnnActivationDescriptor_t () const { return descriptor_; }
};

template<typename T>
class cudnn_activation_layer : public cudnn_layer<T>
{
protected:
	cudnn_activation			activation;
	//cudnn_tensor<T>				error_gradient;
	cudnn_tensor<T>				output_gradient;
public:
	cudnn_activation_layer(cudnnHandle_t cudnn_context,
						   const std::vector<int> input_size,
						   cudnnActivationMode_t mode,
						   cudnnNanPropagation_t nan_propagation,
						   double coefficient)
		: cudnn_layer(cudnn_context),
		  activation(cudnn_context_, mode, nan_propagation, coefficient),
		  //output(cudnn_context_, input_size),
		  //error_gradient(cudnn_context_, input_size),
		  output_gradient(cudnn_context_, input_size)
	{}	 
	void forward(const cudnn_tensor<T>& input, const cudnn_tensor<T>& output)
	{
		result_ = cudnnActivationForward(
			cudnn_context_,
			activation,
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
		result_ = cudnnActivationBackward(
			cudnn_context_,
			activation,
			alpha(),
			error,								// ydesc
			error.device_storage(),				// y
			error_gradient,						// dydesc
			error_gradient.device_storage(),	// dy
			input,								// xdesc
			input.device_storage(),				// x
			beta(),
			input_gradient,						// dxdesc
			input_gradient.device_storage());	// dx
	}
};
#endif