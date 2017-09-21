#include "cudnn_result.h"

#ifndef CUDNN_LAYER_H
#define CUDNN_LAYER_H
template<typename T>
class cudnn_layer : public cudnn_result
{
protected:
	cudnnHandle_t	cudnn_context_;
	float			one;
	float			zero;
public:
	cudnn_layer(cudnnHandle_t cudnn_context)
		: cudnn_context_(cudnn_context),
		  one(1.0f),
		  zero(0.0f)
	{}

	float *alpha()
	{
		return &one;
	}
	float *beta()
	{
		return &zero;
	}

	// forward propagation updates the input tensor inplace
	// for cudnn documentation:
	//   x = input
	//	 otherParams = this object
	virtual void forward(const cudnn_tensor<T>& input,
						 const cudnn_tensor<T>& output) = NULL;

	// backward propagation updates the gradient tensor inplace
	// for cudnn documentation:
	//	 x = input tensor
	//	 dx = input gradient tensor
	//	 y = error tensor
	//	 dy = error gradient tensor
	//	 otherParams = this object
	virtual void backward(const cudnn_tensor<T>& input, const cudnn_tensor<T>& input_gradient,
						  const cudnn_tensor<T>& error, const cudnn_tensor<T>& error_gradient) = NULL;
};
#endif