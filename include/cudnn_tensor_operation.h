#include "cudnn_result.h"

#ifndef CUDNN_TENSOR_OPERATION_H
#define CUDNN_TENSOR_OPERATION_H
class cudnn_tensor_operation : public cudnn_result
{
protected:
	cudnnHandle_t	cudnn_context_;
	cudnnOpTensorDescriptor_t descriptor_;
public:
	cudnn_tensor_operation(cudnnHandle_t cudnn_context)
		: cudnn_context_(cudnn_context)
	{
		result_ = cudnnCreateOpTensorDescriptor(
			&descriptor_);
	}
	virtual ~cudnn_tensor_operation()
	{
		result_ = cudnnDestroyOpTensorDescriptor(
			descriptor_);
	}
	template<typename T>
	void apply(const cudnn_tensor<T>& A,
		const cudnn_tensor<T>& B,
		cudnn_tensor<T>& C)
	{
#ifdef _DEBUG
		auto a_sz = A.dimensions();
		auto b_sz = B.dimensions();
		auto c_sz = C.dimensions();
#endif

		float a_weight = 1.0f;
		float b_weight = 1.0f;
		float c_weight = 0.0f;

		result_ = cudnnOpTensor(
			cudnn_context_,
			descriptor_,
			&a_weight, A, A.device_storage(),
			&b_weight, B, B.device_storage(),
			&c_weight, C, C.device_storage());
	}
};
#endif