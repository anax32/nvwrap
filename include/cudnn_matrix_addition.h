#include "cudnn_internal_type.h"
#include "cudnn_tensor_operation.h"

#ifndef CUDNN_MATRIX_ADDITION_H
#define CUDNN_MATRIX_ADDITION_H
template<typename T>
class cudnn_matrix_addition : public cudnn_tensor_operation,
							  public cudnn_internal_type<T>
{
public:
	cudnn_matrix_addition(cudnnHandle_t cudnn_context)
		: cudnn_tensor_operation(cudnn_context)
	{
		result_ = cudnnSetOpTensorDescriptor(
			descriptor_,
			CUDNN_OP_TENSOR_ADD,
			internal_type(),
			CUDNN_PROPAGATE_NAN);
	}
};
#endif