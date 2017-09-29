#include "cudnn_tensor_operation.h"

#ifndef CUDNN_MATRIX_HADAMARD_H
#define CUDNN_MATRIX_HADAMARD_H
template<typename T>
class cudnn_matrix_hadamard : public cudnn_tensor_operation,
							  public cudnn_internal_type<T>
{
public:
	cudnn_matrix_hadamard(cudnnHandle_t cudnn_context)
		: cudnn_tensor_operation(cudnn_context)
	{
		// CUDNN_OP_TENSOR_MUL is elementwise multiplication
		result_ = cudnnSetOpTensorDescriptor(
			descriptor_,
			CUDNN_OP_TENSOR_MUL,
			internal_type(),
			CUDNN_PROPAGATE_NAN);
	}
};
#endif