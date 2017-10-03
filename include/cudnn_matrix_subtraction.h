#include "cudnn_internal_type.h"
#include "cudnn_matrix_addition.h"
#include "cudnn_matrix_scale.h"

#ifndef CUDNN_MATRIX_SUBTRACTION_H
#define CUDNN_MATRIX_SUBTRACTION_H
template<typename T>
class cudnn_matrix_subtraction : public cudnn_internal_type<T>
{
protected:
	cudnn_matrix_addition<T> add;
	cudnn_matrix_scale<T> scale;

public:
	cudnn_matrix_subtraction(cudnnHandle_t cudnn_context)
		: add(cudnn_context), scale(cudnn_context)
	{}
	template<typename T>
	void apply(const cudnn_tensor<T>& A,
		       const cudnn_tensor<T>& B,
		       cudnn_tensor<T>& C)
	{
		// FIXME: this does not work because the add operation
		// reads and writes to the tensor C, which causes
		// cudnnOpTensor to fail
		// (see: http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnOpTensor)
		// we must find a way to negate B and perform A+B
		// without overwriting the contents of A or B.
		// (unless we overwrite and restore at the end?)
		throw std::exception("not implemented");
#if 0
		// copy B in to C
		cudaMemcpy(
			C.device_storage(),
			B.device_storage(),
			B.size(),
			cudaMemcpyDeviceToDevice);

		// scale B to -B and put in C
		scale.apply(C, -1.0f);

		// add A to C and store the result in C
		add.apply(A, C, C);

		// C now contains A-B
#endif
	}
};
#endif