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
		// copy B in to C
		cudaMemcpy(
			C.device_storage(),
			B.device_storage(),
			B.size(),
			cudaMemcpyDeviceToDevice);

		// scale B to -B and put in C
		scale.apply(C, -1.0f);

		// add A to C and store the result in C
		add.apply(C, A, C);

		// C now contains A-B
	}
};
#endif