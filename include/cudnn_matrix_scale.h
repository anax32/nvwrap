#include "cudnn_result.h"

#ifndef CUDNN_MATRIX_SCALE_H
#define CUDNN_MATRIX_SCALE_H
template<typename T>
class cudnn_matrix_scale : public cudnn_result
{
protected:
	cudnnHandle_t	cudnn_context_;
public:
	cudnn_matrix_scale(cudnnHandle_t cudnn_context)
		: cudnn_context_(cudnn_context)
	{}
	void apply(const cudnn_tensor<T>& input, const T scale)
	{
		result_ = cudnnScaleTensor(
			cudnn_context_,
			input,
			input.device_storage(),
			&scale);
	}
};
#endif