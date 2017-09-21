#include "cublas_matrix_transpose.h"
#include "cudnn_result.h"

#ifndef CUDNN_TRANSPOSE_H
#define CUDNN_TRANSPOSE_H
template<typename T>
class cudnn_transpose : public cudnn_result
{
protected:
	cudnnHandle_t	cudnn_context_;
	cublas_matrix_transpose	cublas_transpose;
public:
	cudnn_transpose(cudnnHandle_t cudnn_context,
		cublasHandle_t cublas_context)
		: cudnn_context_(cudnn_context),
		cublas_transpose(cublas_context)
	{}
	void apply(const cudnn_tensor<T>& input, cudnn_tensor<T>& output)
	{
		cublas_transpose.apply(input, output);

		result_ = cudnn_result::cublas_status_codes[cublas_transpose.result ()];
	}
};
#endif