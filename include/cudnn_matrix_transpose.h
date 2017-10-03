#include "cublas_matrix_transpose.h"
#include "cudnn_result.h"

#ifndef CUDNN_MATRIX_TRANSPOSE_H
#define CUDNN_MATRIX_TRANSPOSE_H
template<typename T>
class cudnn_matrix_transpose : public cudnn_result
{
protected:
	cudnnHandle_t	cudnn_context_;
	cublas_matrix_transpose	cublas_transpose;
public:
	cudnn_matrix_transpose(cudnnHandle_t cudnn_context,
		                   cublasHandle_t cublas_context)
		: cudnn_context_(cudnn_context),
		  cublas_transpose(cublas_context)
	{}
	void apply(const cudnn_tensor<T>& input, cudnn_tensor<T>& output)
	{
		cublas_transpose.apply(input, output);

		result_ = cudnn_result::cublas_status_codes[cublas_transpose.result ()];
	}

	static std::vector<int> get_output_size(const std::vector<int>& input_tensor_size)
	{
		std::vector<int>	out_dims;

		out_dims.resize(input_tensor_size.size());

		std::fill(
			std::begin(out_dims),
			std::end(out_dims),
			1);

		out_dims[out_dims.size() - 2] = input_tensor_size[input_tensor_size.size() - 1];
		out_dims[out_dims.size() - 1] = input_tensor_size[input_tensor_size.size() - 2];

		return out_dims;
	}
};
#endif