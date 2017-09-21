#include "cublas_result.h"

#include <vector>

#ifndef CUBLAS_MATRIX_MULTIPLY_H
#define CUBLAS_MATRIX_MULTIPLY_H
class cublas_matrix_multiply : public cublas_result
{
protected:
	cublasHandle_t	cublas_context_;
public:
	cublas_matrix_multiply(cublasHandle_t cublas_context)
		: cublas_context_(cublas_context)
	{}
	virtual ~cublas_matrix_multiply()
	{}
	template<typename T>
	void apply(const cudnn_tensor<T>& in_A, const cudnn_tensor<T>& in_B, const cudnn_tensor<T>& out_C)
	{
		auto a_sz = in_A.dimensions();
		auto b_sz = in_B.dimensions();
		auto c_sz = out_C.dimensions();

		float alpha = 1.0f;
		float beta = 0.0f;

		result_ = cublasSgemm(
			cublas_context_,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			a_sz[2],	// rows a
			b_sz[3],	// cols b
			a_sz[3],	// cols a rows b
			&alpha,
			(float*)in_A.device_storage(),
			a_sz[2],
			(float*)in_B.device_storage(),
			b_sz[2],
			&beta,
			(float*)out_C.device_storage(),
			c_sz[2]);
	}

	std::vector<int> get_output_size(const std::vector<int>& tensor_a_size, const std::vector<int>& tensor_b_size)
	{
		std::vector<int>	out_dims;

		if (tensor_a_size.size() != tensor_b_size.size())
			return out_dims;

		out_dims.resize(tensor_a_size.size());

		std::fill(
			std::begin(out_dims),
			std::end(out_dims),
			1);

		out_dims[out_dims.size() - 2] = tensor_a_size[tensor_a_size.size() - 2];
		out_dims[out_dims.size() - 1] = tensor_b_size[tensor_b_size.size() - 1];

		return out_dims;
	}
};
#endif