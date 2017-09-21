#include "cublas_result.h"

#ifndef CUBLAS_MATRIX_TRANSPOSE_H
#define CUBLAS_MATRIX_TRANSPOSE_H
class cublas_matrix_transpose : public cublas_result
{
protected:
	cublasHandle_t	cublas_context_;
public:
	cublas_matrix_transpose(cublasHandle_t cublas_context)
		: cublas_context_(cublas_context)
	{}
	virtual ~cublas_matrix_transpose()
	{}
	template<typename T>
	void apply(const cudnn_tensor<T>& input, cudnn_tensor<T>& output)
	{
		auto input_sz = input.dimensions();
		auto output_sz = output.dimensions();

		float alpha = 1.0f;
		float beta = 0.0f;

		result_ = cublasSgeam(
			cublas_context_,
			CUBLAS_OP_T,
			CUBLAS_OP_N,
			input_sz[3],	// rows a(transposed)
			output_sz[3],	// cols b
			&alpha,
			(float*)input.device_storage(),
			input_sz[2],
			&beta,
			(float*)output.device_storage(),	// in_place mode requires B and
			output_sz[2],						// output are commensurate
			(float*)output.device_storage(),
			output_sz[2]);
	}
};
#endif