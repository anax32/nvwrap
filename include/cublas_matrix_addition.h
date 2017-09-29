#include "cublas_result.h"

#ifndef CUBLAS_MATRIX_ADDITION_H
#define CUBLAS_MATRIX_ADDITION_H
class cublas_matrix_addition : public cublas_result
{
protected:
	cublasHandle_t	cublas_context_;
public:
	cublas_matrix_addition(cublasHandle_t cublas_context)
		: cublas_context_(cublas_context)
	{}
	virtual ~cublas_matrix_addition()
	{}
	template<typename T>
	void apply(const cudnn_tensor<T>& in_A,
			   const cudnn_tensor<T>& in_B,
		       const cudnn_tensor<T>& out_C)
	{
		auto a_sz = in_A.dimensions();
		auto b_sz = in_B.dimensions();
		auto c_sz = out_C.dimensions();

		float alpha = 1.0f;
		float beta = 1.0f;

		result_ = cublasSgeam(
			cublas_context_,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			a_sz[2],	// rows a
			b_sz[3],	// cols b
			&alpha,
			(float*)in_A.device_storage(),
			a_sz[2],
			&beta,
			(float*)in_B.device_storage(),
			b_sz[2],
			(float*)out_C.device_storage(),
			c_sz[2]);
	}
};
#endif