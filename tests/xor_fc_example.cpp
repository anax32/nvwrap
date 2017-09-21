#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include "cuda_initialise.h"
#include "cudnn_initialise.h"
#include "cublas_initialise.h"
#include "cudnn_tensor.h"
#include "cublas_matrix_multiply.h"
#include "cublas_matrix_transpose.h"
#include "cudnn_activation.h"
#include "cudnn_transpose.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <array>

class cudnn_tensor_operation : public cudnn_result
{
protected:
	cudnnHandle_t	cudnn_context_;
	cudnnOpTensorDescriptor_t descriptor_;
public:
	cudnn_tensor_operation (cudnnHandle_t cudnn_context)
		: cudnn_context_(cudnn_context)
	{
		result_ = cudnnCreateOpTensorDescriptor(
			&descriptor_);
	}
	virtual ~cudnn_tensor_operation()
	{
		result_ = cudnnDestroyOpTensorDescriptor(
			descriptor_);
	}
};

template<typename T>
class cudnn_multiply : public cudnn_tensor_operation,
					   public cudnn_internal_type<T>
{
public:
	cudnn_multiply(cudnnHandle_t cudnn_context)
		: cudnn_tensor_operation(cudnn_context)
	{
		result = cudnnSetOpTensorDescriptor(
			descriptor_,
			CUDNN_OP_TENSOR_MUL,
			internal_type(),
			CUDNN_PROPAGATE_NAN);
	}
	template<typename T>
	static auto get_output_size(const cudnn_tensor<T>& A, const cudnn_tensor<T>& B) -> std::vector < int >
	{
		auto A_size = A.dimensions();
		auto B_size = B.dimensions();

		// retain only the last two dimensions,
		// we don't do general tensor multiplication yet
		return std::vector<int> 
		{
			1,
			1,
			A_size[2],
			B_size[3]
		};
	}
	void mult(const cudnn_tensor<T>& A, const cudnn_tensor<T>& B, cudnn_tensor<T>& C)
	{
		float one = 1.0f;
		float zero = 0.0f;

		result = cudnnOpTensor(
			cudnn_context_,
			descriptor_,
			&one, A, A.device_storage(),
			&one, B, B.device_storage(),
			&zero, C, C.device_storage());
	}
};

template<typename T>
class cudnn_add : public cudnn_tensor_operation,
				  public cudnn_internal_type<T>
{
public:
	cudnn_add(cudnnHandle_t cudnn_context)
		: cudnn_tensor_operation(cudnn_context)
	{
		result_ = cudnnSetOpTensorDescriptor(
			descriptor_,
			CUDNN_OP_TENSOR_ADD,
			internal_type(),
			CUDNN_PROPAGATE_NAN);
	}
	void apply(const cudnn_tensor<T>& A, const cudnn_tensor<T>& B, cudnn_tensor<T>& C)
	{
#ifdef _DEBUG
		auto a_sz = A.dimensions();
		auto b_sz = B.dimensions();
		auto c_sz = C.dimensions();
#endif

		float one = 1.0f;
		float zero = 0.0f;

		result_ = cudnnOpTensor(
			cudnn_context_,
			descriptor_,
			&one, A, A.device_storage(),
			&one, B, B.device_storage(),
			&zero, C, C.device_storage());
	}
};

template<typename T>
class cudnn_scale : public cudnn_result
{
protected:
	cudnnHandle_t	cudnn_context_;
public:
	cudnn_scale(cudnnHandle_t cudnn_context)
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

int main(int argc, char** argv)
{
	cuda_initialise		cuda;
	cudnn_initialise	cudnn;
	cublas_initialise	cublas;

	// cublas matrix multiplication of square matrices
	{
		cublas_matrix_multiply		matmul(cublas);
		cudnn_tensor<float>			A(cudnn, { 1, 1, 2, 2 }, 1.0f);
		cudnn_tensor<float>			B(cudnn, { 1, 1, 2, 2 }, 1.0f);
		cudnn_tensor<float>			C(cudnn, { 1, 1, 2, 2 }, 0.0f);
		std::vector<float>			h_C;

		matmul.apply<float>(A, B, C);
		C.get(h_C);

		auto square_mult_correct = std::all_of(
			std::begin(h_C),
			std::end(h_C),
			[](const float x){return x == 2.0f; });

		if (square_mult_correct == false)
		{
			return 1;
		}
	}
	// cublas matrix multiplcation of non-square matrices
	{
		cublas_matrix_multiply		matmul(cublas);
		cudnn_tensor<float>			A(cudnn, { 1, 1, 4, 2 }, 1.0f);
		cudnn_tensor<float>			B(cudnn, { 1, 1, 2, 3 }, 1.0f);
		cudnn_tensor<float>			C(cudnn, matmul.get_output_size(A.dimensions(), B.dimensions()), 0.0f);
		std::vector<float>			h_C;

		matmul.apply<float>(A, B, C);
		C.get(h_C);

		auto wonky_mult_correct = std::all_of(
			std::begin(h_C),
			std::end(h_C),
			[](const float x) { return x == 2.0f; });

		if (wonky_mult_correct == false)
		{
			return 2;
		}
	}
	// cudnn matrix addition
	{
		cudnn_tensor<float>		A(cudnn, { 1, 1, 8, 4 }, 1.0f);
		cudnn_tensor<float>		B(cudnn, { 1, 1, 8, 4 }, 3.0f);
		cudnn_tensor<float>		C(cudnn, { 1, 1, 8, 4 }, 0.0f);
		std::vector<float>		h_C;

		cudnn_add<float>		add(cudnn);
		add.apply(A, B, C);
		C.get(h_C);

		auto add_correct = std::all_of(
			std::begin(h_C),
			std::end(h_C),
			[](const float x) { return x == 4.0f; });

		if (add_correct == false)
		{
			return 3;
		}
	}

	// cudnn matrix transpose
	{
		cudnn_tensor<float>		A(cudnn, { 1, 1, 6, 3 }, 1.0f);
		cudnn_tensor<float>		B(cudnn, { 1, 1, 3, 6 }, 0.0f);
		std::vector<float>		h_B;

		cudnn_transpose<float>	transpose(cudnn, cublas);
		transpose.apply(A, B);
		B.get(h_B);

		auto transpose_correct = std::all_of(
			std::begin(h_B),
			std::end(h_B),
			[](const float x) {return x == 1.0f; });

		if (transpose_correct == false)
		{
			return 4;
		}
	}

	// NN learning XOR function
	{
		// read the data
		auto input = std::array <float, 3 * 4 >
		{{
				0.0f, 0.0f, 1.0f,
				0.0f, 1.0f, 1.0f,
				1.0f, 0.0f, 1.0f,
				1.0f, 1.0f, 1.0f
		}};

		auto output = std::array<float, 4>
		{{
				0.0f,
					1.0f,
					1.0f,
					0.0f
			}};

		// host vars
		std::vector<float>	h_X, h_Y, h_syn0, h_syn1, h_l1, h_l2;

		// create the tensors
		cudnn_tensor<float>	X(
			cudnn,
			{ 1, 1, 4, 3 },
			0.0f);
		X.set(input);
		X.get(h_X);

		cudnn_tensor<float>	Y(
			cudnn,
			{ 1, 1, 1, 4 },
			0.0f);
		Y.set(output);
		Y.get(h_Y);
		cudnn_tensor<float> Y_T(
			cudnn,
			{ 1, 1, 4, 1 },
			0.0f);

		cudnn_tensor<float> syn0(
			cudnn,
			{ 1, 1, 3, 4 },
			[]() {return ((2.0f*((float)rand() / (float)RAND_MAX)) - 1.0f); });

		cudnn_tensor<float> syn1(
			cudnn,
			{ 1, 1, 4, 1 },
			[](){return (2.0f*((float)rand() / (float)RAND_MAX)) - 1.0f; });

		cudnn_tensor<float> l1(
			cudnn,
			cudnn_multiply<float>::get_output_size(X, syn0),
			0.0f);
		cudnn_tensor<float> l2(
			cudnn,
			cudnn_multiply<float>::get_output_size(l1, syn1),
			0.0f);

		cudnn_tensor<float>	error(
			cudnn,
			l2.dimensions(),
			0.0f);

		cudnn_activation_layer<float>	l1_activation_layer(
			cudnn,
			l1.dimensions(),
			CUDNN_ACTIVATION_SIGMOID,
			CUDNN_PROPAGATE_NAN,
			1.0f);

		cudnn_activation_layer<float>	l2_activation_layer(
			cudnn,
			l2.dimensions(),
			CUDNN_ACTIVATION_SIGMOID,
			CUDNN_PROPAGATE_NAN,
			1.0f);

		cublas_matrix_multiply	matmul(cublas);
		cudnn_scale<float>		scale(cudnn);
		cudnn_add<float>		add(cudnn);
		cudnn_transpose<float>	transpose(cudnn, cublas);


		//for (auto i = 0; i < 60000; i++)
		{
			matmul.apply(X, syn0, l1);
			l1_activation_layer.forward(l1, l1);

			matmul.apply(l1, syn1, l2);
			l2_activation_layer.forward(l2, l2);

			// compute the error
			error.fill(0.0f);				// zero the error
			add.apply(error, l2, error);	// copy l2 into error
			scale.apply (error, -1.0f);		// make error -l2
			transpose.apply(Y, Y_T);			// transpose Y
			add.apply(Y_T, error, error);	// add Y_T and -l2

			
			// get the derivative of l2



			// get the derivative of l1


		}
	}
	
	return 0;
}