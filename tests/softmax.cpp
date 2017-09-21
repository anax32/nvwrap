#include <cuda.h>
#include <cudnn.h>

#include "cuda_initialise.h"
#include "cudnn_initialise.h"
#include "cudnn_tensor.h"
#include "cudnn_layer.h"
#include "cudnn_activation.h"

template<typename T>
class cudnn_softmax_layer : public cudnn_layer<T>
{
protected:
	cudnnSoftmaxAlgorithm_t		algorithm_;
	cudnnSoftmaxMode_t			mode_;

public:
	cudnn_softmax_layer(cudnnHandle_t cudnn_context,
						cudnnSoftmaxAlgorithm_t algorithm,
						cudnnSoftmaxMode_t mode)
		: cudnn_layer (cudnn_context),
		  algorithm_ (algorithm),
		  mode_ (mode)
	{}
	cudnn_softmax_layer (cudnnHandle_t cudnn_context)
		: cudnn_softmax_layer (cudnn_context, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE)
	{}
	void forward(const cudnn_tensor<T>& input, const cudnn_tensor<T>& output)
	{
		result_ = cudnnSoftmaxForward(
			cudnn_context_,
			algorithm_,
			mode_,
			alpha(),
			input,
			input.device_storage(),
			beta(),
			output,
			output.device_storage());
	}
	void backward(const cudnn_tensor<T>& input, const cudnn_tensor<T>& input_gradient,
				  const cudnn_tensor<T>& error, const cudnn_tensor<T>& error_gradient)
	{
		result_ = cudnnSoftmaxBackward(
			cudnn_context_,
			algorithm_,
			mode_,
			alpha(),
			error,
			error.device_storage(),
			error_gradient,
			error_gradient.device_storage(),
			beta(),
			input_gradient,
			input_gradient.device_storage());
	}
};

template<typename T>
std::vector<T> makeDiffData(int m, int c)
{
	std::vector<T> diff (m*c);

	for (int j = 0; j < m; j++)
	{
		int cls = rand() % c;

		printf("%d class: %d\n", j, cls);

		for (int i = 0; i < c; i++)
		{
			diff[j * c + i] = cls == i ? -c / (T)m : 0;
		}
	}

	return diff;
}

int main(int argc, char** argv)
{
	cuda_initialise	cuda;
	cudnn_initialise cudnn;

	// check the softmax sums to one
	// FIXME: does not work with doubles
	{
		cudnn_softmax_layer<float>	softmax(cudnn);

		cudnn_tensor<float>		A(cudnn, { 1, 1, 1, 4 });
		cudnn_tensor<float>		B(cudnn, { 1, 1, 1, 4 });

		std::vector<float>		A_h{ 0.4f, 0.5f, 0.6f, 0.8f };
		std::vector<float>		B_h(4);
		A.set(A_h);

		softmax.forward(A, B);

		B.get(B_h);

		auto sum = std::accumulate(
			std::begin(B_h),
			std::end(B_h),
			0.0f);

		auto sums_to_one = std::abs (sum - 1.0f) < 0.0001f;

		if (sums_to_one == false)
		{
			return 1;
		}
	}

	// backprop
	{
		cudnn_softmax_layer<float>	softmax(cudnn);

		cudnn_tensor<float>		A(cudnn, { 1, 1, 1, 4 });
		cudnn_tensor<float>		dA(cudnn, { 1, 1, 1, 4 });
		cudnn_tensor<float>		E(cudnn, { 1, 1, 1, 4 });
		cudnn_tensor<float>		dE(cudnn, { 1, 1, 1, 4 });
		cudnn_tensor<float>		B(cudnn, { 1, 1, 1, 4 });
		cudnn_tensor<float>		D(cudnn, { 1, 1, 1, 4 });

		std::vector<float>		A_h { 0.4f, 0.5f, 0.6f, 0.8f };
		std::vector<float>		dA_h{ 0.1f, 0.1f, 0.1f, 0.1f };
		std::vector<float>		exp { 1.0f, 0.0f, 0.0f, 0.0f };
		std::vector<float>		E_h { 0.6f, 0.0f, 0.0f, 0.0f };
		std::vector<float>		dE_h{ 0.0f, 0.0f, 0.0f, 0.0f };
		std::vector<float>		B_h(4);
		std::vector<float>		D_h(4);
		A.set(A_h);

		softmax.forward(A, B);

		// get an error
		std::transform(
			std::begin(A_h),
			std::end(A_h),
			std::begin(exp),
			std::begin(E_h),
			[](const float a, const float x) {return std::abs(x - a); });

		dA.set(dA_h);
		E.set(E_h);

		auto diff = makeDiffData<float>(1, 4);
		dA.set(diff);
		
		softmax.backward(A, dA, E, dE);

		dE.get(dE_h);

		auto sum = std::accumulate(
			std::begin(B_h),
			std::end(B_h),
			0.0f);

		auto sums_to_one = std::abs(sum - 1.0f) < 0.0001f;

		if (sums_to_one == false)
		{
			return 2;
		}
	}

	return 0;
}