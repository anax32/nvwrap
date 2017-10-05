#include "cudnn_result.h"
#include "cudnn_layer.h"

#include <vector>
#include <algorithm>
#include <numeric>

#ifndef CUDNN_CONVOLUTION_H
#define CUDNN_CONVOLUTION_H
template<typename T>
class cudnn_convolution : public cudnn_result,
						  public cudnn_internal_type<T>
{
protected:
	cudnnHandle_t					cudnn_context_;
	cudnnConvolutionDescriptor_t	descriptor_;

public:
	cudnn_convolution(cudnnHandle_t cudnn_context)
		: cudnn_context_(cudnn_context),
		descriptor_(NULL)
	{
		result_ = cudnnCreateConvolutionDescriptor(&descriptor_);
	}
	cudnn_convolution(cudnnHandle_t cudnn_context,
		std::vector<int> input_padding,
		std::vector<int> filter_stride,
		std::vector<int> dilation)
		: cudnn_convolution(cudnn_context)
	{
		auto mode = CUDNN_CROSS_CORRELATION;// CUDNN_CONVOLUTION;
		auto type = internal_type();
#if 1
		result_ = cudnnSetConvolutionNdDescriptor(
			descriptor_,
			static_cast<unsigned int>(input_padding.size()),
			input_padding.data(),
			filter_stride.data(),
			dilation.data(),
			mode,
			type);
#else
		result = cudnnSetConvolution2dDescriptor(
			descriptor_,
			2, 2,
			1, 1,
			1, 1,
			mode,
			type);
#endif
	}

	virtual ~cudnn_convolution()
	{
		result_ = cudnnDestroyConvolutionDescriptor(descriptor_);
	}
	operator cudnnConvolutionDescriptor_t() const
	{
		return descriptor_;
	}

	auto get_output_size(const cudnnTensorDescriptor_t& input_tensor,
						 const cudnnFilterDescriptor_t& filter) -> std::vector<int>
	{
		std::vector<int>	output_dimensions;
		int					output_tensor_dimensionality;

#ifdef _DEBUG
		int					dummy;
		cudnnDataType_t		filter_type, input_type, conv_type;
		cudnnTensorFormat_t	filter_format;
		int					filter_dims, input_dims, conv_dims;
		cudnnConvolutionMode_t	conv_mode;
		std::vector<int>	filter_shape, input_shape, input_stride;
		int					filter_feature_index;

		result_ = cudnnGetFilterNdDescriptor(
			filter,
			1,
			&filter_type,
			&filter_format,
			&filter_dims,
			&dummy);

		filter_shape.resize(filter_dims);
		result_ = cudnnGetFilterNdDescriptor(
			filter,
			filter_dims,
			&filter_type,
			&filter_format,
			&filter_dims,
			filter_shape.data());

		result_ = cudnnGetTensorNdDescriptor(
			input_tensor,
			1,
			&input_type,
			&input_dims,
			&dummy,
			&dummy);

		input_shape.resize(input_dims);
		input_stride.resize(input_dims);
		result_ = cudnnGetTensorNdDescriptor(
			input_tensor,
			input_dims,
			&input_type,
			&input_dims,
			input_shape.data(),
			input_stride.data());

		result_ = cudnnGetConvolutionNdDescriptor(
			*this,
			1,
			&conv_dims,
			&dummy,
			&dummy,
			&dummy,
			&conv_mode,
			&conv_type);

		// check requirements:
		//	filter dimensionality = input_tensor dimensionality
		if (filter_dims != input_dims)
			throw std::exception("filter dimensionality != input dimensionality");

		//	convolution dimenionality = input_tensor dimensionality - 2
		if (conv_dims != input_dims - 2)
			throw std::exception("convolution dimensionality != input dimensionality - 2");

		//	filter features map == input_tensor features
		switch (filter_format)
		{
		case CUDNN_TENSOR_NCHW:
			filter_feature_index = 1;
			break;
		case CUDNN_TENSOR_NHWC:
			filter_feature_index = 0;
			break;
		default:
			throw std::exception("unknown filter_format");
		}

		if (filter_shape[filter_feature_index] != input_shape[filter_feature_index])
			throw std::exception("filter_shape feature != input_shape feature");

		//	dilated filter size > padded input tensor size


		output_tensor_dimensionality = input_dims;

		//	0 < output dimensionality <= input_tensor dimensionality
		if (output_tensor_dimensionality < 0)
			throw std::exception("output_tensor_dimensionality < 0");

		if (output_tensor_dimensionality > input_dims)
			throw std::exception("output_tensor_dimensionality > input_dims");
#endif
		output_dimensions.resize(output_tensor_dimensionality);

		result_ = cudnnGetConvolutionNdForwardOutputDim(
			static_cast<cudnnConvolutionDescriptor_t>(*this),
			input_tensor,
			filter,
			output_tensor_dimensionality,
			output_dimensions.data());

		return output_dimensions;
	}
};

template<typename T, typename U>
class cudnn_convolution_algorithm : public cudnn_result,
	public cuda_device_storage
{
public:
	std::vector<T>		algorithms;
	std::vector<size_t>	workspace_sizes;
	const int			default_query_size = 10;
protected:
	cudnnHandle_t		cudnn_context_;
	unsigned int		selected_algorithm_;
public:
	cudnn_convolution_algorithm(cudnnHandle_t cudnn_context)
		: cudnn_context_(cudnn_context),
		selected_algorithm_(0)
	{}
	virtual ~cudnn_convolution_algorithm()
	{}
	void select(unsigned int algorithm_index)
	{
		cuda_device_storage::request_device_storage(size());
	}
	operator U () const
	{
		return algorithms[selected_algorithm_].algo;
	}
	size_t size() const
	{
		return workspace_sizes[selected_algorithm_];
	}
	size_t count() const
	{
		return size();
	}
};
class cudnn_convolution_forward_algorithm : public cudnn_convolution_algorithm<cudnnConvolutionFwdAlgoPerf_t, cudnnConvolutionFwdAlgo_t>
{
public:
	cudnn_convolution_forward_algorithm(cudnnHandle_t cudnn_context,
		cudnnTensorDescriptor_t input_tensor,
		cudnnFilterDescriptor_t filter,
		cudnnConvolutionDescriptor_t convolution,
		cudnnTensorDescriptor_t output_tensor)
		: cudnn_convolution_algorithm(cudnn_context)
	{
		int result_size = default_query_size;

		algorithms.resize(4);
		workspace_sizes.resize(0);

		result_ = cudnnFindConvolutionForwardAlgorithm(
			cudnn_context,
			input_tensor,
			filter,
			convolution,
			output_tensor,
			static_cast<unsigned int>(algorithms.size()),
			&result_size,
			algorithms.data());

		for (auto& a : algorithms)
		{
			size_t exp_size = 0;

			cudnnGetConvolutionForwardWorkspaceSize(
				cudnn_context,
				input_tensor,
				filter,
				convolution,
				output_tensor,
				a.algo,
				&exp_size);

			workspace_sizes.push_back(exp_size);
		}

		select(0);
	}

	void apply(cudnnTensorDescriptor_t input_tensor, void *input_tensor_memory,
		cudnnTensorDescriptor_t output_tensor, void *output_tensor_memory,
		cudnnFilterDescriptor_t filter, void *filter_memory,
		cudnnConvolutionDescriptor_t convolution,
		cudnnConvolutionFwdAlgo_t algorithm,
		void *algorithm_workspace, size_t algorithm_workspace_size,
		const float *alpha,
		const float *beta)
	{
#ifdef _DEBUG
		// requires:
		//	At least one of the following is NULL: handle, xDesc, wDesc, convDesc, yDesc, xData, w, yData, alpha, beta
		if (cudnn_context_ == NULL)
			throw std::exception("cudnn context is null");
		if (input_tensor == NULL)
			throw std::exception("input tensor descriptor is null");
		if (input_tensor_memory == NULL)
			throw std::exception("input tensor memory is null");
		if (output_tensor == NULL)
			throw std::exception("output tensor descriptor is null");
		if (output_tensor_memory == NULL)
			throw std::exception("output tensor memory is null");
		if (filter == NULL)
			throw std::exception("filter descriptor is null");
		if (filter_memory == NULL)
			throw std::exception("filter memory is null");
		if (alpha == NULL)
			throw std::exception("alpha is null");
		if (beta == NULL)
			throw std::exception("beta is null");
		//	xDesc and yDesc have a non-matching number of dimensions 
		//	xDesc and wDesc have a non - matching number of dimensions
		//	xDesc has fewer than three number of dimensions
		//	xDesc's number of dimensions is not equal to convDesc's array length + 2
		//	xDesc and wDesc have a non - matching number of input feature maps per image(or group in case of Grouped Convolutions)
		//	yDesc or wDesc indicate an output channel count that isn't a multiple of group count (if group count has been set in convDesc). 
		//	xDesc, wDesc and yDesc have a non - matching data type
		//	For some spatial dimension, wDesc has a spatial size that is larger than the input spatial size(including zero - padding size)
#endif

		result_ = cudnnConvolutionForward(
			cudnn_context_,
			alpha,
			input_tensor,
			input_tensor_memory,
			filter,
			filter_memory,
			convolution,
			algorithm,
			algorithm_workspace,
			algorithm_workspace_size,
			beta,
			output_tensor,
			output_tensor_memory);
	}

	template<typename T>
	void apply(const cudnn_tensor<T>& input_tensor,
		const cudnn_tensor<T>& output_tensor,
		const cudnn_filter<T>& filter,
		const cudnn_convolution<T>& convolution,
		const float *alpha,
		const float *beta)
	{
		apply(
			input_tensor,
			input_tensor.device_storage(),
			output_tensor,
			output_tensor.device_storage(),
			filter,
			filter.device_storage(),
			convolution,
			*this,
			device_storage(),
			size(),
			alpha,
			beta);
	}
};
class cudnn_convolution_backward_filter_algorithm : public cudnn_convolution_algorithm<cudnnConvolutionBwdFilterAlgoPerf_t, cudnnConvolutionBwdFilterAlgo_t>
{
public:
	cudnn_convolution_backward_filter_algorithm(cudnnHandle_t cudnn_context,
		cudnnTensorDescriptor_t input_tensor,
		cudnnFilterDescriptor_t filter,
		cudnnConvolutionDescriptor_t convolution,
		cudnnTensorDescriptor_t diff_tensor)
		: cudnn_convolution_algorithm(cudnn_context)
	{
		int result_size = default_query_size;

		algorithms.resize(4);
		workspace_sizes.resize(0);

		result_ = cudnnFindConvolutionBackwardFilterAlgorithm(
			cudnn_context,
			input_tensor,
			diff_tensor,
			convolution,
			filter,
			static_cast<unsigned int>(algorithms.size()),
			&result_size,
			algorithms.data());

		for (auto& a : algorithms)
		{
			size_t exp_size = 0;

			cudnnGetConvolutionBackwardFilterWorkspaceSize(
				cudnn_context,
				input_tensor,
				diff_tensor,
				convolution,
				filter,
				a.algo,
				&exp_size);

			workspace_sizes.push_back(exp_size);
		}

		select(0);
	}
	template<typename T>
	void apply(const cudnn_tensor<T>& input_tensor,
		const cudnn_tensor<T>& diff_tensor,
		const cudnn_filter<T>& filter_gradient,
		const cudnn_convolution<T>& convolution,
		const float* alpha,
		const float* beta)
	{
#ifdef _DEBUG
		auto input_tensor_type = input_tensor.internal_type();
		auto diff_tensor_type = diff_tensor.internal_type();
		auto filter_gradient_type = filter_gradient.internal_type();
		auto filter_gradient_format = filter_gradient.format();
		auto algo = static_cast<cudnnConvolutionBwdFilterAlgo_t>(*this);

		// FIXME: encode all the checks from 
		// http://docs.nvidia.com/deeplearning/sdk/cudnn-user-guide/index.html#cudnnConvolutionBackwardFilter
		// to give some indication as to why CUDNN_STATUS_NOT_SUPPORTED is returned
		switch (algo)
		{
		case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0:
		{
			break;
		}
		}
#endif
		result_ = cudnnConvolutionBackwardFilter(
			cudnn_context_,
			alpha,
			input_tensor,
			input_tensor.device_storage(),
			diff_tensor,
			diff_tensor.device_storage(),
			convolution,
			*this,
			device_storage(),
			size(),
			beta,
			filter_gradient,
			filter_gradient.device_storage());
	}
};
class cudnn_convolution_backward_data_algorithm : public cudnn_convolution_algorithm < cudnnConvolutionBwdDataAlgoPerf_t, cudnnConvolutionBwdDataAlgo_t >
{
public:
	cudnn_convolution_backward_data_algorithm(cudnnHandle_t cudnn_context,
		cudnnTensorDescriptor_t output_tensor,
		cudnnFilterDescriptor_t filter,
		cudnnConvolutionDescriptor_t convolution,
		cudnnTensorDescriptor_t diff_tensor)
		: cudnn_convolution_algorithm(cudnn_context)
	{
		int result_size = default_query_size;

		algorithms.resize(4);
		workspace_sizes.resize(0);

		result_ = cudnnFindConvolutionBackwardDataAlgorithm(
			cudnn_context,
			filter,
			diff_tensor,
			convolution,
			output_tensor,
			static_cast<unsigned int>(algorithms.size()),
			&result_size,
			algorithms.data());

		for (auto& a : algorithms)
		{
			size_t exp_size = 0;

			cudnnGetConvolutionBackwardDataWorkspaceSize(
				cudnn_context,
				filter,
				diff_tensor,
				convolution,
				output_tensor,
				a.algo,
				&exp_size);

			workspace_sizes.push_back(exp_size);
		}

		select(0);
	}

	template<typename T>
	void apply(const cudnn_tensor<T>& input,
			   const cudnn_tensor<T>& error_gradient,
			   const cudnn_filter<T>& filter,
			   const cudnn_convolution<T>& convolution,
			   const cudnn_tensor<T>& input_gradient,
			   const float *alpha,
			   const float *beta)
	{
		result_ = cudnnConvolutionBackwardData(
			cudnn_context_,
			alpha,
			filter,
			filter.device_storage(),
			error_gradient,
			error_gradient.device_storage(),
			convolution,
			*this,
			device_storage(),
			size(),
			beta,
			input_gradient,
			input_gradient.device_storage());
	}
};

template<typename T>
class cudnn_convolution_layer : public cudnn_layer<T>
{
public:
	cudnn_convolution<T>						convolution;
	cudnn_filter<T>								filter;
	cudnn_filter<T>								filter_gradient;
	cudnn_tensor<T>								gradient;
	cudnn_convolution_forward_algorithm			forward_;
	cudnn_convolution_backward_filter_algorithm	backward_filter_;
	cudnn_convolution_backward_data_algorithm	backward_data_;

public:
	cudnn_convolution_layer(cudnnHandle_t cudnn_context,
							std::vector<int> input_padding,
							std::vector<int> filter_stride,
							std::vector<int> dilation,
							std::vector<int> filter_size,
							const cudnn_tensor<T>& input,
							const cudnn_tensor<T>& output)
			: cudnn_layer(cudnn_context),
			  convolution(cudnn_context, input_padding, filter_stride, dilation),
			  filter(filter_size),
			  filter_gradient(filter_size),
//			  output(cudnn_context, convolution.get_output_size(input_tensor, filter)),
			  gradient(cudnn_context, convolution.get_output_size(input, filter)),
			  forward_(cudnn_context, input, filter, convolution, output),
			  backward_filter_(cudnn_context, input, filter, convolution, gradient),
			  backward_data_(cudnn_context, output, filter, convolution, gradient)
	{
	}

	void forward(const cudnn_tensor<T>& input, const cudnn_tensor<T>& output)
	{
		forward_.apply(input, output, filter, convolution, alpha (), beta ());
	}

	void backward(const cudnn_tensor<T>& input, const cudnn_tensor<T>& input_gradient,
				  const cudnn_tensor<T>& error, const cudnn_tensor<T>& error_gradient)
	{
		// compute the gradient filter first
		backward_filter_.apply(
			input,
			input_gradient,
			filter_gradient,
			convolution,
			alpha (),
			beta ());

		// compute the error propagation second
		backward_data_.apply(
			input,
			input_gradient,
			filter,
			convolution,
			error,
			alpha (),
			beta ());
	}
};
#endif