#include "cudnn_result.h"
#include "cudnn_internal_type.h"
#include "cuda_device_storage.h"

#include <numeric>

#ifndef CUDNN_TENSOR_H
#define CUDNN_TENSOR_H
template<typename T>
class cudnn_tensor : public cudnn_result,
					 public cudnn_internal_type<T>,
					 public cuda_device_storage
{
public:
	static std::vector<int> create_stride_pattern(const std::vector<int>& sizes)
	{
		std::vector<int> strides(sizes.size());

		std::partial_sum(
			std::begin(sizes),
			std::end(sizes),
			std::begin(strides),
			[](int x, int y){return x*y; });

		std::reverse(
			std::begin(strides),
			std::end(strides));

		return strides;
	}

protected:
	cudnnHandle_t			cudnn_handle_;
	cudnnTensorDescriptor_t	descriptor_;

public:
	cudnn_tensor(cudnnHandle_t cudnn_handle)
		: descriptor_(NULL),
		  cudnn_handle_(cudnn_handle)
	{
		result_ = cudnnCreateTensorDescriptor(&descriptor_);
	}
	cudnn_tensor(cudnnHandle_t cudnn_handle, const std::vector<int>& sizes, const std::vector<int>& strides)
		: cudnn_tensor(cudnn_handle)
	{
#if 0
		result = cudnnSetTensorNdDescriptor(
			descriptor_,
			internal_type(),
			sizes.size(),
			sizes.data(),
			strides.data());
#else		
		result_ = cudnnSetTensor4dDescriptor(
			descriptor_,
			CUDNN_TENSOR_NCHW,
			internal_type(),
			sizes[0],
			sizes[1],
			sizes[2],
			sizes[3]);
#endif

		cuda_device_storage::request_device_storage(size());
	}
	cudnn_tensor(cudnnHandle_t cudnn_handle, const std::vector<int>& sizes)
		: cudnn_tensor(cudnn_handle, sizes, create_stride_pattern(sizes))
	{}
	cudnn_tensor(cudnnHandle_t cudnn_handle, const std::vector<int>& sizes, std::function<T()> initial_values)
		: cudnn_tensor(cudnn_handle, sizes)
	{
		cuda_device_storage::fill(initial_values);
	}
	cudnn_tensor(cudnnHandle_t cudnn_handle, const std::vector<int>& sizes, const T initial_value)
		: cudnn_tensor(cudnn_handle, sizes)
	{
		fill(initial_value);
	}
	virtual ~cudnn_tensor()
	{
		result_ = cudnnDestroyTensorDescriptor(descriptor_);
	}

	operator cudnnTensorDescriptor_t () const { return descriptor_; }

	size_t size() const
	{
		size_t size_in_bytes = 0;
		// NB: if we modify result, we lose constness...
		/*result = */cudnnGetTensorSizeInBytes(
			descriptor_,
			&size_in_bytes);
		return size_in_bytes;
	}
	
	size_t count() const
	{
		return (size() / sizeof(T));
	}
	
	void fill(const T value)
	{
		result_ = cudnnSetTensor(
			cudnn_handle_,
			*this,
			device_storage(),
			&value);
	}

	int dimensionality() const
	{
		cudnnDataType_t		data_type;
		int					dimensionality = 1;
		std::vector<int>	dims(1);
		std::vector<int>	stride(1);

		cudnnGetTensorNdDescriptor(
			this,
			1,
			&data_type,
			&dimensionality,
			dims.data(),
			stride.data());

		return dimensionality;
	}

	std::vector<int> dimensions() const
	{
		cudnnDataType_t	data_type;
		int				dimensionality = 1;
		std::vector<int>	dims(1);
		std::vector<int>	stride(1);

		/*result = */cudnnGetTensorNdDescriptor(
			*this,
			1,
			&data_type,
			&dimensionality,
			dims.data(),
			stride.data());

		dims.resize(dimensionality);
		stride.resize(dimensionality);

		/*result = */cudnnGetTensorNdDescriptor(
			*this,
			dimensionality,
			&data_type,
			&dimensionality,
			dims.data(),
			stride.data());

		return dims;
	}

	void get_description()
	{
		cudnnDataType_t	data_type;
		int				dimensionality = 1;
		std::vector<int>	dims(1);
		std::vector<int>	stride(1);

		result = cudnnGetTensorNdDescriptor(
			this,
			1,
			&data_type,
			&dimensionality,
			dims.data(),
			stride.data());

		dims.resize(dimensionality);
		stride.resize(dimensionality);

		result = cudnnGetTensorNdDescriptor(
			this,
			dimensionality,
			&data_type,
			&dimensionality,
			dims.data(),
			stride.data());
	}
};
#endif