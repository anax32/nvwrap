#include "cudnn_result.h"
#include "cudnn_internal_type.h"
#include "cuda_device_storage.h"

#include <algorithm>
#include <functional>
#include <numeric>

#ifndef CUDNN_FILTER_H
#define CUDNN_FILTER_H
template<typename T>
class cudnn_filter : public cudnn_result,
					 public cudnn_internal_type<T>,
					 public cuda_device_storage
{
protected:
	cudnnFilterDescriptor_t	descriptor_;
public:
	cudnn_filter()
		: descriptor_(NULL)
	{
		result_ = cudnnCreateFilterDescriptor(&descriptor_);
	}
	cudnn_filter(std::vector<int> dimensions)
		: cudnn_filter()
	{
		auto type = internal_type();
		auto format = CUDNN_TENSOR_NCHW;

		if (dimensions.size() < 4)
		{
			throw std::exception("dimensions.size must > 3, pad with ones");
		}

#if 0
		result = cudnnSetFilterNdDescriptor(
			descriptor_,
			type,
			format,
			dimensions.size(),
			dimensions.data());
#else
		result_ = cudnnSetFilter4dDescriptor(
			descriptor_,
			type,
			format,
			dimensions[0],
			dimensions[1],
			dimensions[2],
			dimensions[3]);
#endif
		cuda_device_storage::request_device_storage(size());
	}
	cudnn_filter(std::vector<int> dimensions, std::function<T()> initial_values)
		: cudnn_filter(dimensions)
	{
		fill(initial_values);
	}
	cudnn_filter(std::vector<int> dimensions, const T initial_value)
		: cudnn_filter(dimensions)
	{
		fill(initial_value);
	}
	virtual ~cudnn_filter()
	{
		result_ = cudnnDestroyFilterDescriptor(descriptor_);
	}
	operator cudnnFilterDescriptor_t () const { return descriptor_; }

	cudnnTensorFormat_t format() const
	{
		std::vector<int>	dims(1);
		cudnnDataType_t		datatype;
		cudnnTensorFormat_t	format;
		int					dimensionality = 1;

		cudnnGetFilterNdDescriptor(
			descriptor_,
			dimensionality,
			&datatype,
			&format,
			&dimensionality,
			dims.data());

		return format;
	}

	std::vector<int> dimensions() const
	{
		std::vector<int>	dims(1);
		cudnnDataType_t		datatype;
		cudnnTensorFormat_t	format;
		int					dimensionality = 1;

		cudnnGetFilterNdDescriptor(
			descriptor_,
			dimensionality,
			&datatype,
			&format,
			&dimensionality,
			dims.data());

		dims.resize(dimensionality);

		cudnnGetFilterNdDescriptor(
			descriptor_,
			dimensionality,
			&datatype,
			&format,
			&dimensionality,
			dims.data());

		return dims;
	}

	size_t count() const
	{
		std::vector<int>	dims(1);
		cudnnDataType_t		datatype;
		cudnnTensorFormat_t	format;
		int					actual_dimensionality = 1;

		// call with dummy information to get the size
		/*result = */cudnnGetFilterNdDescriptor(
			descriptor_,
			actual_dimensionality,
			&datatype,
			&format,
			&actual_dimensionality,
			dims.data());

		if (datatype != internal_type())
			return 0;

		// resize the array, call properly
		dims.resize(actual_dimensionality);

		/*result = */cudnnGetFilterNdDescriptor(
			descriptor_,
			actual_dimensionality,
			&datatype,
			&format,
			&actual_dimensionality,
			dims.data());

		auto prod = std::accumulate(
			std::begin(dims),
			std::end(dims),
			1,
			[](int x, int y) { return x*y; });

		return prod;
	}

	size_t size() const
	{
		return count() * sizeof(T);
	}
};
#endif