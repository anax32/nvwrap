#include <cuda_runtime_api.h>

#include <array>
#include <vector>
#include <algorithm>
#include <functional>

#ifndef CUDA_DEVICE_STORAGE_H
#define CUDA_DEVICE_STORAGE_H
class cuda_device_storage
{
private:
	// not copyable because of the allocation pointer
	cuda_device_storage(const cuda_device_storage& a) {}

protected:
	void*	device_storage_;
	size_t	device_capacity_;

	void request_device_storage(size_t size)
	{
		if (size != device_capacity_)
		{
			free_device_storage();

			if (size > 0)
			{
				cudaMalloc(&device_storage_, size);
			}

			device_capacity_ = size;
		}
	}
	void free_device_storage()
	{
		if (device_storage_ != NULL)
			cudaFree(device_storage_);

		device_storage_ = NULL;
		device_capacity_ = 0;
	}
public:
	cuda_device_storage()
		: device_storage_(NULL),
		device_capacity_(0)
	{}
	virtual ~cuda_device_storage()
	{
		free_device_storage();
	}
	void *device_storage() const
	{
		return device_storage_;
	}

	// puts a range of values into the device memory
	template<typename T>
	void set(const std::vector<T>& values)
	{
		auto byte_length = values.size() * sizeof(T);

		request_device_storage(byte_length);

		cudaMemcpy(
			device_storage(),
			values.data(),
			byte_length,
			cudaMemcpyHostToDevice);
	}
	template<typename T, size_t N>
	void set(const std::array<T, N>& values)
	{
		std::vector<T> v(N);

		std::copy(
			std::begin(values),
			std::end(values),
			std::begin(v));

		set(v);
	}

	// gets the range of values from device memory.
	// NB: buffer is resized if it does not match.
	template<typename T>
	void get(std::vector<T>& buffer) const
	{
		auto device_elements = device_capacity_ / sizeof(T);

		if (buffer.size() != device_elements)
		{
			buffer.resize (device_elements);
		}

		cudaMemcpy(
			buffer.data(),
			device_storage(),
			device_capacity_,
			cudaMemcpyDeviceToHost);
	}

	// fill the memory with values from a generator
	template<typename T>
	void fill(std::function<T()> generator)
	{
		std::vector<T>	values(count());

		std::generate(
			std::begin(values),
			std::end(values),
			generator);

		set(values);
	}

	// fill the memory with a fixed value
	template<typename T>
	void fill(const T value)
	{
		std::vector<T>	values(count());

		std::fill(
			std::begin(value),
			std::end(value),
			value);

		set(values);
	}

	// number of elements in the memory
	virtual size_t count() const = NULL;
	// size in bytes of the memory
	virtual size_t size() const = NULL;
};
#endif