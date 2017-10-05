#include <iostream>
#include <map>
#include <vector>
#include <algorithm>

#include <cuda.h>
#include <cudnn.h>

#include "cuda_initialise.h"
#include "cudnn_initialise.h"
#include "cudnn_tensor.h"

int main(int argc, char** argv)
{
	cuda_initialise	cuda;
	cudnn_initialise cudnn;

	// test the internal types and template types
	{
		cudnn_tensor<double> double_tensor(cudnn);
		cudnn_tensor<float> float_tensor(cudnn);
		cudnn_tensor<int> int_tensor(cudnn);
		cudnn_tensor<char> byte_tensor(cudnn);

		auto test_results = {
			double_tensor.internal_type() == CUDNN_DATA_DOUBLE,
			float_tensor.internal_type() == CUDNN_DATA_FLOAT,
			int_tensor.internal_type() == CUDNN_DATA_INT32,
			byte_tensor.internal_type() == CUDNN_DATA_INT8
		};

		if (std::all_of(
			std::begin(test_results),
			std::end(test_results),
			[](const bool b)
      {
        return b == true;
      }) == false)
		{
			return 1;
		}

	}

	// test internal buffer sizes
	{
		cudnn_tensor<double> double_tensor(cudnn, { 1, 1, 1, 64 }, { 0, 0, 0, 1 });
		cudnn_tensor<float> float_tensor(cudnn, { 1, 1, 1, 64 }, { 0, 0, 0, 1 });
		cudnn_tensor<int> int_tensor(cudnn, { 1, 1, 1, 64 }, { 0, 0, 0, 1 });
		cudnn_tensor<char> byte_tensor(cudnn, { 1, 1, 1, 64 }, { 0, 0, 0, 1 });

		auto test_results = {
			double_tensor.size() == double_tensor.count() * sizeof(double),
			float_tensor.size() == float_tensor.count () * sizeof(float),
			int_tensor.size() == int_tensor.count () * sizeof(int),
			byte_tensor.size() == byte_tensor.count () * sizeof(char)
		};

		if (std::all_of(
			std::begin(test_results),
			std::end(test_results),
			[](const bool b)
      {
        return b == true;
      }) == false)
		{
			return 2;
		}
	}

	// test tensor buffer set and get
	{
		// create a tensor
		// stride == std::partial_sum reversed?
		cudnn_tensor<float>	tensor(cudnn, { 10, 8, 64, 64 }, { 8 * 64 * 64, 64 * 64, 64, 1 });

		// create some duff data
		std::vector<float>	send(tensor.count());
		std::vector<float>	recv(send.size());
		unsigned int		i = 0;
		std::generate(
        std::begin(send),
        std::end(send),
        [&i]()
        {
            return static_cast<float>(++i);
        });

		std::fill(
        std::begin(recv),
        std::end(recv),
        1.0f);

		// send it to the device and read it back
		auto read_mismatch = !std::equal(
        std::begin(recv),
        std::end(recv),
        std::begin(send));

		tensor.set(send);
		tensor.get(recv);
		auto read_match = std::equal(
        std::begin(recv),
        std::end(recv),
        std::begin(send));

		// clear the tensor on the device
		tensor.fill(1.0f);

		// read back the tensor with the fill values
		std::fill(std::begin(recv), std::end(recv), 0.0f);
		auto all_zero = std::count_if(
        std::begin(recv),
        std::end(recv),
        [](float x)
        {
            return x == 0.0f;
        }) == recv.size();

		tensor.get(recv);
		auto all_ones = std::count_if(
        std::begin(recv),
        std::end(recv),
        [](float x)
        {
            return x == 1.0f;
        }) == recv.size();

		auto test_results = {
			read_mismatch == true,
			read_match == true,
			all_zero == true,
			all_ones == true
		};

		if (std::all_of(
			std::begin(test_results),
			std::end(test_results),
			[](const bool b)
      {
        return b == true;
      }) == false)
		{
			return 3;
		}
	}

	// raii cleanup
	return 0;
}