#include <vector>
#include <numeric>

#include <cuda.h>
#include <cudnn.h>

#include "cuda_initialise.h"
#include "cudnn_initialise.h"
#include "cudnn_filter.h"

int main(int argc, char** argv)
{
	cuda_initialise	cuda;
	cudnn_initialise cudnn;

	std::vector <float> filter_values;

	cudnn_filter<float>	filter({ 1, 1, 5, 5 });

	filter_values.resize(filter.count());

	std::fill(
		std::begin(filter_values),
		std::end(filter_values),
		2.5f);

	filter.set(filter_values);

	return 0;
}