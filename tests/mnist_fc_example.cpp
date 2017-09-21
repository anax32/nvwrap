#include <cuda.h>
#include <cudnn.h>

#include "cuda_initialise.h"
#include "cudnn_initialise.h"
#include "cudnn_tensor.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <array>

// read the mnist data
#define MNIST_TESTING_IMAGES	"../data/mnist-t10k-images.idx3-ubyte"
#define MNIST_TESTING_LABELS	"../data/mnist-t10k-labels.idx1-ubyte"
#define	MNIST_TRAINING_IMAGES	"../data/mnist-train-images.idx3-ubyte"
#define	MNIST_TRAINING_LABELS	"../data/mnist-train-labels.idx1-ubyte"

typedef unsigned char						label_t;
typedef float								output_t;
typedef	std::array<unsigned char, 28*28>	data_t;
typedef std::array<float, 28*28>			image_t;

typedef std::pair <label_t, data_t>			pair_t;

template<typename T>
void swap(T& value)
{
	char *s = reinterpret_cast<char*>(&value);
	char *n = reinterpret_cast<char*>(&value) + sizeof(T);
	std::reverse(s, n);
}

auto read_images(const char *image_path) -> std::vector <data_t>
{
	std::ifstream		fimages(image_path, std::ios::binary);
	std::vector<data_t>	images;
	size_t				len, exp_len;
	unsigned int		magic;
	unsigned int		count;
	unsigned int		rows, cols;

	if (fimages.good() == false)
		return images;

	fimages.seekg(0, std::ios::end);
	len = fimages.tellg();
	fimages.seekg(0, std::ios::beg);
	len -= fimages.tellg();
	
	fimages.read((char*)&magic, sizeof(magic));
	fimages.read((char*)&count, sizeof(count));
	fimages.read((char*)&rows, sizeof(rows));
	fimages.read((char*)&cols, sizeof(cols));

	swap(magic);
	swap(count);
	swap(rows);
	swap(cols);
	exp_len = (sizeof(unsigned int) * 4) + (sizeof(data_t)*count);
	
	// check the data
	if (magic != 2051)
		return images;

	if ((rows != 28) || (cols != 28))
		return images;

	if (len != exp_len)
		return images;

	images.resize(count);
	fimages.read((char*)images.data(), count);
	fimages.close();

	return images;
}
auto read_labels(const char *labels_path) -> std::vector <label_t>
{
	std::ifstream			flabels(labels_path, std::ios::binary);
	std::vector<label_t>	labels;
	size_t					len, exp_len;
	unsigned int			magic;
	unsigned int			count;

	if (flabels.good() == false)
		return labels;

	flabels.seekg(0, std::ios::end);
	len = flabels.tellg();
	flabels.seekg(0, std::ios::beg);
	len -= flabels.tellg();

	flabels.read((char*)&magic, sizeof(magic));
	flabels.read((char*)&count, sizeof(count));

	swap(magic);
	swap(count);
	exp_len = (sizeof(unsigned int) * 2) + (sizeof(label_t)*count);

	// check the data
	if (magic != 2049)
		return labels;

	if (len != exp_len)
		return labels;

	labels.resize(count);
	flabels.read((char*)labels.data(), count);
	flabels.close();

	return labels;
}

auto rescale (const std::vector<data_t>& byte_images) -> std::vector < image_t >
{
	std::vector<image_t>	float_images (byte_images.size());

	std::transform(
		std::begin(byte_images),
		std::end(byte_images),
		std::begin(float_images),
		[](const data_t& byte_array) 
		{
			image_t	float_array;

			std::transform(
				std::begin(byte_array),
				std::end(byte_array),
				std::begin(float_array),
				[](const data_t::value_type x)
				{
					return ((float)x) / 255.0f;
				});

			return float_array;
		});

	return float_images;
}

auto rescale(const std::vector<label_t>& labels) -> std::vector < output_t >
{
	std::vector<output_t>	output (labels.size());

	std::transform(
		std::begin(labels),
		std::end(labels),
		std::begin(output),
		[](const label_t& label)
		{
			return ((float)label);
		});

	return output;
}

int main(int argc, char** argv)
{
	cuda_initialise	cuda;
	cudnn_initialise cudnn;

	// read the data
	auto training_images = read_images(MNIST_TRAINING_IMAGES);
	auto training_labels = read_labels(MNIST_TRAINING_LABELS);
	auto testing_images = read_images(MNIST_TESTING_IMAGES);
	auto testing_labels = read_labels(MNIST_TESTING_LABELS);

	auto training_images_f = rescale (training_images);
	auto training_labels_f = rescale (training_labels);

	// create the tensors
	cudnn_tensor<float>	training_images_tensor(
		cudnn,
		{ 1, 1, 28, 28 },
		0.0f);

	training_images_tensor.set(training_images_f);

	return 0;
}