#pragma once

#include <vector>
#include <algorithm>
#include <cassert>
#include <string>
#include <random>
#include <tuple>
#include <cmath>


typedef std::size_t INDEX;
typedef double VALUE; 


struct SE
{
	SE()
	{ }

	SE(INDEX i, VALUE v)
	: index(i)
	, value(v)
	{ }

	INDEX index;
	VALUE value;
}; 

// end-of-sparse-vector marker
inline constexpr std::size_t END()
{ return std::numeric_limits<std::size_t>::max(); }

template<typename T>
void permuteVector(std::vector<T>& permutee, std::vector<INDEX>& indices) {
	assert(permutee.size() == indices.size());
	std::size_t size = permutee.size();
	std::vector<T> tmp(size);
	for (INDEX i = 0; i < size; i++) {
		tmp[i] = permutee[indices[i]];
	}
	permutee = std::move(tmp);
}

struct sparseData {
	std::vector<std::vector<SE>> data;
	std::vector<char> labels;
	std::size_t dimension;
	std::vector<SE> mean();

	void delete_element(INDEX index) {
		data[index] = data.back();
		labels[index] = labels.back();
		labels.pop_back();
		data.pop_back();
	}

	void shuffle() {
		std::size_t dataset_size = data.size();
		// Initialize vector with possible indices and shuffle 
		std::vector<INDEX> indices(dataset_size, 0);
		auto walker = indices.begin();
		auto end = indices.end();
		INDEX inc = 0;
		while (walker != end) {
			*walker++ = inc;
			++inc;
		}
		std::random_shuffle(indices.begin(), indices.end());
		permuteVector(data, indices);
		permuteVector(labels, indices);
	}
	void shuffle_ds_dualvec(std::vector<double>& dual_vector) {
		std::size_t dataset_size = data.size();
		// Initialize vector with possible indices and shuffle
		std::vector<INDEX> indices(dataset_size, 0);
		auto walker = indices.begin();
		auto end = indices.end();
		INDEX inc = 0;
		while (walker != end) {
			*walker++ = inc;
			++inc;
		}
		std::random_shuffle(indices.begin(), indices.end());
		permuteVector(data, indices);
		permuteVector(labels, indices);
		permuteVector(dual_vector, indices);
	}
};

void loaddata(std::string filename, sparseData& data, bool shuffle);
std::vector<SE> scaleAddSparseVectors_new(std::vector<SE> const& first_operand, std::vector<SE> const& second_operand, double first_scale, double second_scale);
std::tuple<std::vector<double>, std::vector<double>, std::vector<double> > load_lookup(std::string filename);
