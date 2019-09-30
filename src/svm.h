#pragma once

#include "loadData.h"
#include "kernel.h"
#include <vector>
#include <cassert>
#include <iostream>
#include <stdexcept>


class SVM {
public:
	SVM(std::vector<double> const& dual_variables, sparseData& data, Kernel const& kernel)
	: m_kernel(&kernel)
	{
		for (INDEX index = 0; index < dual_variables.size(); index++) {
			if (dual_variables[index] == 0)	continue;
			support_vectors.push_back(data.data[index]);
			multipliers.push_back(dual_variables[index]);
			sv_labels.push_back(data.labels[index]);
		}
	}

	// enable copy and move
	SVM(SVM const&) = default;
	SVM(SVM &&) = default;
	SVM& operator = (SVM const&) = default;
	SVM& operator = (SVM&&) = default;

	char classifyPoint(std::vector<SE> const& test_point) const {
		//Decision Function evaluation
		double sum = 0;
		for (INDEX i = 0; i < support_vectors.size(); i++) {
			if(support_vectors[i].size() == 0) continue;
			sum += multipliers[i] * sv_labels[i] * m_kernel->evaluate(test_point, support_vectors[i]);
		}
		char decision = (sum > 0) - (sum < 0);
		if (support_vectors.size() == 0) {
			std::cout << "*" << "No support vectors" << std::endl;
			throw std::runtime_error("No support vectors for classification");
		}
		return decision;
	}

	double evaluateTestset(const sparseData& test_data) const {
		std::size_t number_of_errors = 0;
		for (INDEX i = 0; i < test_data.data.size(); i++) {
			char classification = classifyPoint(test_data.data[i]);
			if (classification != test_data.labels[i]) number_of_errors++;
		}
		double success_rate = 1 - ((double)number_of_errors/(double)test_data.labels.size());
		return success_rate;
	}

	std::size_t size() const {
		return support_vectors.size();		
	}

private:
	// attributes
	std::vector<std::vector<SE>> support_vectors;
	std::vector<double> multipliers;
	std::vector<double> sv_labels;
	Kernel const* m_kernel;
};
