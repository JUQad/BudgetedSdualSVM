#pragma once

#include "loadData.h"
#include <vector>
#include <cmath>

class Kernel {
public:
	virtual ~Kernel() {}
	virtual double evaluate(const std::vector<SE>& x1, const std::vector<SE>& x2) const = 0;
};

// Gaussian kernel
class RbfKernel : public Kernel
{
public:
	RbfKernel(double gamma)
	: m_minus_gamma(-gamma)
	{ }

	double evaluate(const std::vector<SE>& x1, const std::vector<SE>& x2) const override {//, const double scaling=1) {
		using namespace std;

		if (&x1 == &x2) {
			return 1.0;
		}

		const SE* x1_ref = &x1[0];
		const SE* x2_ref = &x2[0];
		size_t i = 0;
		size_t j = 0;
		double dist2 = 0.0;
		while ((x1_ref[i].index != END()) || ((INDEX)x2_ref[j].index != END())) {
			
			if (x1_ref[i].index  < x2_ref[j].index) {
				VALUE diff = x1_ref[i].value;
				dist2 += diff * diff;
				i++;
			}

			else if (x1_ref[i].index > x2_ref[j].index) {
				VALUE diff = x2_ref[j].value;
				dist2 += diff * diff;
				j++;
			}

			else if (x1_ref[i].index == x2_ref[j].index) {
				VALUE diff = (x1_ref[i].value - x2_ref[j].value);
				dist2 += diff * diff;
				i++;
				j++;
			}
		}
		double result = std::exp(m_minus_gamma * dist2);
		return result;
	}

double getGamma() {
	return m_minus_gamma*(-1);
}
void setGamma(double gamma) {
	m_minus_gamma = -gamma;
}

protected:
	double m_minus_gamma;
};
