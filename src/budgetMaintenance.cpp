
#include "loadData.h"
#include "kernel.h"
#include "solver.h"
#include <fstream>
#include <cmath>

using namespace std;

vector<INDEX> randomSequence(size_t number_of_points) {
	vector<INDEX> sequence(number_of_points);
	for (size_t counter = 0; counter < number_of_points; counter++) sequence[counter] = counter;
	random_shuffle(sequence.begin(), sequence.end());
	return sequence;
}


// enable exactly one of the following:
#define GSS
//#define GSS_HIGH_PRECISION
//#define LOOKUP_H
//#define LOOKUP_WD

//#define sameLabel



struct dv_struct {
	double dv_value;
	INDEX dv_index;
};

struct by_value {
	bool operator()(dv_struct const &a, dv_struct const &b) {
		return a.dv_value < b.dv_value;
	}
};

struct by_index {
	bool operator()(dv_struct const &a, dv_struct const &b) {
		return a.dv_index < b.dv_index;
	}
};





double squaredWeightDegradation(double kernelmn, double kernelmz, double kernelnz, double alpha_m, double alpha_n, double alpha_z) {
	return (alpha_m*alpha_m) * 1.0 // kernel(m, m)
		 + (alpha_n*alpha_n) * 1.0 // kernel(n, n)
		 - (alpha_z*alpha_z) * 1.0 // kernel(z, z)
		 + 2*alpha_m*alpha_n * kernelmn;
}

double bilinearInterpolation(double q11, double q12, double q21, double q22, double x1, double x2, double y1, double y2, double x, double y){
	double x2x1	= x2 - x1;
	double y2y1	= y2 - y1;
	double x2x	 = x2 - x;
	double y2y	 = y2 - y;
	double yy1	 = y - y1;
	double xx1	 = x - x1;
	return  1.0 / (x2x1 * y2y1) * (
								  q11 * x2x * y2y +
								  q21 * xx1 * y2y +
								  q12 * x2x * yy1 +
								  q22 * xx1 * yy1
								  );
}

// look up a value from the table with bilinear interpolation
double lookUpTable(vector<double> const& m_table, vector<double> const& k_table, vector<double> const& wd_table, double kernel12, double m)
{
	size_t m_gridDim		= m_table.size();
	double m_gridStepSize   = m_table[1]-m_table[0];

	double x = m;
	double y = kernel12;

	double m_initial = m_table[0];
	double k_initial = k_table[0];

	size_t index_m_before = floor( (m-m_initial)/m_gridStepSize);
	size_t index_m_after  = index_m_before + 1;
	size_t index_k_before = floor( (kernel12 - k_initial)/m_gridStepSize);
	size_t index_k_after  = index_k_before + 1;

	double x1  = m_table[index_m_before];
	double x2  = m_table[index_m_after];

	double y1  = k_table[index_k_before];
	double y2  = k_table[index_k_after];

	double q11 = wd_table[index_m_before*m_gridDim + index_k_before];
	double q12 = wd_table[index_m_before*m_gridDim + index_k_after];

	double q22 = wd_table[index_m_after*m_gridDim + index_k_after];
	double q21 = wd_table[index_m_after*m_gridDim + index_k_before];

	return bilinearInterpolation(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
}

// objective function for the Golden Section Search
double objective(double kernel12, double m, double x)
{
	double kernel1z = pow(kernel12, (1.0 - x) * (1.0 - x));
	double kernel2z = pow(kernel12, x * x);
	return m * kernel1z + (1 - m) * kernel2z;
}

double goldenSectionSearch(double kernel12, double m, double a, double b, double epsilon) {
	double gratio = (sqrt(5.0) - 1.0) / 2.0;
	//double b = 1.0;
	//double a = 0.0;
	double p = b - gratio * (b - a);
	double q = a + gratio * (b - a);

	double fp = objective(kernel12, m, p);
	double fq = objective(kernel12, m, q);
	while ((b - a) >=  epsilon)
	{
		if (fp >= fq)
		{
			b = q;
			q = p;
			fq = fp;

			p = b - gratio * (b - a);
			fp = objective(kernel12, m, p);
		}
		else
		{
			a = p;
			p = q;
			fp = fq;

			q = a + gratio * (b - a);
			fq = objective(kernel12, m, q);
		}
	}

	return ((a + b) / 2.0);
}

tuple<INDEX, INDEX> mergeHeuristicWD(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C)
{
	INDEX index_m = END();
	INDEX index_aux = END();
	double current_min_m = numeric_limits<double>::infinity();
	double current_min_aux = numeric_limits<double>::infinity();

	// Step 1: finding the first SV with the smallest alpha (line 2 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)
	for (INDEX dv_index = 0; dv_index < dual_variables.size(); dv_index++)
	{
		double dv_current_value = abs(dual_variables[dv_index]);
		if (dv_current_value == 0) continue;
		if (dv_current_value < current_min_m) {
			current_min_aux = current_min_m;
			index_aux = index_m;
			current_min_m = dv_current_value;
			index_m = dv_index;
		} else if (dv_current_value < current_min_aux) {
			current_min_aux = dv_current_value;
			index_aux = dv_index;
		}
	}
	//--------------------------------------------------------------------------//
	double alpha_m = dual_variables[index_m];
	vector<SE> const& x_m = dataset.data[index_m];
	vector<SE> const& x_aux = dataset.data[index_aux];
	char label_m = dataset.labels[index_m];
	char label_aux = dataset.labels[index_aux];
	double alpha_aux = dual_variables[index_aux];

	// Step 2: finding the merge partner based on the WD method
	double min_weight_degradation = numeric_limits<double>::infinity();
	INDEX index_n = END();
	double m, alpha_candidate;

	for (INDEX i = 0; i < dual_variables.size(); i++)
	{
	   if ((i == index_m) || (label_m != dataset.labels[i])) continue; //same label
		//if(i == index_m) // different label
		alpha_candidate = dual_variables[i];
		/*line 5 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
		m = alpha_m / (alpha_m + alpha_candidate);

		vector<SE> const& x_candidate = dataset.data[i];
		double kernel12 = kernel.evaluate(x_m, x_candidate);
		/*iterative method golden section search or bilinear interpolation to compute h(slice D in figure 3 and line 7 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
#ifndef LOOKUP_WD
#ifdef GSS_HIGH_PRECISION

		double optimal_h = goldenSectionSearch(kernel12, 0, 1, m, 1e-10); //precise GSS
#endif
#ifdef GSS
		double optimal_h;
		if(label_m == dataset.labels[i])
			 optimal_h = goldenSectionSearch(kernel12, m, 0, 1, 0.01); //standard GSS
		else if (alpha_m > alpha_candidate)
			 optimal_h = goldenSectionSearch(kernel12, m, 1, 6, 0.01); //standard GSS
		else
			 optimal_h = goldenSectionSearch(kernel12, m,-5, 0, 0.01); //standard GSS
#endif
#ifdef LOOKUP_H
		double optimal_h = lookUpTable(get<0>(wd_parameters), get<1>(wd_parameters), get<2>(wd_parameters), kernel12, m); //Lookup-h
#endif
		/*computation of z-coefficient(line 9 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
		double kernel1z = pow(kernel12, (1.0 - optimal_h) * (1.0 - optimal_h));
		double kernel2z = pow(kernel12, optimal_h * optimal_h);
		double z_coefficient = alpha_m * kernel1z + alpha_candidate * kernel2z;

		/*computation of WD(slice F in figure 3 and line 10 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
		double weight_degradation = squaredWeightDegradation(kernel12, kernel1z, kernel2z, alpha_m, alpha_candidate, z_coefficient);
#else
		//--------------------------------------------------------------------------//
		/*bilinear interpolation to compute WD(slice D in figure 3 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/

		double lookup_wd = lookUpTable(get<0>(wd_parameters), get<1>(wd_parameters), get<2>(wd_parameters), kernel12,  m);

		lookup_wd *= pow((alpha_m + dual_variables[i]), 2);
	   
		double weight_degradation = lookup_wd;
		//--------------------------------------------------------------------------//
#endif
		//Evaluate the minimmum weight degredation(line 11 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)
		if (weight_degradation < min_weight_degradation)
		{
			/*(line 12 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
			min_weight_degradation = weight_degradation;
			/*(line 13 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
			index_n = i;
		}
	}
 
	if (index_n == END()) {
		// No fitting point with the same label was found, choose the second smallest SV (has different label and will find match guaranteed for B >= 3)
		index_m = index_aux;
		label_m = label_aux;
		alpha_m = alpha_aux;
		vector<SE> const& x_m = x_aux;
		for (INDEX i = 0; i < dual_variables.size(); i++)
		{
			if ((i == index_m) || (label_m != dataset.labels[i])) continue; //same label
			//if (i == index_m)  continue; //different label
			alpha_candidate = dual_variables[i];
			m = alpha_m / (alpha_m + alpha_candidate);
			vector<SE> const& x_candidate = dataset.data[i];
			double kernel12 = kernel.evaluate(x_m, x_candidate);
#ifndef LOOKUP_WD
#ifdef GSS_HIGH_PRECISION
			double optimal_h = goldenSectionSearch(kernel12, m, 0.0, 1.0, 1e-10); //precise GSS
#endif
#ifdef GSS
			double optimal_h = goldenSectionSearch(kernel12, m, 0.0, 1.0, 0.01); //standard GSS
#endif
#ifdef LOOKUP_H
			double optimal_h = lookUpTable(get<0>(wd_parameters), get<1>(wd_parameters), get<2>(wd_parameters), kernel12, m); //Lookup-h
#endif
			/*computation of z-coefficient(line 9 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
			double kernel1z = pow(kernel12, (1.0 - optimal_h) * (1.0 - optimal_h));
			double kernel2z = pow(kernel12, optimal_h * optimal_h);
			double z_coefficient = alpha_m * kernel1z + alpha_candidate * kernel2z;

			/*computation of WD(slice F in figure 3 and line 10 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
			double weight_degradation = squaredWeightDegradation(kernel12, kernel1z, kernel2z, alpha_m, alpha_candidate, z_coefficient);
#else
			//--------------------------------------------------------------------------//
			/*bilinear interpolation to compute WD(slice D in figure 3 and line 7 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/

			double lookup_wd = lookUpTable(get<0>(wd_parameters), get<1>(wd_parameters), get<2>(wd_parameters), kernel12, m);

			lookup_wd *= pow((alpha_m + dual_variables[i]), 2);

			double weight_degradation = lookup_wd;
			//--------------------------------------------------------------------------//
#endif
			//Evaluate the minimmum weight degredation(line 11 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)
			if (weight_degradation < min_weight_degradation)
			{
				/*(line 12 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
				min_weight_degradation = weight_degradation;
				/*(line 13 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
				index_n = i;
			}
		}
	}
	return make_tuple(index_m, index_n);
}


tuple<INDEX, INDEX> mergeHeuristicRandom(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C)
{
	// Select 2 SVs (m,n) for Merging
	INDEX index_m;
	INDEX index_n = END();

	// Choose two SVs randomly
	size_t number_of_variables = dual_variables.size();
	index_m = rand() % number_of_variables;
	index_n = rand() % number_of_variables;

	char label_m = dataset.labels[index_m];
	char label_n = dataset.labels[index_n];

	if (!(label_m == label_n)) {
		INDEX index_aux = rand() % number_of_variables;
		char label_aux = dataset.labels[index_aux];
		if (label_m == label_aux) index_n = index_aux;
		else index_m = index_aux;
	}

	return make_tuple(index_m, index_n);
}

tuple<INDEX, INDEX> mergeHeuristicMinAlpha(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C)
{
	// Select 2 SVs (m,n) for Merging
	INDEX index_m = END();
	INDEX index_aux = END();
	double current_min_m = numeric_limits<double>::infinity();
	double current_min_aux = numeric_limits<double>::infinity();

	// Step 1: finding the first SV with the smallest alpha (line 2 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)
	for (INDEX dv_index = 0; dv_index < dual_variables.size(); dv_index++)
	{
		double dv_current_value = abs(dual_variables[dv_index]);
		if (dv_current_value == 0) continue;
		if (dv_current_value < current_min_m) {
			current_min_aux = current_min_m;
			index_aux = index_m;
			current_min_m = dv_current_value;
			index_m = dv_index;
		} else if (dv_current_value < current_min_aux) {
			current_min_aux = dv_current_value;
			index_aux = dv_index;
		}
	}
	//--------------------------------------------------------------------------//
	double alpha_m = dual_variables[index_m];
	vector<SE> const& x_m = dataset.data[index_m];

	//return make_tuple(index_m, index_aux);
	return make_tuple(index_m, index_m);
}

tuple<INDEX, INDEX> mergeHeuristicMintwoAlphas(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C)
{
	// Select 2 SVs (m,n) for Merging
	INDEX index_m = END();
	INDEX index_aux = END();
	double current_min_m = numeric_limits<double>::infinity();
	double current_min_aux = numeric_limits<double>::infinity();

	// Step 1: finding the first SV with the smallest alpha (line 2 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)
	for (INDEX dv_index = 0; dv_index < dual_variables.size(); dv_index++)
	{
		double dv_current_value = abs(dual_variables[dv_index]);
		if (dv_current_value == 0) continue;
		if (dv_current_value < current_min_m) {
			current_min_aux = current_min_m;
			index_aux = index_m;
			current_min_m = dv_current_value;
			index_m = dv_index;
		} else if (dv_current_value < current_min_aux) {
			current_min_aux = dv_current_value;
			index_aux = dv_index;
		}
	}
	//--------------------------------------------------------------------------//
	unsigned int dv_size = dual_variables.size();

	vector<dv_struct> dual_variables_abs_struct;
	vector<dv_struct> dual_variables_ori_struct;
	dv_struct dv_current_abs_struct;//[dv_size];
	dv_struct dv_current_ori_struct;

	vector<double> current_wd_vector;
	vector<INDEX> current_indexWD_vector;
	double wdValue = INFINITY;


	/*finding SV with smallest alpha, next SV with smallest alpha, next SV with smallest alpha and different sign*/

	for (INDEX dv_index = 0; dv_index < dual_variables.size(); dv_index++)
	{
		double dv_current_value = abs(dual_variables[dv_index]);
		dv_current_abs_struct.dv_value = dv_current_value;
		dv_current_abs_struct.dv_index = dv_index;
		dual_variables_abs_struct.push_back(dv_current_abs_struct);
	}

	sort(dual_variables_abs_struct.begin(), dual_variables_abs_struct.end(), by_value());
	index_m = dual_variables_abs_struct[0].dv_index;
	INDEX index_oppositeto_m;
	INDEX index_n = END();


	for (INDEX dv_ind = 1; dv_ind < dual_variables.size(); dv_ind++)
	{
		INDEX index_candidate = dual_variables_abs_struct[dv_ind].dv_index;
		if(dataset.labels[index_m] == dataset.labels[index_candidate])
		{
			index_n = index_candidate;
			break;
		}
	}

	if(index_n == END())
	{
		for (INDEX dv_ind = 1; dv_ind < dual_variables.size(); dv_ind++)
		{
			INDEX index_candidate = dual_variables_abs_struct[dv_ind].dv_index;
			if(dataset.labels[index_m] != dataset.labels[index_candidate])
			{
				index_oppositeto_m = index_candidate;
				break;
			}
		}

		for (INDEX dv_ind = 1; dv_ind < dual_variables.size(); dv_ind++)
		{
			INDEX index_candidate = dual_variables_abs_struct[dv_ind].dv_index;
			if ((index_oppositeto_m!= index_candidate) && (dataset.labels[index_oppositeto_m] == dataset.labels[index_candidate]))
				{
					index_m = index_oppositeto_m;
					index_n = index_candidate;
					break;
				}
		}
	}




	//----------------------------------------------------------------------//
	return make_tuple(index_m, index_n);
}



tuple<INDEX, INDEX> mergeHeuristicRandomWD(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C)
{
	// Select 2 SVs (m,n) for Merging
	INDEX index_m;
	INDEX index_n;
	INDEX index_aux;
	//----------------------------------------------//
	size_t number_of_variables = dual_variables.size();
	index_m = rand() % number_of_variables;
	index_aux = rand() % number_of_variables;
	//----------------------------------------------//
	double alpha_m = dual_variables[index_m];
	vector<SE> const& x_m = dataset.data[index_m];
	vector<SE> const& x_aux = dataset.data[index_aux];
	char label_m = dataset.labels[index_m];
	char label_aux = dataset.labels[index_aux];
	double alpha_aux = dual_variables[index_aux];

	// Step 2: finding the merge partner based on the WD method
	double min_weight_degradation = numeric_limits<double>::infinity();
	double m, alpha_candidate;

	for (INDEX i = 0; i < dual_variables.size(); i++)
	{
		if ((i == index_m) || (label_m != dataset.labels[i])) continue; //same label
		//if(i == index_m) // different label
		alpha_candidate = dual_variables[i];
		/*line 5 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
		m = alpha_m / (alpha_m + alpha_candidate);

		vector<SE> const& x_candidate = dataset.data[i];
		double kernel12 = kernel.evaluate(x_m, x_candidate);
		/*iterative method golden section search or bilinear interpolation to compute h(slice D in figure 3 and line 7 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
#ifndef LOOKUP_WD
#ifdef GSS_HIGH_PRECISION

		double optimal_h = goldenSectionSearch(kernel12, 0, 1, m, 1e-10); //precise GSS
#endif
#ifdef GSS
		double optimal_h;
		if(label_m == dataset.labels[i])
			optimal_h = goldenSectionSearch(kernel12, m, 0, 1, 0.01); //standard GSS
		else if (alpha_m > alpha_candidate)
			optimal_h = goldenSectionSearch(kernel12, m, 1, 6, 0.01); //standard GSS
		else
			optimal_h = goldenSectionSearch(kernel12, m,-5, 0, 0.01); //standard GSS
#endif
#ifdef LOOKUP_H
		double optimal_h = lookUpTable(get<0>(wd_parameters), get<1>(wd_parameters), get<2>(wd_parameters), kernel12, m); //Lookup-h
#endif
		/*computation of z-coefficient(line 9 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
		double kernel1z = pow(kernel12, (1.0 - optimal_h) * (1.0 - optimal_h));
		double kernel2z = pow(kernel12, optimal_h * optimal_h);
		double z_coefficient = alpha_m * kernel1z + alpha_candidate * kernel2z;

		/*computation of WD(slice F in figure 3 and line 10 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
		double weight_degradation = squaredWeightDegradation(kernel12, kernel1z, kernel2z, alpha_m, alpha_candidate, z_coefficient);
#else
		//--------------------------------------------------------------------------//
		/*bilinear interpolation to compute WD(slice D in figure 3 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/

		double lookup_wd = lookUpTable(get<0>(wd_parameters), get<1>(wd_parameters), get<2>(wd_parameters), kernel12,  m);

		lookup_wd *= pow((alpha_m + dual_variables[i]), 2);

		double weight_degradation = lookup_wd;
		//--------------------------------------------------------------------------//
#endif
		//Evaluate the minimmum weight degredation(line 11 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)
		if (weight_degradation < min_weight_degradation)
		{
			/*(line 12 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
			min_weight_degradation = weight_degradation;
			/*(line 13 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
			index_n = i;
		}
	}

	if (index_n == END()) {
		// No fitting point with the same label was found, choose the second smallest SV (has different label and will find match guaranteed for B >= 3)
		index_m = index_aux;
		label_m = label_aux;
		alpha_m = alpha_aux;
		vector<SE> const& x_m = x_aux;
		for (INDEX i = 0; i < dual_variables.size(); i++)
		{
			if ((i == index_m) || (label_m != dataset.labels[i])) continue; //same label
			//if (i == index_m)  continue; //different label
			alpha_candidate = dual_variables[i];
			m = alpha_m / (alpha_m + alpha_candidate);
			vector<SE> const& x_candidate = dataset.data[i];
			double kernel12 = kernel.evaluate(x_m, x_candidate);
#ifndef LOOKUP_WD
#ifdef GSS_HIGH_PRECISION
			double optimal_h = goldenSectionSearch(kernel12, m, 0.0, 1.0, 1e-10); //precise GSS
#endif
#ifdef GSS
			double optimal_h = goldenSectionSearch(kernel12, m, 0.0, 1.0, 0.01); //standard GSS
#endif
#ifdef LOOKUP_H
			double optimal_h = lookUpTable(get<0>(wd_parameters), get<1>(wd_parameters), get<2>(wd_parameters), kernel12, m); //Lookup-h
#endif
			/*computation of z-coefficient(line 9 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
			double kernel1z = pow(kernel12, (1.0 - optimal_h) * (1.0 - optimal_h));
			double kernel2z = pow(kernel12, optimal_h * optimal_h);
			double z_coefficient = alpha_m * kernel1z + alpha_candidate * kernel2z;

			/*computation of WD(slice F in figure 3 and line 10 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
			double weight_degradation = squaredWeightDegradation(kernel12, kernel1z, kernel2z, alpha_m, alpha_candidate, z_coefficient);
#else
			//--------------------------------------------------------------------------//
			/*bilinear interpolation to compute WD(slice D in figure 3 and line 7 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/

			double lookup_wd = lookUpTable(get<0>(wd_parameters), get<1>(wd_parameters), get<2>(wd_parameters), kernel12, m);

			lookup_wd *= pow((alpha_m + dual_variables[i]), 2);

			double weight_degradation = lookup_wd;
			//--------------------------------------------------------------------------//
#endif
			//Evaluate the minimmum weight degredation(line 11 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)
			if (weight_degradation < min_weight_degradation)
			{
				/*(line 12 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
				min_weight_degradation = weight_degradation;
				/*(line 13 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
				index_n = i;
			}
		}
	}
	vector<double> q_matrix;

	return make_tuple(index_m, index_n);
}

tuple<INDEX, INDEX> mergeHeuristic59plusWD(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C)
{
	INDEX index_m = END();
	INDEX index_aux = END();
	double current_min_m = numeric_limits<double>::infinity();
	double current_min_aux = numeric_limits<double>::infinity();

	// Step 1: finding the first SV with the smallest alpha (line 2 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)
	for (INDEX dv_index = 0; dv_index < dual_variables.size(); dv_index++)
	{
		double dv_current_value = abs(dual_variables[dv_index]);
		if (dv_current_value == 0) continue;
		if (dv_current_value < current_min_m) {
			current_min_aux = current_min_m;
			index_aux = index_m;
			current_min_m = dv_current_value;
			index_m = dv_index;
		} else if (dv_current_value < current_min_aux) {
			current_min_aux = dv_current_value;
			index_aux = dv_index;
		}
	}
	//--------------------------------------------------------------------------//
	double alpha_m = dual_variables[index_m];
	vector<SE> const& x_m = dataset.data[index_m];
	vector<SE> const& x_aux = dataset.data[index_aux];
	char label_m = dataset.labels[index_m];
	char label_aux = dataset.labels[index_aux];
	double alpha_aux = dual_variables[index_aux];

	// Step 2: finding the merge partner based on the WD method
	double min_weight_degradation = numeric_limits<double>::infinity();
	INDEX index_n = END();
	double m, alpha_candidate;

	vector<INDEX> sequence = randomSequence(dual_variables.size());
	size_t sequence_size = 59; //sequence.size();


	for (INDEX i = 0; i < 59; i++)
	{
		 INDEX ws = sequence[i];
		if ((ws == index_m) || (label_m != dataset.labels[ws])) continue; //same label
		//if(i == index_m) // different label
		alpha_candidate = dual_variables[ws];
		/*line 5 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
		m = alpha_m / (alpha_m + alpha_candidate);

		vector<SE> const& x_candidate = dataset.data[ws];
		double kernel12 = kernel.evaluate(x_m, x_candidate);
		/*iterative method golden section search or bilinear interpolation to compute h(slice D in figure 3 and line 7 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
#ifndef LOOKUP_WD
#ifdef GSS_HIGH_PRECISION

		double optimal_h = goldenSectionSearch(kernel12, 0, 1, m, 1e-10); //precise GSS
#endif
#ifdef GSS
		double optimal_h;
		if(label_m == dataset.labels[ws])
			optimal_h = goldenSectionSearch(kernel12, m, 0, 1, 0.01); //standard GSS
		else if (alpha_m > alpha_candidate)
			optimal_h = goldenSectionSearch(kernel12, m, 1, 6, 0.01); //standard GSS
		else
			optimal_h = goldenSectionSearch(kernel12, m,-5, 0, 0.01); //standard GSS
#endif
#ifdef LOOKUP_H
		double optimal_h = lookUpTable(get<0>(wd_parameters), get<1>(wd_parameters), get<2>(wd_parameters), kernel12, m); //Lookup-h
#endif
		/*computation of z-coefficient(line 9 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
		double kernel1z = pow(kernel12, (1.0 - optimal_h) * (1.0 - optimal_h));
		double kernel2z = pow(kernel12, optimal_h * optimal_h);
		double z_coefficient = alpha_m * kernel1z + alpha_candidate * kernel2z;

		/*computation of WD(slice F in figure 3 and line 10 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
		double weight_degradation = squaredWeightDegradation(kernel12, kernel1z, kernel2z, alpha_m, alpha_candidate, z_coefficient);
#else
		//--------------------------------------------------------------------------//
		/*bilinear interpolation to compute WD(slice D in figure 3 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/

		double lookup_wd = lookUpTable(get<0>(wd_parameters), get<1>(wd_parameters), get<2>(wd_parameters), kernel12,  m);

		lookup_wd *= pow((alpha_m + dual_variables[i]), 2);

		double weight_degradation = lookup_wd;
		//--------------------------------------------------------------------------//
#endif
		//Evaluate the minimmum weight degredation(line 11 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)
		if (weight_degradation < min_weight_degradation)
		{
			/*(line 12 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
			min_weight_degradation = weight_degradation;
			/*(line 13 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
			index_n = ws;
		}
	}

	if (index_n == END()) {
		// No fitting point with the same label was found, choose the second smallest SV (has different label and will find match guaranteed for B >= 3)
		index_m = index_aux;
		label_m = label_aux;
		alpha_m = alpha_aux;
		vector<SE> const& x_m = x_aux;

		vector<INDEX> sequence = randomSequence(dual_variables.size());
		size_t sequence_size = 59; //sequence.size();
		for (INDEX i = 0; i < 59; i++)
		{
			INDEX ws = sequence[i];
			if ((ws == index_m) || (label_m != dataset.labels[ws])) continue; //same label
			//if (i == index_m)  continue; //different label
			alpha_candidate = dual_variables[ws];
			m = alpha_m / (alpha_m + alpha_candidate);
			vector<SE> const& x_candidate = dataset.data[ws];
			double kernel12 = kernel.evaluate(x_m, x_candidate);
#ifndef LOOKUP_WD
#ifdef GSS_HIGH_PRECISION
			double optimal_h = goldenSectionSearch(kernel12, m, 0.0, 1.0, 1e-10); //precise GSS
#endif
#ifdef GSS
			double optimal_h = goldenSectionSearch(kernel12, m, 0.0, 1.0, 0.01); //standard GSS
#endif
#ifdef LOOKUP_H
			double optimal_h = lookUpTable(get<0>(wd_parameters), get<1>(wd_parameters), get<2>(wd_parameters), kernel12, m); //Lookup-h
#endif
			/*computation of z-coefficient(line 9 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
			double kernel1z = pow(kernel12, (1.0 - optimal_h) * (1.0 - optimal_h));
			double kernel2z = pow(kernel12, optimal_h * optimal_h);
			double z_coefficient = alpha_m * kernel1z + alpha_candidate * kernel2z;

			/*computation of WD(slice F in figure 3 and line 10 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
			double weight_degradation = squaredWeightDegradation(kernel12, kernel1z, kernel2z, alpha_m, alpha_candidate, z_coefficient);
#else
			//--------------------------------------------------------------------------//
			/*bilinear interpolation to compute WD(slice D in figure 3 and line 7 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/

			double lookup_wd = lookUpTable(get<0>(wd_parameters), get<1>(wd_parameters), get<2>(wd_parameters), kernel12, m);

			lookup_wd *= pow((alpha_m + dual_variables[i]), 2);

			double weight_degradation = lookup_wd;
			//--------------------------------------------------------------------------//
#endif
			//Evaluate the minimmum weight degredation(line 11 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)
			if (weight_degradation < min_weight_degradation)
			{
				/*(line 12 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
				min_weight_degradation = weight_degradation;
				/*(line 13 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
				index_n = ws;
			}
		}
	}
	vector<double> q_matrix;
	return make_tuple(index_m, index_n);
}

tuple<INDEX, INDEX> mergeHeuristicKernel(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C)
{
	// Select 2 SVs (m,n) for Merging
	INDEX index_m = END();
	INDEX index_aux = END();
	double current_min_m = numeric_limits<double>::infinity();
	double current_min_aux = numeric_limits<double>::infinity();

	// Search for smallest absolute alpha to merge
	for (INDEX dv_index = 0; dv_index < dual_variables.size(); dv_index++) {
		double dv_current_value = abs(dual_variables[dv_index]);
		if (dv_current_value == 0) continue;
		if (dv_current_value < current_min_m) {
			current_min_aux = current_min_m;
			index_aux = index_m;
			current_min_m = dv_current_value;
			index_m = dv_index;
		} else if (dv_current_value < current_min_aux) {
			current_min_aux = dv_current_value;
			index_aux = dv_index;
		}
	}

	vector<SE> x_m = dataset.data[index_m];
	vector<SE> x_aux = dataset.data[index_aux];
	char label_m = dataset.labels[index_m];
	char label_aux = dataset.labels[index_aux];


	double max_kernel = -numeric_limits<double>::infinity();
	double k_tmp;
	INDEX index_n = END();
	vector<SE> z;

	for (INDEX i = 0; i < dual_variables.size(); i++) {
		if ((i == index_m) || (label_m != dataset.labels[i])) continue;
		vector<SE> x_candidate = dataset.data[i];
		k_tmp = kernel.evaluate(x_m, x_candidate);
		if (k_tmp > max_kernel) {
			max_kernel = k_tmp;
			index_n = i;
		}
	}
	if (index_n == END()) {
		// No fitting point with the same label was found, choose the second smallest SV (has different label and will find match guaranteed for B >= 3)
		index_m = index_aux;
		label_m = label_aux;
		x_m = x_aux;
		for (INDEX i = 0; i < dual_variables.size(); i++) {
			if ((i == index_m) || (label_m != dataset.labels[i])) continue;
			vector<SE> x_candidate = dataset.data[i];
			k_tmp = kernel.evaluate(x_m, x_candidate);
			if (k_tmp > max_kernel) {
				max_kernel = k_tmp;
				index_n = i;
			}
		}
	}

	 return make_tuple(index_m, index_n);
}



int mergeAndDeleteSV(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters,double C, Heuristic heuristic) {
	//activate the budget maintenance method and get the indices for merging
	INDEX index_m, index_n;
	tie(index_m, index_n) = heuristic(dual_variables, dataset, kernel, wd_parameters, C);

	char label			  = dataset.labels[index_m];
	double alpha_m		  = dual_variables[index_m];
	double alpha_n		  = dual_variables[index_n];
	double m				= alpha_m / (alpha_m + alpha_n);
	vector<SE> const& x_m		  = dataset.data[index_m];
	vector<SE> const& x_n		  = dataset.data[index_n];

	//construct z from the merging partners
	double kernel12 = kernel.evaluate(x_m, x_n);
	double optimal_h		= goldenSectionSearch(kernel12, m, 0.0, 1.0, 0.001);
	vector<SE> z			= scaleAddSparseVectors_new(x_m, x_n, optimal_h, 1-optimal_h);
	double k_mz			 = pow(kernel12, (1.0 - optimal_h) * (1.0 - optimal_h));
	double k_nz			 = pow(kernel12, optimal_h * optimal_h);
	double z_coefficient	= alpha_m*k_mz + alpha_n*k_nz;

	if (label != dataset.labels[index_n]) {
		cout << 1;
	}

	//delete old SVs
	dual_variables[index_m] = dual_variables.back();
	dual_variables[index_n] = dual_variables[dual_variables.size() - 2];
	dual_variables.pop_back();
	dual_variables.pop_back();

	dataset.data[index_m] = dataset.data.back();
	dataset.data[index_n] = dataset.data[dataset.data.size() - 2];
	dataset.data.pop_back();
	dataset.data.pop_back();

	dataset.labels[index_m] = dataset.labels.back();
	dataset.labels[index_n] = dataset.labels[dataset.labels.size() - 2];
	dataset.labels.pop_back();
	dataset.labels.pop_back();

	if (z_coefficient != 0) {
		//Add the created SV and its coefficient
		dataset.data.push_back(z);
		dual_variables.push_back(z_coefficient);
		// Add corresponding label
		dataset.labels.push_back(label);
	}
	return (z_coefficient != 0)? 1:0;
}



tuple<INDEX, INDEX, vector<double>> mergeHeuristicWDVector(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C, vector<double> p)
{
	INDEX index_m = END();
	INDEX index_aux = END();
	double current_min_m = numeric_limits<double>::infinity();
	double current_min_aux = numeric_limits<double>::infinity();

	// Choose two SVs randomly
	size_t number_of_variables = dual_variables.size();
	index_m = rand() % (number_of_variables-1);

	//--------------------------------------------------------------------------//

	char label_m = dataset.labels[index_m];
	INDEX index_n = END();
	double p_sum = 0.0;
	for (INDEX j = 0; j < p.size()-1; j++) {
		p_sum += p[j];
	}

	//vector<double> q_vector;
	vector<double> q_matrix;

	vector<double>q;
	double q_sum = 0.0;
	double gradient = 0.0;
	double s_studentDist_sum = 0.0;
	for (INDEX j = 0; j < dataset.data.size(); j++) {
		if(j == index_m ) continue;
		double q_i = kernel.evaluate(dataset.data[index_m], dataset.data[j]);
	   
		gradient += dual_variables[j]* q_i*dataset.labels[j];
		q.push_back(q_i);

		s_studentDist_sum += q_i;

	}
   
	gradient *= dataset.labels[index_m];
	double gr = 1 - gradient;

	for (INDEX j = 0; j < q.size(); j++) {
		q[j] = q[j]/s_studentDist_sum;
		 q_sum += q[j];
	}

	q.push_back(gr);
	double KL_MIN = INFINITY;
	double KL_MAX = -INFINITY;

	// cout << endl << "***********  ************************************  ***********" << endl;
	double KL_div_sum = 0.0;
	for (INDEX j = 0; j < q.size()-1; j++) {
		if(j == index_m || (label_m != dataset.labels[j])) continue;
		double KL_divergence = p[j] * log(p[j]/q[j]);
		KL_div_sum += -KL_divergence;

		if (KL_divergence < KL_MIN){
			//KL_MAX = KL_divergence; index_n = j;
			KL_MIN = KL_divergence; index_n = j;
			//cout << "index = " << index_n << ", KL_min: " << KL_MIN << endl;
		}
	}
	q.push_back(KL_div_sum);
	size_t inx_grad= q.size()-1;

	return make_tuple(index_m, index_n, q);
}

