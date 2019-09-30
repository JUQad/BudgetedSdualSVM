
#include "loadData.h"
#include "kernel.h"
#include "svm.h"
#include "budgetMaintenance.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <fstream>

#include <limits>
#include <iomanip>
#include <iostream>

using namespace std;
template<class T>
typename enable_if<!numeric_limits<T>::is_integer, bool>::type
almost_equal(T x, T y, int ulp)
{
	// the machine epsilon has to be scaled to the magnitude of the values used
	// and multiplied by the desired precision in ULPs (units in the last place)
	return abs(x-y) <= numeric_limits<T>::epsilon() * abs(x+y) * ulp
	// unless the result is subnormal
	|| abs(x-y) < numeric_limits<T>::min();
}


#define MERGE
//#define FREEZE
//#define PROJECTION
//////////////////
//#define adult
#define codrna
//#define covtype
//#define susy
//////////////////
#define CHANGE_RATE 0.3
#define PREF_MIN 0.05
#define PREF_MAX 20.0
#define INF HUGE_VAL
//////////////////


// see budgetMaintenance.cpp
extern vector<INDEX> randomSequence(size_t number_of_points);

void downgrade(unsigned long oto, vector<double>& pseudo_variables)
{
	for (int i= 0; i < pseudo_variables.size() ; i++ )
		pseudo_variables[i] *= (1.0 - 1.0 / (long double) oto);
};

double computeMargin(vector<SE>& point, char label, vector<double>& pseudo_variables, sparseData& pseudo_data, Kernel const& kernel) {
	double pseudo_gradient = 0;

	size_t number_of_variables = pseudo_variables.size();
	for (INDEX j = 0; j < number_of_variables; j++) {

		pseudo_gradient += pseudo_variables[j]*kernel.evaluate(point, pseudo_data.data[j])*pseudo_data.labels[j];
	}
	pseudo_gradient *= label;
	pseudo_gradient  = 1 - pseudo_gradient;
   
	return pseudo_gradient;
}

tuple< int, double*, int*, vector<double>> scheduling( size_t l, int slen, double prefsum, double* acc , double* pref, int* indexX)
{
	// define schedule
	slen = 0;
	double q = l / prefsum;
	vector<double> ws_vector;
	for (int i=0; i<l; i++)
	{
		double a = acc[i] + q * pref[i];
		int n = (int)floor(a);
		for (int j=0; j<n; j++)
		{
			indexX[slen] = i;
			slen++;
		}
		acc[i] = a - n;
	}

	for (int s=0; s<slen;s++ )
	{
		ws_vector.push_back(indexX[s]);
	}
	for (unsigned int i=0; i<ws_vector.size(); i++)
	{
		swap(ws_vector[i], ws_vector[rand() % ws_vector.size()]);
	}

	tuple< int,  double*, int*, vector<double>> results;
	get<0>(results) = slen;
	get<1>(results) = acc;
	get<2>(results) = indexX;
	get<3>(results) = ws_vector;
	return results;

}

double computeGradient(vector<SE>& point, char label, vector<double>& pseudo_variables, sparseData& pseudo_data, Kernel const& kernel) {
	double pseudo_gradient = 0;
	double max_partpseudo = -INFINITY;
	size_t number_of_variables = pseudo_variables.size();
	for (INDEX j = 0; j < number_of_variables; j++) {
	   
		pseudo_gradient += pseudo_variables[j]*kernel.evaluate(point, pseudo_data.data[j])*pseudo_data.labels[j];
	   
	}
	pseudo_gradient *= label;
	pseudo_gradient  = 1 - pseudo_gradient;

	return pseudo_gradient;
}

vector<double> computeGradientPseudoMaxMin( double& alpha_point, vector<SE>& point, char label, vector<double>& pseudo_variables, sparseData& pseudo_data, Kernel const& kernel) {
	double pseudo_gradient = 0.0;
	double part_m = 0.0;

	double max_partpseudo = -INFINITY;
	double min_partpseudo = INFINITY;
	double max_partm = -INFINITY;
	double min_partm = INFINITY;
	double pseudo_m = 0.0;

	INDEX jMax, jMin;
	size_t number_of_variables = pseudo_variables.size();
	for (INDEX j = 0; j < number_of_variables; j++) {
		double part_pseudo = pseudo_variables[j]*kernel.evaluate(point, pseudo_data.data[j])*pseudo_data.labels[j];
		//(alpha_point ==0.0)? part_m= -INFINITY:part_m = pseudo_variables[j]/alpha_point;
		pseudo_gradient += part_pseudo; //pseudo_variables[j]*kernel.evaluate(point, pseudo_data.data[j])*pseudo_data.labels[j];

		if(part_pseudo > max_partpseudo)
		{
			max_partpseudo = part_pseudo;
		}

		if(part_pseudo < min_partpseudo)
		{
			min_partpseudo = part_pseudo;
		}
	}
	pseudo_gradient *= label;
	pseudo_gradient  = 1 - pseudo_gradient;
	vector<double> g;
	g.push_back(pseudo_gradient);
	g.push_back(max_partpseudo);
	g.push_back(min_partpseudo);
	g.push_back(1+pseudo_gradient);//(max_partm);
	return g;
}





tuple<double,double,double> primalObjectiveFunction (double C, vector<double>& pseudo_variables, vector<double>& dual_variables, sparseData& pseudo_data, sparseData& data, Kernel const& kernel){
	double primaltemp_minW  = 0.0;
	double primaltemp_Hloss  = 0.0;
	for(unsigned int iIter = 0; iIter < pseudo_variables.size(); iIter++)
	{
		for(unsigned int jIter = 0; jIter < pseudo_variables.size(); jIter++)
		{
			double ai = pseudo_variables[iIter];
			double aj = pseudo_variables[jIter];
			double k = kernel.evaluate(pseudo_data.data[iIter], pseudo_data.data[jIter]);
			double yi = pseudo_data.labels[iIter];
			double yj = pseudo_data.labels[jIter];
			primaltemp_minW += ai * aj * k * yi * yj;
		}
	}

	for(unsigned int iIter = 0; iIter < dual_variables.size(); iIter++)
	{
		double margin = 0.0;
		for(unsigned int jIter = 0; jIter < pseudo_variables.size(); jIter++)
		{

			double aj = pseudo_variables[jIter];
			double k = kernel.evaluate(data.data[iIter], pseudo_data.data[jIter]);
			double yj = pseudo_data.labels[jIter];
			margin += aj * k * yj;
			//margin += pseudo_variables[jIter]*kernel.evaluate(data.data[iIter], pseudo_data.data[jIter])*pseudo_data.labels[jIter];
		}
		margin *= data.labels[iIter];
		double violation = max(0.0, 1 - margin);
		primaltemp_Hloss += violation;
	}

	return make_tuple(0.5*primaltemp_minW + C*(primaltemp_Hloss), 0.5*primaltemp_minW, C*(primaltemp_Hloss));
}



tuple<double,double,double> dualObjectiveFunction ( double C, vector<double>& pseudo_variables, vector<double>& dual_variables, sparseData& pseudo_data, sparseData& data, Kernel const& kernel){
	double dualVartemp = 0.0;
	double dualtemp  = 0.0;
	for(unsigned int iIter = 0; iIter < dual_variables.size(); iIter++)
	{
		dualVartemp += dual_variables[iIter];
	}

	for(unsigned int iIter = 0; iIter < dual_variables.size(); iIter++)
	{
		if (dual_variables[iIter] == 0) continue;
		for(unsigned int jIter = 0; jIter < pseudo_variables.size(); jIter++)
		{
			double ai = dual_variables[iIter];
			double aj = pseudo_variables[jIter];
			double k = kernel.evaluate(data.data[iIter], pseudo_data.data[jIter]);
			double yi = data.labels[iIter];
			double yj = pseudo_data.labels[jIter];
			dualtemp += ai * aj * k * yi * yj;


		}
	}


	// return (dualVartemp - 0.5* dualtemp);
	return make_tuple(dualVartemp - 0.5* dualtemp, dualVartemp, 0.5* dualtemp);
}

//Budgeted primal solver
SVM BSGD(sparseData& dataset, sparseData& testdataset, double C, Kernel const& kernel, LookupTable& wd_parameters, double accuracy, size_t B, size_t max_epochs, Heuristic heuristic)
{
	cout << "Optimisation ... begin \n";
	size_t number_of_training_points = dataset.data.size();
	double lambda = 1.0 / ( (double)number_of_training_points * C);
	cout << "Number of training points: " << number_of_training_points << endl;

	sparseData pseudo;
	vector<double> pseudo_variables;

	vector<INDEX> sequence(0);
	vector<double> dual_variables(number_of_training_points, 0);

	unsigned int numIter = 0;

	string PATH = "BSGDsolver/";
	system("mkdir \"BSGDsolver\"");
	cout << "Max Epochs: " << max_epochs << "  Current:";
	ofstream dualObjfn_primalfile;
	dualObjfn_primalfile.open ( PATH + "dualobjective.txt");
	

	ofstream primalObjfn_primalfile;
	primalObjfn_primalfile.open (PATH + "primalobjective.txt");
	ofstream primalObjfn_primalfile_param;
	
	primalObjfn_primalfile_per.open (PATH + "testaccuracy.txt");
	ofstream primalObjfn_primalfile_traint;
	primalObjfn_primalfile_traint.open (PATH + "trainingtime.txt");

	
	double train_start_t = 0.0;
	double  train_end_t = 0.0;
	

	for (size_t epoch = 0; epoch < max_epochs; epoch++)
	{
		cout <<  epoch+1 << ":";

		sequence.clear();
		for (INDEX i=0; i<number_of_training_points; i++)
		{
			sequence.push_back(i);
		}
		for (unsigned int i=0; i<number_of_training_points; i++)
		{
			swap(sequence[i], sequence[rand() % number_of_training_points]);
		}
		size_t sequence_size = sequence.size();
		double mergeAndDeleteSV_counter = 0.0;
		unsigned int countMerges = 0;
		train_start_t = (double)clock() / CLOCKS_PER_SEC;
		for (INDEX i = 0; i < sequence_size; i++)
		{
			INDEX ws = sequence[i];
			numIter++;
			//+++++++++++++ define & implement modelBsgdMap +++++++++++++//

			if (numIter == 1)
			{
				pseudo.data.push_back(dataset.data[ws]);
				pseudo.labels.push_back(dataset.labels[ws]);
				pseudo_variables.push_back( 1.0);
				dual_variables[ws] = 1.0;
				continue;
			}

			/*calculation the margin(slice A in figure 3 and line 4 in algorithm 1 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
			double marginViolation = computeMargin(dataset.data[ws], dataset.labels[ws], pseudo_variables, pseudo, kernel);

			/*calculation the downgrade step(slice B in figure 3 and line 5 in algorithm 1 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
			downgrade(numIter, pseudo_variables);

			/*line 6 in algorithm 1 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search*/
			if ( (marginViolation) > 0.0)  // check margin violation
			{
				//add a new SV to the pseudo model(line 7 in algorithm 1 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)
				pseudo_variables.push_back( (1.0 / ((double)numIter * lambda)));
				dual_variables[ws] += (1.0 / ((double)numIter * lambda));
				pseudo.data.push_back(dataset.data[ws]);
				pseudo.labels.push_back(dataset.labels[ws]);
			}

			//objfun_start_time = (double)clock() / CLOCKS_PER_SEC;
			while (pseudo.data.size() > B)
			{
				//Check the model size compared to the budget(line 9 in algorithm 1 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)
				double mergeAndDeleteSV_start_time = (double)clock() / CLOCKS_PER_SEC;

				mergeAndDeleteSV(pseudo_variables,  pseudo, kernel, wd_parameters, C, heuristic);

				double mergeAndDeleteSV_end_time = (double)clock() / CLOCKS_PER_SEC;

				mergeAndDeleteSV_counter += mergeAndDeleteSV_end_time - mergeAndDeleteSV_start_time;
				countMerges++;
			}
		} // end sequence

		///////////////////////////////////////////////////////////////////////////////////////// Objective function routine
		train_end_t = (double)clock() / CLOCKS_PER_SEC;
		pObjfn_pf_merging << "epoch :"  << epoch +1  << ":mergingtime:" << mergeAndDeleteSV_counter << ":mergingsteps:" << countMerges << endl;
		countMerges = 0;
		SVM svm(pseudo_variables, pseudo, kernel);
		primalObjfn_primalfile_per << "epoch :"  << epoch +1  << ": per.:"  << svm.evaluateTestset(testdataset)<< ":" << endl;

		primalObjfn_primalfile_traint << "epoch :"  << epoch +1  << ":trainingtime.:"  << train_end_t - train_start_t << ":" << endl;

		
	}
	dualObjfn_primalfile.close();
	primalObjfn_primalfile.close();


	primalObjfn_primalfile_per.close();
	primalObjfn_primalfile_traint.close();
	
	return SVM(pseudo_variables, pseudo, kernel);
}
//Budgeted dual solver
SVM BDCA(sparseData& dataset, sparseData& testdataset, double C, Kernel const& kernel, LookupTable& wd_parameters, double accuracy, size_t B, size_t max_epochs, Heuristic heuristic)
{
	cout << "Optimisation ... begin \n";
	size_t number_of_training_points = dataset.data.size();
	cout << "Number of training points: " << number_of_training_points << endl;
	sparseData pseudo;
	vector<double> pseudo_variables;

	vector<INDEX> sequence(0);
	vector<double> dual_variables(number_of_training_points, 0);
	string budget = to_string(B);
	string datasetname;
#ifdef adult
	datasetname = "a";
	cout << "dataset: adult\n";
#endif
#ifdef codrna
	datasetname = "c";
	 cout << "dataset: codrna\n";
#endif
#ifdef susy
	datasetname = "s";
	cout << "dataset: susy\n";
#endif
#ifdef covtype
	datasetname = "cov";
	cout << "dataset: covtype\n";
#endif

	// Main optimization loop
	string PATH;

#ifdef PROJECTION
	PATH = "projection2/";
	system("mkdir \"projection2\"");
	cout << "Projection mode\n";
#endif
#ifdef MERGE
	PATH = "BSCAsolver/";
	system("mkdir \"BSCAsolver\"");
	cout << "Merging mode\n";
#endif

	cout << "Max Epochs: " << max_epochs << "  Current:";
	ofstream dobjfn_dfile;
	dobjfn_dfile.open (PATH + "dualobjective.txt");
	
	ofstream dobjfn_dfile_per;
	dobjfn_dfile_per.open (PATH + "testaccuracy.txt");
	ofstream dobjfn_dfile_pseudoVariables;
	

	ofstream pobjfn_dfile;
	pobjfn_dfile.open (PATH + "primalobjective.txt");
	

	ofstream dobjfn_dfile_traint;
	dobjfn_dfile_traint.open (PATH + "trainingtime.txt");
	ofstream dobjfn_dfile_merging;
	dobjfn_dfile_merging.open (PATH + "merging.txt");

	double train_start_t = 0.0;
	double mergeAndDeleteSV_counter = 0.0;
	unsigned int countMerges = 0;

	unsigned int dualvarisZero = 0;
	unsigned int dualvarisC = 0;
	unsigned int dualvarbetZeroC = 0;

	INDEX numIter = 1;
	size_t M = 20;
	unsigned int topk = 16;
	size_t n = number_of_training_points;

	for (size_t epoch = 0; epoch < max_epochs; epoch++)
	{
		cout << endl << epoch+1 << ":";
		double gradientMIN = INFINITY;
		double gradientMAX = -INFINITY;

		double gradientpseudoMIN = INFINITY;
		double gradientpseudoMAX = -INFINITY;
		
		sequence.clear();

    
		train_start_t = (double)clock() / CLOCKS_PER_SEC;
		double maxPseudo = 0.0;
		double maxG_pseudovar = -INFINITY;
		double sum_maxG_pseudovar = 0.0;

		unsigned int counter = 0;
		vector<double> extradualvariables;
		sparseData extradataset;
		sequence.clear();
		for (INDEX i=0; i<number_of_training_points; i++)
		{
			sequence.push_back(i);
		}
		for (unsigned int i=0; i<number_of_training_points; i++)
		{
			swap(sequence[i], sequence[rand() % number_of_training_points]);
		}
		size_t sequence_size = sequence.size();
		int lr = 1;


		for (INDEX i = 0; i < sequence_size; i++)
		{
			numIter++;

			INDEX ws = sequence[i];
			double alpha_point = dual_variables[ws];
			vector<double> gradientminmax;
			gradientminmax = computeGradientPseudoMaxMin(alpha_point, dataset.data[ws], dataset.labels[ws], pseudo_variables, pseudo, kernel);

			double gradient = gradientminmax[0];
			double gradientmax = gradientminmax[1];
			double gradientmin = gradientminmax[2];
			double pseudogradient   = gradientminmax[3];

			if (pseudogradient < gradientpseudoMIN) gradientpseudoMIN = pseudogradient-(1+gradientmin);
			if (pseudogradient > gradientpseudoMAX) gradientpseudoMAX = pseudogradient-(1+gradientmax);

			if (gradientmin < gradientMIN) gradientMIN = gradientmin;
			if (gradientmax > gradientMAX) gradientMAX = gradientmax;

			double old_alpha = dual_variables[ws];
			double new_alpha = max(0.0, min(old_alpha + gradient, C));
			dual_variables[ws] = new_alpha;

			///////////////////////////////
			vector<double> p_matrix;
			////////////////////////////////

			if(new_alpha == 0) dualvarisZero+=1;
			if(new_alpha == C) dualvarisC+=1;
			if(new_alpha<C && new_alpha>0) dualvarbetZeroC+=1;
			//////////////////////////////////////////////////////////////////////
			double absG_pseudovar = 0.0;
			int score = 0;
			if (old_alpha == 0.0 && gradient <= 0.0){}
			else if (old_alpha == C && gradient >= 0.0){}
			else
			{
				absG_pseudovar = abs(gradient);
				(absG_pseudovar > maxG_pseudovar)? score=1 : score=2 ;
				if (absG_pseudovar > maxG_pseudovar) maxG_pseudovar = absG_pseudovar;
				sum_maxG_pseudovar += absG_pseudovar;

			}

			if ( (new_alpha != old_alpha) )
			{
				pseudo.data.push_back(dataset.data[ws]);
				pseudo_variables.push_back((new_alpha-old_alpha)); //-old_alpha
				pseudo.labels.push_back(dataset.labels[ws]);

			}

			//////////////////////////////////////////////////////////////////////
 
			if (pseudo.data.size() > B )
			{
				counter++;

				//Check the model size compared to the budget
				double mergeAndDeleteSV_start_time = (double)clock() / CLOCKS_PER_SEC;
				mergeAndDeleteSV(pseudo_variables,  pseudo, kernel, wd_parameters, C, heuristic);
				double mergeAndDeleteSV_end_time = (double)clock() / CLOCKS_PER_SEC;
				mergeAndDeleteSV_counter += mergeAndDeleteSV_end_time - mergeAndDeleteSV_start_time;
				countMerges++;

			}
		} //end of sequence

		double train_end_t = (double)clock() / CLOCKS_PER_SEC;
		

		SVM svm(pseudo_variables, pseudo, kernel);
		dobjfn_dfile_per << "epoch :"  << epoch +1  << ": per.:"  << svm.evaluateTestset(testdataset)<< ":" << endl;
		unsigned int nSV = 0; unsigned int nBSV = 0;
		for (size_t i=0; i<number_of_training_points; i++)
		{
			if(fabs(dual_variables[i])>0)
			{
				++nSV;
				if(dataset.labels[i]>0)
				{
					if(fabs(dual_variables[i])>=C)++nBSV;

				} else
				{
					if(fabs(dual_variables[i])>=0)++nBSV;
				}
			}

		}
		dobjfn_dfile_dualpseudoCounter <<"epoch :"  << epoch +1 << ":nSV: "<<nSV<<":nBSV:"<<nBSV<< endl;
		dobjfn_dfile_traint << "epoch :"  << epoch +1  << ":trainingtime.:"  << train_end_t - train_start_t << ":" << endl;

		/* dualObjFunValue =  \sum dual-variables - 0.5 (minW)^2 */
		double dualObjFunValue = 0.0;
		double dualVariable = 0.0;
		double dual_05_minWsquare = 0.0;
		tie(dualObjFunValue, dualVariable, dual_05_minWsquare) = dualObjectiveFunction ( C, pseudo_variables, dual_variables, pseudo, dataset, kernel );
		dobjfn_dfile <<"epoch:"  << epoch +1 << ":" << dualObjFunValue << ":" << endl;
		
		dualObjFunValue = 0.0;
		dualVariable = 0.0;
		dual_05_minWsquare = 0.0;
		/* primalObjFunValue =  C*HLoss + 0.5 (minW)^2 */
		double primalObjFunValue = 0.0;
		double primal_05_minWsquare = 0.0;
		double primal_C_mul_Hloss = 0.0;

		tie(primalObjFunValue, primal_05_minWsquare, primal_C_mul_Hloss) = primalObjectiveFunction (C, pseudo_variables, dual_variables, pseudo, dataset, kernel );
		pobjfn_dfile <<"epoch:"  <<epoch +1 << ":" << primalObjFunValue << ":" << endl;
		

		primalObjFunValue = 0.0;
		primal_05_minWsquare = 0.0;
		primal_C_mul_Hloss = 0.0;

		dualvarisZero  = 0;
		dualvarisC	 = 0;
		dualvarbetZeroC = 0;
	}
	dobjfn_dfile.close();
	pobjfn_dfile.close();

	dobjfn_dfile_param.close();
	pobjfn_dfile_param.close();

	dobjfn_dfile_per.close();
	dobjfn_dfile_traint.close();
	dobjfn_dfile_merging.close();

	dobjfn_dfile_gradvariants.close();

	dobjfn_dfile_pseudoVariables.close();

	return SVM(pseudo_variables, pseudo, kernel);
}


//Budgeted dual solver
SVM acfBDCA(sparseData& dataset, sparseData& testdataset, double C, Kernel const& kernel, LookupTable& wd_parameters, double accuracy, size_t B, size_t max_epochs, Heuristic heuristic)
{
	cout << "Optimisation ... begin \n";
	size_t number_of_training_points = dataset.data.size();
	cout << "Number of training points: " << number_of_training_points << endl;
	sparseData pseudo;
	vector<double> pseudo_variables;

	vector<INDEX> sequence(0);
	vector<double> dual_variables(number_of_training_points, 0);
	vector<INDEX> ws_sequence(0);

	// Main optimization loop

	double dualObjFunValue = 0.0 , primalObjFunValue = 0.0, dualobjfnt= 0.0;
	double dualVariable = 0.0, dual_05_minWsquare = 0.0;
	double primal_05_minWsquare = 0.0 , primal_C_mul_Hloss = 0.0;

	string PATH = "acf_BSCAsolver/";
	system("mkdir \"acf_BSCAsolver\"");
	cout << "Max Epochs: " << max_epochs << "  Current:";
	ofstream dobjfn_dfile;
	dobjfn_dfile.open (PATH + "dualobjective.txt");
	ofstream dobjfn_dfile_per;
	dobjfn_dfile_per.open (PATH + "testaccuracy.txt");

	
	ofstream pobjfn_dfile;
	pobjfn_dfile.open (PATH + "primalobjective.txt");
	ofstream dobjfn_dfile_traint;
	dobjfn_dfile_traint.open (PATH + "trainingtime.txt");

	ofstream dobjfn_dfile_merging;
	dobjfn_dfile_merging.open (PATH + "merging.txt");

	double train_start_t = 0.0;

	double mergeAndDeleteSV_counter = 0.0;
	unsigned int countMerges = 0;
	double sum_maxG_pseudovar = 0.0;

	double maxG_pseudovar = -INFINITY;
	double gradientMIN = INFINITY;
	double gradientMAX = -INFINITY;

	double gradientpseudoMIN = INFINITY;
	double gradientpseudoMAX = -INFINITY;
	unsigned int dualvarisZero = 0;
	unsigned int dualvarisC = 0;
	unsigned int dualvarbetZeroC = 0;
	//////////////////////////////////////////////////////////
	//acf-bsca initialization
	int iter = 1;
	int* index= new int[2*number_of_training_points];
	unsigned long long steps = 0;
	int max_iter= 15000;
	double eps=0.0001;
	int slen ;
	double dualVartemp=0.0;
	double prefsum=0.0;
	// prepare preferences for scheduling
	double* pref = new double[number_of_training_points]();
	const double gain_learning_rate = 1.0 / (number_of_training_points);
	double average_gain = 0.0;
	double stopping = INFINITY;

	//////////////////////////////////////////////////////////
	//acf-bsca prepare data
	for (size_t i=0; i<number_of_training_points; i++) pref[i] = rand()/number_of_training_points ; //rand() % 20;
	for (size_t i=0; i<number_of_training_points; i++) prefsum+=pref[i];
	double nOversum = number_of_training_points / prefsum;
	double* acc = new double[number_of_training_points]() ;
	for (size_t i=0; i<number_of_training_points; i++)
	{
		double a = acc[i] + nOversum * pref[i];
		int n = (int)floor(a);
		acc[i] = a - n;
	}

	//////////////////////////////////////////////////////////
	INDEX numIter = 1;
	int num_iter = 1;
	size_t epoch = 0;
	double storage_old_pseudoGmin = 0.0;
	double storage_new_pseudoGmin = 0.0;
	double storage_old_pseudoGmax = 0.0;
	double storage_new_pseudoGmax = 0.0;
	for (size_t epoch = 0; epoch < max_epochs; epoch++)
	{
		//dataset.shuffle_ds_dualvec(dual_variables);
		vector<double> pseudo_variables_size(number_of_training_points, 0.0);
		int variable = 0;
		double storage_old_pseudoGmin ;
		double storage_new_pseudoGmin ;
		double storage_old_pseudoGmax ;
		double storage_new_pseudoGmax ;
		cout << endl << epoch+1 << ":";
		double gradientMIN = INFINITY;
		double gradientMAX = -INFINITY;

		double KL_max = -INFINITY;
		sequence = randomSequence(number_of_training_points);
		double mergeAndDeleteSV_counter = 0.0;
		double scheduling_time_counter=0.0;
		unsigned int countMerges = 0;

		//start training:
		train_start_t = (double)clock() / CLOCKS_PER_SEC;
		double maxPseudo = 0;

		unsigned int counter = 0;
		vector<double> extradualvariables;
		sparseData extradataset;

		/////////////////////////////////////////////////////////////////////
		//Scheduling start
		double scheduling_start_t = (double)clock() / CLOCKS_PER_SEC;
		slen = 0.0;
		nOversum = number_of_training_points / prefsum;

		for (INDEX i = 0; i < number_of_training_points; i++)
		{
			numIter++;
			double a	= acc[i] + nOversum * pref[i];
			int n	   = (int)floor(a);
			for (int j=0; j<n; j++){index[slen] = i; slen++;}
			acc[i]	  = a - (double)n;
			//cout << acc[i] << endl;
		}
		double scheduling_end_t = (double)clock() / CLOCKS_PER_SEC;
		scheduling_time_counter+=scheduling_end_t-scheduling_start_t;

		/////////////////////////////////////////////////////////////////////
		ws_sequence.clear();
		for (int s=0; s<slen;s++ )
		{
			ws_sequence.push_back(index[s]);
		}
		for (unsigned int i=0; i<ws_sequence.size(); i++)
		{
			swap(ws_sequence[i], ws_sequence[rand() % ws_sequence.size()]);
		}

		cout << "ws size is: " << ws_sequence.size();
		size_t ws_sequence_size = ws_sequence.size();
		steps +=ws_sequence_size;

		for (INDEX i = 0; i < ws_sequence_size; i++)
		{
			INDEX ws = ws_sequence[i];
			double alpha_point = dual_variables[ws];
			vector<double> gradientminmax;
			gradientminmax = computeGradientPseudoMaxMin(alpha_point, dataset.data[ws], dataset.labels[ws], pseudo_variables, pseudo, kernel);

			double gradient = gradientminmax[0];
			double gradientmax = gradientminmax[1];
			double gradientmin = gradientminmax[2];

			if (gradientmin < gradientMIN) gradientMIN = gradientmin;
			if (gradientmax > gradientMAX) gradientMAX = gradientmax;
			double gain = 0.0;
			double change = 0.0;
			double newpref;

			//cout << maxPseudo << endl;
			double old_alpha = dual_variables[ws];
			double new_alpha = max(0.0, min(old_alpha + gradient, C));
			dual_variables[ws] = new_alpha;

			if(new_alpha == 0) dualvarisZero+=1;
			if(new_alpha == C) dualvarisC+=1;
			if(new_alpha<C && new_alpha>0) dualvarbetZeroC+=1;

			//////////////////////////////////////////////////////////////////////
			// dual variable changes
			if (new_alpha != old_alpha)
			{
				pseudo.data.push_back(dataset.data[ws]);
				pseudo_variables.push_back(new_alpha-old_alpha);
				pseudo.labels.push_back(dataset.labels[ws]);
				double delta = new_alpha - old_alpha;
				gain = delta * (gradient - 0.5 * delta);
			 }
			//////////////////////////////////////////////////////////////////////
			//cout << "\n" << numIter;
			if (numIter == 0) average_gain += gain/(double)slen;
			else{
				change = CHANGE_RATE * (gain/average_gain - 1.0);
				newpref = min(PREF_MAX, max(PREF_MIN, pref[ws] * exp(change)));
				prefsum += newpref - pref[ws];
				pref[ws] = newpref;
				average_gain = (1.0 - gain_learning_rate) * average_gain + gain_learning_rate * gain;
			}
			//////////////////////////////////////////////////////////////////////
			while (pseudo.data.size() > B)
			{
				counter++;

				//Check the model size compared to the budget
				double mergeAndDeleteSV_start_time = (double)clock() / CLOCKS_PER_SEC;
				mergeAndDeleteSV(pseudo_variables,  pseudo, kernel, wd_parameters, C, heuristic);
				double mergeAndDeleteSV_end_time = (double)clock() / CLOCKS_PER_SEC;
				mergeAndDeleteSV_counter += mergeAndDeleteSV_end_time - mergeAndDeleteSV_start_time;
				countMerges++;

			}

			numIter++;


		} //end of sequence
		//epoch++;
		num_iter++;
		double train_end_t = (double)clock() / CLOCKS_PER_SEC;
		//dobjfn_dfile_merging << "epoch :"  << epoch +1 << ":KL:" << KL_max;
		dobjfn_dfile_merging << "epoch :"  << epoch +1  <<  ":mergingtime:" << mergeAndDeleteSV_counter << ":mergingsteps:" << countMerges << endl;

		dobjfn_dfile_gradvariants << "epoch :"  << epoch +1  << ":maxG_pseudovar:"  << maxG_pseudovar<< ":avg gradient:" << maxG_pseudovar/sum_maxG_pseudovar << ":minPgradient:" << gradientMIN << ":maxPgradient:" << gradientMAX <<  ":gradientpseudoMIN:" << gradientpseudoMIN << ":gradientpseudoMAX:" << gradientpseudoMAX  << endl;
		countMerges = 0;
		mergeAndDeleteSV_counter = 0.0;

		SVM svm(pseudo_variables, pseudo, kernel);
		dobjfn_dfile_per << "epoch :"  << epoch +1  << ": per.:"  << svm.evaluateTestset(testdataset)<< ":" << endl;

		unsigned int nSV = 0; unsigned int nBSV = 0;
		for (size_t i=0; i<number_of_training_points; i++)
		{
			if(fabs(dual_variables[i])>0)
			{
				++nSV;
				if(dataset.labels[i]>0)
				{
					if(fabs(dual_variables[i])>=C)++nBSV;

				} else
				{
					if(fabs(dual_variables[i])>=0)++nBSV;
				}
			}

		}
		dobjfn_dfile_dualpseudoCounter <<"epoch :"  << epoch +1 << ":nSV: "<<nSV<<":nBSV:"<<nBSV<< endl;

		dobjfn_dfile_traint << "epoch :"  << epoch +1  << ":trainingtime.:"  << train_end_t - train_start_t << ":" << endl;
		
		dualvarisZero  = 0;
		dualvarisC	 = 0;
		dualvarbetZeroC = 0;

		/* dualObjFunValue =  \sum dual-variables - 0.5 (minW)^2 */
		tie(dualObjFunValue, dualVariable, dual_05_minWsquare) = dualObjectiveFunction ( C, pseudo_variables, dual_variables, pseudo, dataset, kernel );
		dobjfn_dfile  << "epoch:"<< epoch << ":" <<  dualObjFunValue << endl;
		/* primalObjFunValue =  C*HLoss + 0.5 (minW)^2 */
		tie(primalObjFunValue, primal_05_minWsquare, primal_C_mul_Hloss) = primalObjectiveFunction (C, pseudo_variables, dual_variables, pseudo, dataset, kernel );
		pobjfn_dfile  << "epoch:"<< epoch << ":" <<  primalObjFunValue << endl;
		double old_diff = 0.0;
		if (num_iter ==2)
		{
			storage_new_pseudoGmax = gradientMAX;
			storage_new_pseudoGmin = gradientMIN;
		}
		else {
			storage_old_pseudoGmax = storage_new_pseudoGmax;
			storage_old_pseudoGmin = storage_new_pseudoGmin;
			storage_new_pseudoGmax = gradientMAX;
			storage_new_pseudoGmin = gradientMIN;
			old_diff = storage_old_pseudoGmax - storage_old_pseudoGmin;
		}
		double new_diff = storage_new_pseudoGmax - storage_new_pseudoGmin;
		//old_diff = storage_old_pseudoGmax - storage_old_pseudoGmin;
		double new_diff_log = log(new_diff);
		double old_diff_log;
		if (num_iter ==2)
		{
			old_diff_log = 0.0;
		}
		else
			old_diff_log = log(old_diff);
		stopping = new_diff_log - old_diff_log;
		ws_sequence.clear() ;
		
	}
	dobjfn_dfile.close();
	pobjfn_dfile.close();

	dobjfn_dfile_param.close();
	pobjfn_dfile_param.close();

	dobjfn_dfile_per.close();
	dobjfn_dfile_traint.close();
	dobjfn_dfile_merging.close();

	
	return SVM(pseudo_variables, pseudo, kernel);
}

