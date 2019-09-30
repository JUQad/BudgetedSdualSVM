
#include "loadData.h"
#include "solver.h"
#include "kernel.h"
#include "svm.h"
#include "budgetMaintenance.h"

#include <iostream>
#include <fstream>
#include <string>
#include <tuple>
#include <cmath>
#include <limits>
#include <random>

using namespace std;


int parseCLI(int argc, char** argv) {
// --test : test dataset path
// --train : training dataset path
// --method : BDGD or BSGD
// --heuristic : LWD
// --epochs : number of epochs
// --output : output file path
// --C
// --gamma


	string train_path = "NOPATH";
	string test_path = "NOPATH";
	string output_path = "NOPATH";
	string lookup_path = "NOPATH";
	double C = -1;
	double gamma;
	bool gamma_set = false;
	size_t method = numeric_limits<size_t>::max();
	size_t heuristic_number = 0;
	size_t epochs = 1;
	size_t budget = 0;
	double accuracy = 0.1;

	vector<string> arguments(argv + 1, argv + argc);

	for (int i = 0; i < argc; i++) {
		if (arguments[i] == "--train") {
			train_path = arguments[i+1];
		} else if (arguments[i] == "--test") {
			test_path = arguments[i+1];
		} else if (arguments[i] == "--output") {
			output_path = arguments[i+1];
		} else if (arguments[i] == "--lookup") {
			lookup_path = arguments[i+1];
		} else if (arguments[i] == "--method") {
			method = stoi(arguments[i+1]);
		} else if (arguments[i] == "--accuracy") {
			accuracy = stod(arguments[i+1]);
		} else if(arguments[i] == "--epochs") {
			epochs = stoi(arguments[i+1]);
		} else if(arguments[i] == "--heuristic") {
			heuristic_number = stoi(arguments[i+1]);
		} else if (arguments[i] == "--C") {
			C = stod(arguments[i+1]);
		} else if (arguments[i] == "--gamma") {
			gamma = stod(arguments[i+1]);
			gamma_set = true;
		} else if (arguments[i] == "--B") {
			budget = stoi(arguments[i+1]);
		}
	}

	if (method == numeric_limits<size_t>::max()) {
		cout << "Provide a method, e.g. --method 0" << endl;
		return 1;
	}
	if (train_path == "NOPATH") {
		cout << "Provide a path to training_data, e.g. --train C:\\data\\a.txt" << endl;
		return 1;
	}
	if (lookup_path == "NOPATH") {
		cout << "Provide a path to get the WD data, e.g. --lookup C:\\data\\a.txt" << endl;
		return 1;
	}

	if (budget == 0) {
		cout << "please note that there is no budget provided" << endl;
		budget = numeric_limits<size_t>::max();
	}

	// load training dataset
	sparseData train_data;
	sparseData test_data;

	cout << "Loading training dataset ..." << endl;
	double loaddata_time = (double)clock() / CLOCKS_PER_SEC;
	loaddata(train_path, train_data, true);
	double loaddata_time_end = (double)clock() / CLOCKS_PER_SEC;
	cout << "loading time = " << loaddata_time_end - loaddata_time << endl;
	cout << "Training dataset is complete." << endl << endl;

	//load the lookup-WD or lookup-h
	LookupTable wd_parameters = load_lookup(lookup_path);


	// Choose solver
	auto solver = BSGD;
	if (method == 0) {
		cout << "method = 0, Solver: BSGD" << endl;
		solver = BSGD;
	}
	else if (method == 1) {
		cout << "method = 1, Solver: BDCA" << endl;
		solver = BDCA;
	}
	else if (method == 2) {
		cout << "method = 2, Solver: acfBDCA" << endl;
		solver = acfBDCA;
	}
	else if (method == 3) {
		cout << "method = 3, Solver: BMVPSMO" << endl;
		solver = BMVPSMO;
		//BMVPSMOSimplified
	}
	else if (method == 5) {
		cout << "method = 5, Solver: BMVPSMOSimplified" << endl;
		solver = BMVPSMOSimplified;
	}
	else if (method == 4) {
		cout << "method = 4, Solver: SBSCA" << endl;
		solver = SBSCA;
	}else {
		cout << "Method: " << method << " undefined" << endl;
		throw runtime_error("Method undefined!");
	}

	//Choose Budget maintenance merging method
	auto heuristic = mergeHeuristicWD;
	if (heuristic_number == 0) {
		cout << "heuristic = 0, Budget maintanenace: mergeLWD" << endl;
		heuristic = mergeHeuristicWD;
	}
	else if (heuristic_number == 1) {
		cout << "heuristic = 1, Budget maintanenace: mergeHeuristicRandom" << endl;
		heuristic = mergeHeuristicRandom;
	}
	else if (heuristic_number == 2) {
		cout << "heuristic = 2, Budget maintanenace: mergeHeuristicKernel" << endl;
		heuristic = mergeHeuristicKernel;
	}
	else if (heuristic_number == 3) {
		cout << "heuristic = 3, Budget maintanenace: mergeHeuristicRandomWD" << endl;
		heuristic = mergeHeuristicRandomWD;
	}
	else if (heuristic_number == 4) {
		cout << "heuristic = 4, Budget maintanenace: mergeHeuristicMinAlpha" << endl;
		heuristic = mergeHeuristicMinAlpha;
	}
	//mergeHeuristicMintwoAlphas
	else if (heuristic_number == 5) {
		cout << "heuristic = 5, Budget maintanenace: mergeHeuristicMintwoAlphas" << endl;
		heuristic = mergeHeuristicMintwoAlphas;
	}
	//mergeHeuristic59plusWD
	else if (heuristic_number == 6) {
		cout << "heuristic = 6, Budget maintanenace: mergeHeuristic59plusWD" << endl;
		heuristic = mergeHeuristic59plusWD;
	}
	
	else
	{
		cout << "*Heuristic " << method << " undefined" << endl;
		throw runtime_error("Method undefined");
	}

	auto HeuristicWithmoreVectors = mergeHeuristicWDVector;

	streambuf* buffer;
	ofstream output_file;
	if (output_path == "NOPATH") {
		cout << "*" << endl;
		buffer = cout.rdbuf();
	} else {
		output_file.open(output_path, ios::out);
		buffer = output_file.rdbuf();
	}
	ostream out(buffer);

	//to the txt file
	out << "budget=" << ((budget == numeric_limits<size_t>::max()) ? 0 : budget) << endl;
	out << "epochs=" << epochs << endl;
	out << "method=" << method << endl;
	out << "heuristic=" << heuristic_number << endl;
	out << "#C=" << C << endl;
	out << "#gamma=" << gamma << endl;

	//loading testing dataset
	size_t max_repeat = 1;
		if (test_path != "NOPATH")
		{
			loaddata(test_path, test_data, true);
			double performance_avg = 0.0, train_time_avg = 0.0;

			for (INDEX rep_i = 0; rep_i < max_repeat; rep_i++)
			{
			   
				train_data.shuffle();
				RbfKernel kernel(gamma);
				//training!
				double train_start_time = (double)clock() / CLOCKS_PER_SEC;
				cout << "first hello! " << endl;
				SVM svm = solver(train_data, test_data, C, kernel, wd_parameters, accuracy, budget, epochs, heuristic);
				
				double test_start_time = (double)clock() / CLOCKS_PER_SEC;
				//testing!
				double performance = svm.evaluateTestset(test_data);
				double test_end_time = (double)clock() / CLOCKS_PER_SEC;

				//cout << svm.size() << endl;
				out << performance << ":" << svm.size() << ":" << test_start_time - train_start_time << ":" << test_end_time - test_start_time << endl;
				//cout << "per.: " << performance << ", train. time:" << (test_start_time - train_start_time)  << ", test. time: "<< test_end_time - test_start_time << endl;
				performance_avg += performance;
				train_time_avg  += (test_start_time - train_start_time);

			}
			cout << "average performance : " << (performance_avg) *100/max_repeat << endl;
			cout << "average train. time : " << train_time_avg/max_repeat << endl;
			cout << "average merging time : " << endl;
			performance_avg = 0.0; train_time_avg = 0.0;
		}

	if (output_path != "NOPATH") output_file.close();
	return 0;
}


int main(int argc, char** argv) {
	if (argc > 1) { 
		int exit = parseCLI(argc, argv);
		return exit;
	} 
	cout << "No parameters are given! please check the BSVM scheme." << endl;
	cout << "Example for a scheme:" << endl;
	cout << "--method 0 --heuristic 0 --C 32 --gamma 0.0078 --B 100 --train  ~/adult_train --test ~/adult_test --output ~/adult_output --lookup ~/lookup-WD" << endl;
}
