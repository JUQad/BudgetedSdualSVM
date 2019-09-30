
#include "loadData.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

using namespace std;


//load sparse data files 
void loaddata(string filename, sparseData& data, bool shuffle)
{
	size_t dataset_size;
	// first pass: find number of non-empty lines and entries
	size_t n = 0;
	size_t nnz = 0;
	{
		ifstream ifs(filename.c_str());
		if (! ifs.good()) throw runtime_error(string("[loaddata] failed to load file ").append(filename));
		while (true)
		{
			string s;
			if (! getline(ifs, s)) break;
			if (s.empty()) break;

			for (size_t i=0; i<s.size(); i++) if (s[i] == ':') nnz++;
			n++;
		}
	}

	// second pass: load contents
	{
		ifstream ifs(filename.c_str());
		if (! ifs.good()) throw runtime_error(string("[loaddata] failed to load file ").append(filename));
		for (size_t i=0; i<n; i++)
		{
			string s;
			if (! getline(ifs, s)) break;
			if (s.empty()) break;

			// extract label
			size_t start = 0;
			size_t end = min(s.find(' ', start), s.size());
			string y = s.substr(start, end - start);
			int i_dec = stoi (y);
			if (i_dec%2 == 0)
				data.labels.push_back((char)-1);
			else
				data.labels.push_back((char)strtol(y.c_str(), NULL, 10));
			for (start = end; start < s.size() && s[start] == ' '; start++);

			// extract features
			vector<SE> data_storage;
			while (start < s.size())
			{
				size_t colon = s.find(':', start);
				if (colon == string::npos) throw runtime_error(string("[loaddata] failed to load file ").append(filename));
				end = min(s.find(' ', colon), s.size());
				size_t feature = strtol(s.substr(start, colon - start).c_str(), NULL, 10) - 1;
				double value = strtod(s.substr(colon + 1, end - colon - 1).c_str(), NULL);
				data_storage.push_back(SE(feature, value));
				for (start = end; start < s.size() && s[start] == ' '; start++);
			}
			data_storage.push_back(SE(END(), 0.0));
			data.data.push_back(data_storage);
		}
	}

	dataset_size = data.data.size();
	if (shuffle) {
		// Initialize vector with possible indices and shuffle 
		vector<INDEX> indices(dataset_size, 0);
		auto walker = indices.begin();
		auto end = indices.end();
		INDEX inc = 0;
		while (walker != end) {
			*walker++ = inc;
			++inc;
		}
		random_shuffle(indices.begin(), indices.end());
		permuteVector(data.data, indices);
		permuteVector(data.labels, indices);
	}
}

vector<SE> scaleAddSparseVectors_new(vector<SE> const& first_operand, vector<SE> const& second_operand, double first_scale, double second_scale) {
	vector<SE> result;

	size_t i = 0;
	size_t j = 0;
	size_t x1_size = first_operand.size();
	size_t x2_size = second_operand.size();
	INDEX index_i = (i >= x1_size) ? END():first_operand[i].index;
	INDEX index_j =  (j >= x2_size) ? END():second_operand[j].index;

	while (index_i != END() ||index_j != END()) {
		if (index_i < index_j) {
			result.push_back(SE(first_operand[i].index, first_operand[i].value*first_scale));
			i++;
		}

		if (index_i > index_j) {
			result.push_back(SE(second_operand[j].index, second_operand[j].value*second_scale));
			j++;
		}

		if (index_i == index_j) {
			result.push_back(SE(second_operand[j].index, first_operand[i].value*first_scale + second_operand[j].value*second_scale));
			i++;
			j++;
		}

		index_i = (i >= x1_size) ? END():first_operand[i].index;
		index_j =  (j >= x2_size) ? END():second_operand[j].index;
	}
	result.push_back(SE(END(), 0.0));
	return result;
}


tuple<vector<double>, vector<double>, vector<double> > load_lookup (string filename)
{
	size_t size		 = 400;
	size_t n			= 0;

	vector<double> wdValue;
	vector<double> mValue(size+1);
	vector<double> kernelValue(size+1);

	long double initial = 0.0;
	long double step_m  = (1 - initial )/ size;
	long double step_k  = (1 - initial) / size;

	//for the server this should be changed to push_back(), i guess!
	for (unsigned int i = 0; i< size+1; i++)
	{
		mValue.push_back( initial + i * step_m);
		kernelValue.push_back(  initial + i * step_k);
	}

	//first pass
	{
		ifstream ifs(filename.c_str());

		if (! ifs.good()) throw runtime_error(string("[loaddata] failed to load file ").append(filename));
		while (true)
		{
			string s;
			if (! getline(ifs, s)) break;
			if (s.empty()) break;

			n++;
		}
	}
	//second pass
	{

		ifstream ifs(filename.c_str());
		if (! ifs.good()) throw runtime_error(string("[loaddata] failed to load file ").append(filename));
		for (size_t i=0; i<n; i++)
		{


			string s;
			if (! getline(ifs, s)) break;
			if (s.empty()) break;
			size_t start = 0;
			size_t end = min(s.find(' ', start), s.size());
			while (start < s.size())
			{
				double wd_val = strtod (s.substr(start, end).c_str(), NULL);
				wdValue.push_back(wd_val);
				for (start = end; start < s.size() && s[start] == ' '; start++);
			}
		}
	}
	tuple<vector<double>, vector<double>, vector<double> > results;
	get<0>(results) = mValue; //wdValue;//mValue;
	get<1>(results) = kernelValue; //wdValue;//kernelValue;
	get<2>(results) = wdValue;
	return results;
}


