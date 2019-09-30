#pragma once

#include "loadData.h"
#include "kernel.h"
#include <tuple>
#include <vector>


using LookupTable = std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>;
using Heuristic = std::tuple<INDEX, INDEX>(std::vector<double>&, sparseData&, Kernel const&, LookupTable const&, double);

Heuristic mergeHeuristicWD;
Heuristic mergeHeuristicRandom;
Heuristic mergeHeuristicKernel;
Heuristic mergeHeuristicRandomWD;
Heuristic mergeHeuristicMinAlpha;
Heuristic mergeHeuristicMintwoAlphas;
Heuristic mergeHeuristic59plusWD;


int mergeAndDeleteSV(std::vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C, Heuristic heuristic);

