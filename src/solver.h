#pragma once

#include "loadData.h"
#include "kernel.h"
#include "budgetMaintenance.h"
#include "svm.h"


//primal solver with budget
SVM BSGD(sparseData& dataset, sparseData& testdataset, double C, Kernel const& kernel, LookupTable& wd_parameters, double accuracy, size_t B, size_t max_epochs, Heuristic heuristic);

//primal solver with budget
SVM BDCA(sparseData& dataset, sparseData& testdataset, double C, Kernel const& kernel, LookupTable& wd_parameters, double accuracy, size_t B, size_t max_epochs, Heuristic heuristic);

SVM acfBDCA(sparseData& dataset, sparseData& testdataset, double C, Kernel const& kernel, LookupTable& wd_parameters, double accuracy, size_t B, size_t max_epochs, Heuristic heuristic);

