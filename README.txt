-----------------------------------------------------------
-- BudgeteddualSVM_SW: A Toolbox for Large-scale Non-linear Dual SVM Training on a Budget--
-----------------------------------------------------------

BudgeteddualSVM is an efficient software for large-scale
non-linear dual SVM classification. 


Table of Contents
=================
- "BudgeteddualSVM_SW" Usage
- Example
- Acknowledgment



`BudgeteddualSVM_SW' Usage
==========================
In order to get the detailed usage description, run the BudgeteddualSVM_SW function
without providing any arguments.

Usage:
BudgeteddualSVM_SW [options] train test [output_file]

Inputs:
options        	- parameters of the model
train_file     	- url of training file in LIBSVM format
test_file     	- url of testing file in LIBSVM format
output_file     - file that will hold a learned model
	--------------------------------------------
Options are specified in the following format:
'-OPTION1 VALUE1 -OPTION2 VALUE2 ...'

--method [0 ... 2] 	(default: BSGD)
--heuristic [0 ... 7] 	(default: Minimum Weight Degradation)
--train 		%TRAINING_DATA% (LIBSVM Sparse Format)
--test 			%TEST_DATA% (LIBSVM Sparse Format)
--output 		%DIR_OF_THE_OUTPUT_FILE%
--lookup		%DIR_OF_THE_LOOKUP_FILE%
--C 			%HYPERPARAMETER_REGULARISATION%
--gamma 		%HYPERPARAMETER_KERNEL WIDTH%
--B 			%BUDGET_SIZE%
--epochs 		%EPOCHS_SIZE% 



Example: 
========
BudgeteddualSVM_SW --C 32 --gamma 0.0078 --B 500 --epochs 5 --method 0 --heuristic 0 --train data\\training_data.txt --test data\\testing_data.txt ----lookup wd_param.txt --output outputfile.txt
*** this will do training+evaluation using the provided hyperparameters and budget.
*** the results will be written into outputfile.txt 

Possible Heuristics:
0 = Minimum Weight Degradation
1 = Fully Random Selection
2 = Minimum Kernel Value
3 = Random Minimum Weight Degradation
4 = Removal (SV with Minimum Alpha)
5 = Merging SVs with Minimum Alphas
6 = Random-59 Sub-sampling
7 = LASVM



Possible Methods (Solvers):
0 = BSGD
1 = BSCA
2 = ACF-BSCA


Acknowledgment:
===============
This work is supported by the Deutsche Forschungsgemeinschaft (DFG) throughgrant GL 839/3-1.