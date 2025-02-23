% -------------------------------------------------------------------------
% Toolbox for Metaheuristic Feature Selection in Multivariate Time Series  
% (TMFS-MTS) - Demo 
% -------------------------------------------------------------------------
% Author: Mariusz Oszust & Marian Wysocki
% Affiliation: Rzeszow University of Technology
% Email: marosz@kia.prz.edu.pl 
% Date: 2025-02-22
% Version: 1.0  
% -------------------------------------------------------------------------
% Description:
% This script demonstrates the use of TMFS-MTS for feature selection in  
% multivariate time series using metaheuristic optimization techniques.  
% It includes dataset loading, parameter configuration, execution of  
% metaheuristic algorithms, and visualization of results.  
%
% Steps:
% 1. Setup optimization options.
% 2. Select and process the dataset.
% 3. Execute metaheuristic algorithms for feature selection.
% 4. Compute performance metrics.
% 5. Generate plots and visualize results.
%
% Note:
% - Make sure the required datasets are available in the 'features' folder.
% - The script allows selecting different evaluation metrics and datasets.
% -------------------------------------------------------------------------


clc, clear all, close all;
disp('Toolbox for Metaheuristic Feature Selection in Multivariate Time Series (TMFS-MTS)')

% --- Options Setup ---------------------------------------------------- 
options.N = 50;            % Population size
options.max_Iter = 2000;   % Maximum number of iterations
options.maxEvals = 2000;   % Maximum number of fitness function evaluations
options.computeMetrics = 1; % Metric selection 							
							% %       1 - Accuracy only
							% %       2 - Macro F1-score
							% %       3 - Macro Precision
							% %       4 - Recall
							% %       5 - Specificity
							% %       6 - Balanced Accuracy
							% %       7 - MCC (Matthews Correlation Coefficient)
							% %       0 - Compute all metrics
Num_Runs = 30;      % Number of runs of each method 
 
selectDataset = 1;          % Load dataset 
							% %       1 - MSRA I
							% %       2 - MSRA II
							% %       3 - MSRA III
							% %       4 - UTD 
							% %       5 - Florence
							% %       6 - UTK
							
% Add paths to useful folders						
addpath(genpath(fullfile(pwd, 'features')));
addpath(genpath(fullfile(pwd, 'methods')));
addpath(genpath(fullfile(pwd, 'misc')));

% --- Load and process selected dataset -------------------------------- 
datasetProcessing   
 
% --- Run experiments --------------------------------------------------
runMetaheuristics      

% --- Visualization ----------------------------------------------------
visualization

 
