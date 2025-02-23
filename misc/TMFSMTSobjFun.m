% Returns objective function value as classification error.
function objFv = TMFSMTSobjFun(X, options)
    alpha = 0.99;
    beta = 0.01;
    
    % Objective function works only on training and validation subsets  
    val = determinePerformance(X, options.OutU, options.OutLabelsU, options.OutV, options.OutLabelsV, options.computeMetrics);     
    val = val + determinePerformance(X, options.OutV, options.OutLabelsV, options.OutU, options.OutLabelsU, options.computeMetrics);     
    val = val / 2;  % Mean value 
    
    % Swapping small validation set with larger training set prevents overfitting to the validation set and promotes features that lead to 
    % a classifier with better generalization capabilities. For even better classifier, time-consuming k-fold cross-validation should be considered.  
    
    % %   - computeMetrics: Numeric flag determining which metrics to compute
    % %       1 - Accuracy only
    % %       2 - Macro F1-score
    % %       3 - Macro Precision
    % %       4 - Recall
    % %       5 - Specificity
    % %       6 - Balanced Accuracy
    % %       7 - MCC (Matthews Correlation Coefficient)
    % %       0 - Compute all metrics
    
    % Compute objective function value based on selected metric
    if options.computeMetrics == 1 ||options.computeMetrics == 0
        error = 1 - val(1); % Accuracy (minimizing error)
    elseif ismember(options.computeMetrics, [2, 3, 4, 5, 6, 7])
        error = -val(options.computeMetrics); % Other metrics (maximize by minimizing negative value)
    else
        error = -val(options.computeMetrics); % Default behavior for unknown metrics
    end
    
    % Adjust the feature selection penalty term based on metric type
    if ismember(options.computeMetrics, [2, 3, 4, 5, 6, 7])
        penalty = - (sum(X == 1) / length(X)); % Invert penalty for maximizing metrics
    else
        penalty = sum(X == 1) / length(X); % Standard penalty for error-based metrics
    end
    
    % Compute final objective function value
    if sum(X == 1) > 0
        objFv = alpha * error + beta * penalty;
    else
        objFv = 1;
    end
	  
 
end
