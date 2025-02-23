function result = determinePerformance(X, OutU, OutLabelsU, OutTV, OutLabelsTV, computeMetrics)
    % Description:
    %   Function evaluates classification performance using a 1-NN
    %   classifier with Dynamic Time Warping (DTW).
    %   It supports multiple evaluation metrics, including Accuracy, F1-score,
    %   Precision, Recall, Specificity, Balanced Accuracy, and MCC.
    %
    % Inputs:
    %   - X: Feature selection vector (binary)
    %   - OutU: Cell array of training feature matrices
    %   - OutLabelsU: Labels corresponding to training features
    %   - OutTV: Cell array of test/validation feature matrices
    %   - OutLabelsTV: Labels corresponding to test/validation features
    %   - computeMetrics: Numeric flag or array determining which metrics to compute:
    %       1 - Accuracy only
    %       2 - Macro F1-score
    %       3 - Macro Precision
    %       4 - Recall
    %       5 - Specificity
    %       6 - Balanced Accuracy
    %       7 - MCC (Matthews Correlation Coefficient)
    %       0 - Compute all metrics, but only Accuracy is used for optimization
    %
    % Outputs:
    %   - result: Array containing computed metrics

    % Apply feature selection (convert binary selection vector to logical mask)
    X = X > 0.5;

    % Define the distance metric for DTW
    metric = 'euclidean';

    % Identify unique labels in the dataset
    uniqueLabels = unique([OutLabelsU; OutLabelsTV]);
    numClasses = length(uniqueLabels);

    % Initialize accuracy-related counters
    correct = 0;
    sizeTT = length(OutTV);
    sizeU = length(OutU);

    % Initialize storage for metrics
    TP = zeros(numClasses, 1);
    FP = zeros(numClasses, 1);
    FN = zeros(numClasses, 1);
    TN = zeros(numClasses, 1);

    % Iterate through test/validation samples
    for i = 1:sizeTT
        test_features = OutTV{i}(X == 1, :);
        closest_index = 1;
        min_distance = inf;

        % Compare the test/validation sample with all training samples
        for k = 1:sizeU
            train_features = OutU{k}(X == 1, :);
            window = ceil(min(size(test_features, 2), size(train_features, 2)) / 10);
            distance = dtw(test_features, train_features, window, metric) / ...
                       (size(test_features, 2) + size(train_features, 2));
            
            if distance < min_distance
                min_distance = distance;
                closest_index = k;
            end
        end

        % Retrieve predicted and actual labels
        predicted_label = OutLabelsU(closest_index);
        actual_label = OutLabelsTV(i);

        % Convert labels to indices within the uniqueLabels array
        actual_label_idx = find(uniqueLabels == actual_label, 1);
        predicted_label_idx = find(uniqueLabels == predicted_label, 1);

        % Compute accuracy
        if predicted_label == actual_label
            correct = correct + 1;
        end

        % Update TP, FP, FN, TN
        TP(actual_label_idx) = TP(actual_label_idx) + (predicted_label == actual_label);
        FP(predicted_label_idx) = FP(predicted_label_idx) + (predicted_label ~= actual_label);
        FN(actual_label_idx) = FN(actual_label_idx) + (predicted_label ~= actual_label);
        
        % Compute TN correctly for each class
        for j = 1:numClasses
            if j ~= actual_label_idx && j ~= predicted_label_idx
                TN(j) = TN(j) + 1;
            end
        end
    end

    % Compute classification accuracy
    accuracy = correct / sizeTT;

    % Initialize result array
    result = zeros(1, 7);
    result(1) = accuracy;

    % Compute additional metrics based on computeMetrics
    if computeMetrics == 0 || computeMetrics == 2 || computeMetrics == 3 || computeMetrics == 4 || computeMetrics == 5 || computeMetrics == 6 || computeMetrics == 7
        Precision = TP ./ (TP + FP);
        Recall = TP ./ (TP + FN);
        Specificity = TN ./ (TN + FP);

        % Handle NaN values (caused by division by zero)
        Precision(isnan(Precision)) = 0;
        Recall(isnan(Recall)) = 0;
        Specificity(isnan(Specificity)) = 0;

        % Compute and fill in the result array for each required metric
        if computeMetrics == 0 || computeMetrics == 2
            FMeasure = 2 * (Precision .* Recall) ./ (Precision + Recall);
            result(2) = mean(FMeasure, 'omitnan');
        end
        if computeMetrics == 0 || computeMetrics == 3
            result(3) = mean(Precision, 'omitnan');
        end
        if computeMetrics == 0 || computeMetrics == 4
            result(4) = mean(Recall, 'omitnan');
        end
        if computeMetrics == 0 || computeMetrics == 5
            result(5) = mean(Specificity, 'omitnan');
        end
        
        % Compute Balanced Accuracy only if required
        if computeMetrics == 0 || computeMetrics == 6
            BalancedAccuracy = (Recall + Specificity) / 2;
            result(6) = mean(BalancedAccuracy, 'omitnan');
        end
    end

    % Compute MCC only if requested
    if computeMetrics == 0 || computeMetrics == 7
        MCC = (TP .* TN - FP .* FN) ./ sqrt((TP + FP) .* (TP + FN) .* (TN + FP) .* (TN + FN));
        % Handle NaN values (caused by division by zero)
        MCC(isnan(MCC)) = 0;
        result(7) = mean(MCC, 'omitnan');
    end
end
