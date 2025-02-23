function visualize2DFEmbeddings(X, OutU, OutLabelsU, OutT, OutLabelsT)
    % This function computes a DTW dissimilarity matrix, performs MDS, and visualizes the results.
    % Inputs:
    %   - X: Binary feature selection vector
    %   - OutU: Cell array of training samples (feature matrices)
    %   - OutLabelsU: Labels corresponding to training samples
    %   - OutT: Cell array of test samples (feature matrices)
    %   - OutLabelsT: Labels corresponding to test samples
    % Outputs:
    %   - 2D visualization of classes based on MDS
    
    % Combine training and test samples
    AllData = [OutU; OutT];
    AllLabels = [OutLabelsU; OutLabelsT]; % Ensure labels are concatenated similarly
    
    % Ensure X is binary (values above 0.5 are set to 1)
    X = X > 0.5; 
    
    numSamples = length(AllData);
    
    % Preallocate the distance matrix D
    D = zeros(numSamples, numSamples); 
    
    % Compute DTW dissimilarity matrix for all features (without feature selection)
    for i = 1:numSamples-1
        for j = i+1:numSamples
            % Compute DTW distance (normalized as in validation)
            dist = dtw(AllData{i}, AllData{j}, ...
                ceil(min(size(AllData{i}, 2), size(AllData{j}, 2)) / 10), 'euclidean');
            dist = dist / (size(AllData{i}, 2) + size(AllData{j}, 2)); % Normalize by sequence lengths

            % Store distances in symmetric matrix
            D(i, j) = dist;
            D(j, i) = dist;
        end
    end
    
    % Run MDS on the full distance matrix (using all features)
    Y_all = mdscale(D, 2, 'Start', 'random');
    
    % Compute DTW dissimilarity matrix for selected features (based on X)
    D_selected = zeros(numSamples, numSamples); 
    
    for i = 1:numSamples-1
        for j = i+1:numSamples
            % Compute DTW distance (normalized as in validation) for selected features
            dist = dtw(AllData{i}(X == 1, :), AllData{j}(X == 1, :), ...
                ceil(min(size(AllData{i}, 2), size(AllData{j}, 2)) / 10), 'euclidean');
            dist = dist / (size(AllData{i}, 2) + size(AllData{j}, 2)); % Normalize by sequence lengths

            % Store distances in symmetric matrix
            D_selected(i, j) = dist;
            D_selected(j, i) = dist;
        end
    end
    
    % Run MDS on the selected distance matrix (using selected features)
    Y_selected = mdscale(D_selected, 2, 'Start', 'random');
    
    % Visualization
    figure;
    
    % Subplot 1: Using all features
    subplot(1, 2, 1);
    gscatter(Y_all(:, 1), Y_all(:, 2), AllLabels);
    title('MDS - All Features');
    xlabel('Dimension 1');
    ylabel('Dimension 2');
    
    % Subplot 2: Using selected features (based on X)
    subplot(1, 2, 2);
    gscatter(Y_selected(:, 1), Y_selected(:, 2), AllLabels);
    title('MDS - Selected Features');
    xlabel('Dimension 1');
    ylabel('Dimension 2');
end
