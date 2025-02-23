
% --- Calculate data for plots -----------------------------------------
calculateDataForPlots 

% Plot 1: Bar plot for mean number of selected features
figure;
bar(MeanFnumber);
xlabel('Method');
ylabel('Mean number of selected features');
title('Mean number of selected features for each method');
xticks(1:Num_Methods);
xticklabels({'AROA', 'RIME', 'PLO', 'COA', 'MPA', 'EO', 'GNDO', 'SCA', 'HHO', 'AO'});

% Plot 2: Bar plot for mean accuracy
figure;
bar(MeanAcc);
xlabel('Method');
ylabel('Mean Accuracy');
title('Mean Accuracy');
xticks(1:Num_Methods);
xticklabels({'AROA', 'RIME', 'PLO', 'COA', 'MPA', 'EO', 'GNDO', 'SCA', 'HHO', 'AO'});

% Plot 3: Bar plot for mean elapsed time
figure;
bar(MeanETime);
xlabel('Method');
ylabel('Mean Elapsed Time');
title('Mean Elapsed Time');
xticks(1:Num_Methods);
xticklabels({'AROA', 'RIME', 'PLO', 'COA', 'MPA', 'EO', 'GNDO', 'SCA', 'HHO', 'AO'});

% Plot 4: Convergence curves based on maxEvals
figure;
hold on;
line_styles = {'-', '--', ':', '-.'};
markers = {'o', '*', 's', 'd', '^', 'v', '>', '<', 'p', 'h'};
colors = lines(Num_Methods);
target_length = 100;
method_names = {'AROA', 'RIME', 'PLO', 'COA', 'MPA', 'EO', 'GNDO', 'SCA', 'HHO', 'AO'};
MeanCcurve_rescaled = cell(1, Num_Methods);

for i = 1:Num_Methods
    original_length = length(MeanCcurve{i});
    MeanCcurve_rescaled{i} = interp1(linspace(0, 1, original_length), MeanCcurve{i}, linspace(0, 1, target_length));
end

for i = 1:Num_Methods
    plot(linspace(0, 100, target_length), MeanCcurve_rescaled{i}, 'DisplayName', method_names{i}, ...
        'LineStyle', line_styles{mod(i-1, length(line_styles)) + 1}, ...
        'Marker', markers{mod(i-1, length(markers)) + 1}, ...
        'Color', colors(i, :), 'MarkerIndices', 1:10:target_length);
end

xlabel('Function evaluations (%)');
ylabel('Mean objective function value');
title('Mean convergence curves for methods');
legend show;
hold off;

% Plot 5: Boxplot of best fitness values
FbestMatrix = zeros(Num_Runs, Num_Methods);
for i = 1:Num_Methods
    FbestMatrix(:, i) = arrayfun(@(x) x.fbest, Results(i, :));
end

figure;
boxplot(FbestMatrix, 'Labels', {'AROA', 'RIME', 'PLO', 'COA', 'MPA', 'EO', 'GNDO', 'SCA', 'HHO', 'AO'});
xlabel('Method');
ylabel('Best fitness values');
title('Comparison of best fitness values'); 

 % --- Wilcoxon test and mean-based comparison ---
WilcoxonResults = zeros(Num_Methods, Num_Methods);
WilcoxonBinary = zeros(Num_Methods, Num_Methods);
MeanFbest = mean(FbestMatrix, 1); %  

for i = 1:Num_Methods
    for j = i+1:Num_Methods
        [p, ~, stats] = signrank(FbestMatrix(:, i), FbestMatrix(:, j));

        %  p-values
        WilcoxonResults(i, j) = p;
        WilcoxonResults(j, i) = p;

        % Statistical significance (alpha = 0.05)
        if p < 0.05
            if MeanFbest(i) < MeanFbest(j)    
                WilcoxonBinary(i, j) = 1;  % i is better than j
                WilcoxonBinary(j, i) = -1; % j is worse than i
            else
                WilcoxonBinary(i, j) = -1; % i is worse than j
                WilcoxonBinary(j, i) = 1;  % j is better than i
            end
        else
            WilcoxonBinary(i, j) = 0; % Lack of statistical difference
            WilcoxonBinary(j, i) = 0;
        end
    end
end

% --- Heatmap: Wilcoxon Test (p-value) ---
MethodNames = {'AROA', 'RIME', 'PLO', 'COA', 'MPA', 'EO', 'GNDO', 'SCA', 'HHO', 'AO'};
figure;
h1 = heatmap(MethodNames, MethodNames, WilcoxonResults, ...
    'Colormap', parula, 'ColorbarVisible', 'on', 'CellLabelFormat', '%.4f');
h1.Title = 'Heatmap: Wilcoxon Test for 10 Methods (p-value)';
h1.XLabel = 'Method (column)';
h1.YLabel = 'Method (row)';

% --- Heatmap: 0/1/-1 Comparison ---
figure;
h2 = heatmap(MethodNames, MethodNames, WilcoxonBinary, ...
    'Colormap', jet, 'ColorbarVisible', 'on', 'CellLabelFormat', '%d');
h2.Title = 'Heatmap: 0/1/-1 Statistical Significance';
h2.XLabel = 'Method (column)';
h2.YLabel = 'Method (row)';
fprintf('\n');

% --- LaTex Table  ---
MeanFbest = zeros(1, Num_Methods);
MeanFnumber = zeros(1, Num_Methods);
MeanETime = zeros(1, Num_Methods);
MeanAcc = zeros(1, Num_Methods);
MeanFbest_std = zeros(1, Num_Methods);
MeanFnumber_std = zeros(1, Num_Methods);
MeanETime_std = zeros(1, Num_Methods);
MeanAcc_std = zeros(1, Num_Methods);
MeanFbest_4dec = zeros(1, Num_Methods);

% Calculating the means and standard deviations
for i = 1:Num_Methods
    % Mean and standard deviation of fbest values
    FbestValues = arrayfun(@(x) x.fbest, Results(i, :));
    MeanFbest(i) = mean(FbestValues);
    MeanFbest_std(i) = std(FbestValues);
    
    % Mean and standard deviation of fnumber values
    FnumberValues = arrayfun(@(x) x.fnumber, Results(i, :));
    MeanFnumber(i) = mean(FnumberValues);
    MeanFnumber_std(i) = std(FnumberValues);
    
    % Mean and standard deviation of accuracy values
    AccValues = arrayfun(@(x) x.acc(1), Results(i, :));
    MeanAcc(i) = mean(AccValues);
    MeanAcc_std(i) = std(AccValues);
    
    % Mean and standard deviation of elapsed time values
    EtimeValues = arrayfun(@(x) x.elapsedTime, Results(i, :));
    MeanETime(i) = mean(EtimeValues);
    MeanETime_std(i) = std(EtimeValues);
end

% --- Find best methods per column ---
[~, bestFnumberIdx] = min(MeanFnumber); % Min number of selected features (best is less)
[~, bestAccIdx] = max(MeanAcc); % Max accuracy
[~, bestETimeIdx] = min(MeanETime); % Min elapsed time
[~, bestFbestIdx] = min(MeanFbest); % Min objective function

% Open file for writing the table
fileID = fopen('results_table.tex', 'w');

% Write table header
fprintf(fileID, '\\begin{table}[h]\n');
fprintf(fileID, '\\centering\n');
fprintf(fileID, '\\caption{Comparison of methods based on feature selection, accuracy on the testing subset, elapsed time, and objective function.} \n');
fprintf(fileID, '\\resizebox{\\textwidth}{!}{\n');
fprintf(fileID, '\\begin{tabular}{|c|c|c|c|c|}\n');
fprintf(fileID, '\\hline\n');
fprintf(fileID, 'Method & Mean selected features & Mean accuracy & Mean elapsed time (s) & Mean objective function \\\\ \n');
fprintf(fileID, '\\hline\n');

% Iterate over methods and write data to the table
for i = 1:Num_Methods
    % Bold the best values for each column
    if i == bestFnumberIdx
        boldFnumber = sprintf('\\textbf{%.2f} $\\pm$ %.2f', MeanFnumber(i), MeanFnumber_std(i));
    else
        boldFnumber = sprintf('%.2f $\\pm$ %.2f', MeanFnumber(i), MeanFnumber_std(i));
    end

    if i == bestAccIdx
        boldAcc = sprintf('\\textbf{%.4f} $\\pm$ %.3f', MeanAcc(i), MeanAcc_std(i));  
    else
        boldAcc = sprintf('%.4f $\\pm$ %.3f', MeanAcc(i), MeanAcc_std(i)); 
    end

    if i == bestETimeIdx
        boldETime = sprintf('\\textbf{%.2f} $\\pm$ %.2f', MeanETime(i), MeanETime_std(i));
    else
        boldETime = sprintf('%.2f $\\pm$ %.2f', MeanETime(i), MeanETime_std(i));
    end

    if i == bestFbestIdx
        boldFbest = sprintf('\\textbf{%.4f} $\\pm$ %.4f', MeanFbest(i), MeanFbest_std(i));
    else
        boldFbest = sprintf('%.4f $\\pm$ %.4f', MeanFbest(i), MeanFbest_std(i));
    end

    % Write method and corresponding data to table
    fprintf(fileID, '%s & %s & %s & %s & %s \\\\ \n', ...
        method_names{i}, boldFnumber, boldAcc, boldETime, boldFbest);
    fprintf(fileID, '\\hline\n');
end

% Close the table
fprintf(fileID, '\\end{tabular}\n');
fprintf(fileID, '}\n');
fprintf(fileID, '\\label{tab:comparison}\n');
fprintf(fileID, '\\end{table}\n');

% Close file
fclose(fileID);
