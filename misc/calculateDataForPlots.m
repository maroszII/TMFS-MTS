% --- Calculate data for plots ----------------------------
MeanFbest = zeros(1, Num_Methods);
MeanFnumber = zeros(1, Num_Methods);
MeanETime = zeros(1, Num_Methods);
MeanAcc = zeros(1, Num_Methods);
MeanCcurve = cell(1, Num_Methods);

for i = 1:Num_Methods
    % Mean of fbest values
    FbestValues = arrayfun(@(x) x.fbest, Results(i, :));
    MeanFbest(i) = mean(FbestValues);
    
    % Mean of fnumber values
    FnumberValues = arrayfun(@(x) x.fnumber, Results(i, :));
    MeanFnumber(i) = mean(FnumberValues);
    
    % Mean of accuracy values (accuracy is in (1))
    AccValues = arrayfun(@(x) x.acc(1), Results(i, :));
    MeanAcc(i) = mean(AccValues);

    % Mean of elapsed time values
    EtimeValues = arrayfun(@(x) x.elapsedTime, Results(i, :));
    MeanETime(i) = mean(EtimeValues);

    % Mean of convergence curves
    Ccurves = arrayfun(@(x) x.ccurve, Results(i, :), 'UniformOutput', false);
    minLength = min(cellfun(@(x) length(x), Ccurves));
    CcurvesTrimmed = cellfun(@(x) x(1:minLength), Ccurves, 'UniformOutput', false);
    MeanCcurve{i} = mean(cell2mat(CcurvesTrimmed'), 1);
	
	% Find the best feature selection vector `X`
    [~, bestIdx] = min(FbestValues); % Assuming lower fbest is better
    BestX = Results(i, bestIdx).xbest;  % Extract the best X for this method

    % Visualize using DTW-MDS
    fprintf('\nVisualizing method %d...', i);
    visualize2DFEmbeddings(BestX, OutU, OutLabelsU, OutT, OutLabelsT);	
	
end