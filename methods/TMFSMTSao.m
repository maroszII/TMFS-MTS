
% Artemisinin Optimization (AO) % 
% "Artemisinin Optimization based on Malaria Therapy: Algorithm and Applications to Medical Image Segmentation", Displays, Elsevier, 2024
% https://doi.org/10.1016/j.displa.2024.102740

function result = TMFSMTSao(frame, options)


tic
% Parameters
 

lb = 0; % Lower bound
ub = 1; % Upper bound
dim = size(frame, 2); % Number of dimensions (features)

threshold = 0.5;

if isfield(options, 'N'), N = options.N; end

if isfield(options, 'maxEvals'), maxEvals = options.maxEvals; end
maxFEs = maxEvals;

 if N<4
	 disp(' AO experiences problems; N is changed to 4') %marosz
	 N=4;
 end


% Initialization of the population
pop = lb + (ub - lb) .* rand(N, dim); % Random solutions
fitness = inf(1, N); % Objective function values
FEs = 0; % Objective function counter
it = 1; % Iteration counter
bestFitness = inf; % Best objective function value
bestPosition = []; % Best position
% convergenceCurve = zeros(1, maxFEs); % Convergence curve
convergenceCurve =[];% Convergence curve

% Calculate fitness for the initial population
for i = 1:N
    fitness(i) = TMFSMTSobjFun(pop(i, :) > threshold, options); % Threshold 0.5 for binary features
    FEs = FEs + 1;
end

% Find the best solution in the initial population
[bestFitness, idx] = min(fitness);
bestPosition = pop(idx, :);

% Main loop
while FEs <= maxFEs
    % Dynamic parameter K
    K = 1 - ((FEs^(1/6)) / (maxFEs^(1/6)));
    % Dynamic parameter E
    E = exp(-4 * (FEs / maxFEs));
    
    % Update the population
    newPop = pop; % Copy of the population
    for i = 1:N
        fitNorm = (fitness(i) - min(fitness)) / (max(fitness) - min(fitness) + eps); % Normalize fitness
        
        for j = 1:dim
            if rand < K
                if rand < 0.5
                    newPop(i, j) = pop(i, j) + E * pop(i, j) * (-1)^FEs;
                else
                    newPop(i, j) = pop(i, j) + E * bestPosition(j) * (-1)^FEs;
                end
            end
            
            if rand < fitNorm
                A = randperm(N);
                beta = (rand / 2) + 0.1;
                newPop(i, j) = pop(A(3), j) + beta * (pop(A(1), j) - pop(A(2), j));
            end
        end
        
        % Mutation
        newPop(i, :) = applyMutation(newPop(i, :), pop(i, :), bestPosition);
        % Reset boundary violations
        newPop(i, :) = boundaryReset(newPop(i, :), lb, ub, dim, bestPosition);
        
        % Calculate fitness of the new solution
        newFitness = TMFSMTSobjFun(newPop(i, :) > threshold, options);
        FEs = FEs + 1;
        
        % Select the better solution
        if newFitness < fitness(i)
            pop(i, :) = newPop(i, :);
            fitness(i) = newFitness;
        end
    end
    
    % Update the best solution
    [currentBestFitness, idx] = min(fitness);
    if currentBestFitness < bestFitness
        bestFitness = currentBestFitness;
        bestPosition = pop(idx, :);
    end
    
    % Record convergence
    convergenceCurve(it) = bestFitness;
		aP=determinePerformance(bestPosition, options.OutU, options.OutLabelsU, options.OutT, options.OutLabelsT, options.computeMetrics);
    fprintf('\n  Alg  %s Iteration %d Accuracy = %f  FBest %f', options.name, it, aP(1), convergenceCurve(it));
    it = it + 1;
end

% Feature selection
selectedFeatures = find(bestPosition > 0.5);
reducedFeatures = frame(:, selectedFeatures);

% Return results
result.fbest = bestFitness;
result.xbest = bestPosition;
result.fnumber = numel(selectedFeatures);
result.ccurve = convergenceCurve;
result.elapsedTime = toc;
result.acc = determinePerformance(result.xbest, options.OutU, options.OutLabelsU, options.OutT, options.OutLabelsT, options.computeMetrics); 

end

% Mutation function
function z = applyMutation(z, x, b)
    mutationRate1 = 0.05; % Mutation probability for x
    mutationRate2 = 0.2; % Mutation probability for b
    dim = numel(z);
    for j = 1:dim
        if rand < mutationRate1
            z(j) = x(j);
        end
        if rand < mutationRate2
            z(j) = b(j);
        end
    end
end

% Function for resetting boundary violations
function z = boundaryReset(z, lb, ub, dim, best)
    for j = 1:dim
        if z(j) > ub || z(j) < lb
            z(j) = best(j);
        end
    end
end