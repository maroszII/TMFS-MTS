 
% Polar Lights Optimizer: Algorithm and Applications in Image Segmentation and Feature Selection:
% Chong Yuan, Dong Zhao, Ali Asghar Heidari, Lei Liu, Yi Chen, Huiling Chen
% Neurocomputing - 2024
function result = TMFSMTSplo(frame, options)
tic

 
lb = 0;
ub = 1;
threshold = 0.5;
 
if isfield(options,'N'), N = options.N; end%% 
if isfield(options,'max_Iter'), max_Iter = options.max_Iter; end%% 
if isfield(options,'maxEvals'), maxEvals = options.maxEvals; end%% 
 

% Dimensions of the feature set
dim = size(frame, 2);

%% Initialization
FEs = 0; % Function evaluations counter
it = 1; % Iteration counter
fitness = inf * ones(N, 1); % Fitness values for current population
fitness_new = inf * ones(N, 1); % Fitness values for new population
Convergence_curve = []; % Convergence curve

% Initialize population and velocity
X = lb + (ub - lb) .* rand(N, dim); % Initial positions
V = ones(N, dim); % Initial velocities
X_new = zeros(N, dim);

% Evaluate initial population
for i = 1:N
    fitness(i) = TMFSMTSobjFun(X(i, :) > threshold, options);
    FEs = FEs + 1;
end

% Sort population by fitness
[fitness, SortOrder] = sort(fitness);
X = X(SortOrder, :);
Best_pos = X(1, :); % Best position
Best_score = fitness(1); % Best fitness
Convergence_curve(it) = Best_score; % Record initial best fitness

%% Main loop
while it <= max_Iter
    if FEs >= maxEvals
        break;
    end

    % Calculate global mean
    X_mean = mean(X, 1);
    w1 = tansig((FEs / maxEvals)^4);
    w2 = exp(-(2 * FEs / maxEvals)^3);

    % Update positions
    for i = 1:N
        a = rand() / 2 + 1;
        V(i, :) = 1 * exp((1 - a) / 100 * FEs); % Local search component
        LS = V(i, :);

        GS = Levy(dim) .* (X_mean - X(i, :) + (lb + rand(1, dim) .* (ub - lb)) / 2); % Global search component
        X_new(i, :) = X(i, :) + (w1 * LS + w2 * GS) .* rand(1, dim);
    end

    % Fine-tuning with random perturbation
    E = sqrt(FEs / maxEvals);
    A = randperm(N);
    for i = 1:N
        for j = 1:dim
            if (rand < 0.05) && (rand < E)
                X_new(i, j) = X(i, j) + sin(rand * pi) * (X(i, j) - X(A(i), j));
            end
        end

        % Boundary handling
        Flag4ub = X_new(i, :) > ub;
        Flag4lb = X_new(i, :) < lb;
        X_new(i, :) = (X_new(i, :) .* ~(Flag4ub + Flag4lb)) + ub .* Flag4ub + lb .* Flag4lb;

        % Evaluate new fitness
        fitness_new(i) = TMFSMTSobjFun(X_new(i, :) > threshold, options);
        FEs = FEs + 1;

        % Update if new fitness is better
        if fitness_new(i) < fitness(i)
            X(i, :) = X_new(i, :);
            fitness(i) = fitness_new(i);
        end
    end

    % Sort population by fitness
    [fitness, SortOrder] = sort(fitness);
    X = X(SortOrder, :);

    % Update best solution
    if fitness(1) < Best_score
        Best_pos = X(1, :);
        Best_score = fitness(1);
    end

   
    Convergence_curve(it) = Best_score;

     aP=determinePerformance(Best_pos,options.OutU,options.OutLabelsU, options.OutT,options.OutLabelsT,options.computeMetrics);
	fprintf('\n  Alg  %s Iteration %d Accuracy = %f  FBest %f',options.name, it, aP(1)  ,Convergence_curve(it)   ) 
	 % Update convergence curve
    it = it + 1;
	
end

%% Result struct
	result.fbest = Best_score; % Best fitness
	result.xbest = Best_pos; % Best position
	result.fnumber = sum((result.xbest > threshold) == 1); % Number of selected features
	result.ccurve = Convergence_curve; % Convergence curve
	result.elapsedTime=toc;
	result.acc=determinePerformance(result.xbest,options.OutU,options.OutLabelsU, options.OutT,options.OutLabelsT,options.computeMetrics) ;
 
end

%% Levy Flight Function
function o = Levy(d)
beta = 1.5;
sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
u = randn(1, d) * sigma;
v = randn(1, d);
step = u ./ abs(v).^(1 / beta);
o = step;
end
