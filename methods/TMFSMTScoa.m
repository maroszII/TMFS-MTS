% Jia, Heming & Rao, Honghua & Wen, Changsheng & Mirjalili, Seyedali. (2023). 
% Crayfish optimization algorithm. Artificial Intelligence Review. 1. 1. 10.1007/s10462-023-10567-4. 
function result = TMFSMTScoa(frame, options)
tic
 
lb = 0;
ub = 1;
threshold = 0.5;
 
 
if isfield(options,'N'), N = options.N; end% 
if isfield(options,'max_Iter'), max_Iter = options.max_Iter; end% 
if isfield(options,'maxEvals'), maxEvals = options.maxEvals; end% 
 

FEs=0;
% Dimensions of the feature set
dim = size(frame, 2);

%% Initialization
X = lb + (ub - lb) .* rand(N, dim); % Initialize population
fitness = inf * ones(N, 1); % Fitness values for population
Best_fitness = inf; % Best fitness
Best_position = zeros(1, dim); % Best solution position
% Convergence_curve = zeros(1, max_Iter); % Convergence curve
Convergence_curve=[];
% Evaluate initial population
for i = 1:N
    fitness(i) = TMFSMTSobjFun(X(i, :) > threshold, options);
	FEs = FEs + 1;
    if fitness(i) < Best_fitness
        Best_fitness = fitness(i);
        Best_position = X(i, :);
    end
end
Convergence_curve(1) = Best_fitness;

%% Main Loop
t = 1;
while t <= max_Iter
    if t >= max_Iter || sum(fitness == Best_fitness) > 10 || sum(~isfinite(fitness)) > 0
        break;
    end
	if FEs >= maxEvals
        break;
    end

    % Parameters for COA
    C = 2 - (t / max_Iter); % Eq. (7)
    temp = rand * 15 + 20; % Temperature for food behavior (Eq. 3)
    xf = (Best_position + Best_position) / 2; % Mean position (Eq. 5)
    Xfood = Best_position;

    X_new = zeros(N, dim); % Initialize updated positions

    for i = 1:N
        if temp > 30
            % Summer resort stage
            if rand < 0.5
                X_new(i, :) = X(i, :) + C * rand(1, dim) .* (xf - X(i, :)); % Eq. (6)
            else
                % Competition stage
                z = randi(N);
                X_new(i, :) = X(i, :) - X(z, :) + xf; % Eq. (8)
            end
        else
            % Foraging stage
            F1 = TMFSMTSobjFun(Xfood > threshold, options);
			FEs = FEs + 1;
            P = 3 * rand * fitness(i) / F1; % Eq. (4)
            if P > 2
                % Food is too big
                Xfood = exp(-1 / P) .* Xfood; % Eq. (12)
                for j = 1:dim
                    X_new(i, j) = X(i, j) + cos(2 * pi * rand) * Xfood(j) * p_obj(temp, t, max_Iter) - ...
                        sin(2 * pi * rand) * Xfood(j) * p_obj(temp, t, max_Iter); % Eq. (13)
                end
            else
                % Food is manageable
                X_new(i, :) = (X(i, :) - Xfood) * p_obj(temp, t, max_Iter) + ...
                    p_obj(temp, t, max_Iter) .* rand(1, dim) .* X(i, :); % Eq. (14)
            end
        end
    end

    % Boundary control
    X_new = max(lb, min(ub, X_new));

    % Update population
    for i = 1:N
        new_fitness = TMFSMTSobjFun(X_new(i, :) > threshold, options);
		FEs = FEs + 1;
        if new_fitness < fitness(i)
            X(i, :) = X_new(i, :);
            fitness(i) = new_fitness;
            if new_fitness < Best_fitness
                Best_fitness = new_fitness;
                Best_position = X_new(i, :);
            end
        end
    end

 
    Convergence_curve(t) = Best_fitness;
   aP=determinePerformance(Best_position,options.OutU,options.OutLabelsU, options.OutT,options.OutLabelsT,options.computeMetrics);
	fprintf('\n  Alg  %s Iteration %d Accuracy = %f  FBest %f',options.name, t, aP(1) ,Convergence_curve(t)   ) 
	   % Update convergence curve
    t = t + 1;
end

%% Result Struct
result.fbest = Best_fitness; % Best fitness
result.xbest = Best_position; % Best solution
result.fnumber = sum((result.xbest > threshold) == 1); % Number of selected features
result.ccurve = Convergence_curve; % Convergence curve
result.elapsedTime=toc;
result.acc=determinePerformance(result.xbest,options.OutU,options.OutLabelsU, options.OutT,options.OutLabelsT,options.computeMetrics) ;
 
end

%% Supporting Function for Probability Calculation (Eq. 4)
function y = p_obj(temp, t, max_Iter)
C = 0.2; % Scaling constant
Q = 3; % Standard deviation
y = C * (1 / (sqrt(2 * pi) * Q)) * exp(-(temp - 25)^2 / (2 * Q^2));
end
