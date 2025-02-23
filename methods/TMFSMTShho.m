
% Harris hawks optimization: Algorithm and applications
% Ali Asghar Heidari, Seyedali Mirjalili, Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, Huiling Chen
% Future Generation Computer Systems, 
% DOI: https://doi.org/10.1016/j.future.2019.02.028
function result = TMFSMTShho(frame, options)
  
	tic

   % Parameters     
    beta = 1.5; % Levy component 
% Parameters   
	
	
	lb = 0;
	ub = 1;
 
	threshold =0.5;
 
	if isfield(options,'N'), N = options.N; end%%% 
	if isfield(options,'max_Iter'), max_Iter = options.max_Iter; end%% 
	if isfield(options,'maxEvals'), maxEvals = options.maxEvals; end%% 
	Max_iter= max_Iter;   
 
 

    % Number of dimensions
    dim = size(frame, 2);

    % Initialization of population
    X = lb + (ub - lb) .* rand(N, dim); % Initialize positions
    fitR = inf; % Best fitness
    fit = inf(1, N); % Fitness values for population

    % Preallocate for mean position and Levy flights
    Y = zeros(1, dim);
    Z = zeros(1, dim);

    % Convergence curve
    % Convergence_curve = zeros(1, max_Iter);
    Convergence_curve = [];
	FEs=0;
    % Iterations
    t = 1; % Iteration counter
    while t <= max_Iter
		if FEs >= maxEvals
			break;
		end
        % Evaluate fitness for the current population
        for i = 1:N
            fit(i) = TMFSMTSobjFun(X(i, :) > threshold, options); % Evaluate fitness
			FEs = FEs + 1;
            % Update the rabbit (best solution so far)
            if fit(i) < fitR
                fitR = fit(i); % Update best fitness
                Xrb = X(i, :); % Update best position
            end
        end

        % Mean position of hawks
        X_mu = mean(X, 1); % Equation (2)

        for i = 1:N
            % Random number in [-1, 1]
            E0 = -1 + 2 * rand();
            % Escaping energy of rabbit
            E = 2 * E0 * (1 - (t / max_Iter)); % Equation (3)

            % Exploration phase
            if abs(E) >= 1
                q = rand(); % Random number in [0, 1]
                if q >= 0.5
                    % Randomly select a hawk k
                    k = randi([1, N]);
                    r1 = rand();
                    r2 = rand();
                    % Position update (1)
                    X(i, :) = X(k, :) - r1 * abs(X(k, :) - 2 * r2 * X(i, :));
                else
                    r3 = rand();
                    r4 = rand();
                    % Update Hawk (1)
                    X(i, :) = (Xrb - X_mu) - r3 * (lb + r4 * (ub - lb));
                end
            % Exploitation phase
            else
                J = 2 * (1 - rand()); % Jump strength
                r = rand(); % Random number
                if r >= 0.5 && abs(E) >= 0.5
                    % Soft besiege
                    X(i, :) = Xrb - E * abs(J * Xrb - X(i, :));
                elseif r >= 0.5 && abs(E) < 0.5
                    % Hard besiege
                    X(i, :) = Xrb - E * abs(Xrb - X(i, :));
                elseif r < 0.5 && abs(E) >= 0.5
                    % Soft besiege with progressive rapid dives
                    LF = levyFlight(beta, dim); % Levy distribution (9)
                    Y = Xrb - E * abs(J * Xrb - X(i, :)); % Equation (7)
                    Z = Y + rand() * LF; % Equation (8)
                    Y = max(lb, min(ub, Y)); % Boundary
                    Z = max(lb, min(ub, Z)); % Boundary
                    % Evaluate fitness
                    fitY = TMFSMTSobjFun(Y > threshold, options);
					FEs = FEs + 1;
                    fitZ = TMFSMTSobjFun(Z > threshold, options);
					FEs = FEs + 1;
                    % Greedy selection
                    if fitY < fit(i), fit(i) = fitY; X(i, :) = Y; end
                    if fitZ < fit(i), fit(i) = fitZ; X(i, :) = Z; end
                else
                    % Hard besiege with progressive rapid dives
                    LF = levyFlight(beta, dim); % Levy distribution (9)
                    Y = Xrb - E * abs(J * Xrb - X_mu); % Equation (12)
                    Z = Y + rand() * LF; % Equation (13)
                    Y = max(lb, min(ub, Y)); % Boundary
                    Z = max(lb, min(ub, Z)); % Boundary
                    % Evaluate fitness
                    fitY = TMFSMTSobjFun(Y > threshold, options);
					FEs = FEs + 1;
                    fitZ = TMFSMTSobjFun(Z > threshold, options);
					FEs = FEs + 1;
                    % Greedy selection
                    if fitY < fit(i), fit(i) = fitY; X(i, :) = Y; end
                    if fitZ < fit(i), fit(i) = fitZ; X(i, :) = Z; end
                end
            end
            % Boundary control
            X(i, :) = max(lb, min(ub, X(i, :)));
        end

        % Update convergence curve
        Convergence_curve(t) = fitR;
	  aP=determinePerformance(Xrb,options.OutU,options.OutLabelsU, options.OutT,options.OutLabelsT,options.computeMetrics);
      fprintf('\n  Alg  %s Iteration %d Accuracy = %f  FBest %f',options.name, t, aP(1) ,Convergence_curve(t)  ) 
		  t = t + 1; % Increment iteration counter
    end

    % Feature selection
    Pos = 1:dim;
    Sf = Pos(Xrb > threshold); % Selected features
    sFeat = frame(:, Sf); % Selected feature subset

    % Store results
    result.fbest = fitR;  
    result.xbest = Xrb; % 
    result.fnumber  = length(Sf);  
    result.ccurve = Convergence_curve;  
	result.elapsedTime=toc;
    result.acc=determinePerformance(result.xbest,options.OutU,options.OutLabelsU, options.OutT,options.OutLabelsT,options.computeMetrics) ;
 
	
end

% Levy flight function
function LF = levyFlight(beta, dim)
    % Sigma for Levy flight
    nume = gamma(1 + beta) * sin(pi * beta / 2);
    deno = gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2);
    sigma = (nume / deno) ^ (1 / beta);

    % Random values u and v
    u = randn(1, dim) * sigma;
    v = randn(1, dim);
    step = u ./ abs(v) .^ (1 / beta);
    LF = 0.01 * step; % Final Levy flight values
end
