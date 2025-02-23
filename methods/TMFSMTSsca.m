
                             
%  S. Mirjalili, SCA: A Sine Cosine Algorithm for solving optimization problems
%  Knowledge-Based Systems, DOI: http://dx.doi.org/10.1016/j.knosys.2015.12.022
%_______________________________________________________________________________________________
%

function result = TMFSMTSsca(frame, options)
   
tic
 % Parameters   
alpha = 2; % Constant
 
lb = 0;
ub = 1;
threshold = 0.5;
 
 
if isfield(options,'N'), N = options.N; end% 
if isfield(options,'max_Iter'), max_Iter = options.max_Iter; end% 
if isfield(options,'maxEvals'), maxEvals = options.maxEvals; end% 
  

 

     
FEs = 0;
    % Number of dimensions
    dim = size(frame, 2);

    % Initialization of population
    X = lb + (ub - lb) .* rand(N, dim); % Initialize positions
    fitD = inf; % Best fitness
    fit = inf(1, N); % Fitness values for population

    % Preallocate convergence curve
    % Convergence_curve = zeros(1, max_Iter);
    Convergence_curve = [];

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
            % Update destination (best solution so far)
            if fit(i) < fitD
                fitD = fit(i); % Update best fitness
                Xdb = X(i, :); % Update best position
            end
        end

        % Update positions
        r1 = alpha - t * (alpha / max_Iter); % r1 decreases linearly (Eq. 3.4)
        for i = 1:N
            for d = 1:dim
                % Random parameters
                r2 = (2 * pi) * rand();
                r3 = 2 * rand();
                r4 = rand();
                % Position update based on sine or cosine rule
                if r4 < 0.5
                    X(i, d) = X(i, d) + r1 * sin(r2) * abs(r3 * Xdb(d) - X(i, d)); % Sine update (Eq. 3.1)
                else
                    X(i, d) = X(i, d) + r1 * cos(r2) * abs(r3 * Xdb(d) - X(i, d)); % Cosine update (Eq. 3.2)
                end
            end
            % Boundary control
            X(i, :) = max(lb, min(ub, X(i, :)));
        end

        % Update convergence curve
        Convergence_curve(t) = fitD;
		aP=determinePerformance(Xdb,options.OutU,options.OutLabelsU, options.OutT,options.OutLabelsT,options.computeMetrics);
		fprintf('\n  Alg  %s Iteration %d Accuracy = %f  FBest %f',options.name, t, aP(1),Convergence_curve(t)  ) 
        t = t + 1; % Increment iteration counter
    end

    % Feature selection
    Pos = 1:dim;
    Sf = Pos((Xdb > threshold) == 1); % Selected features
     

    % Store results
    result.fbest = fitD; % Best fitness
    result.xbest = Xdb; % Best solution
 
    result.fnumber = length(Sf); % Number of selected features
    result.ccurve  = Convergence_curve; % Convergence curve
    result.elapsedTime=toc;
    result.acc=determinePerformance(result.xbest,options.OutU,options.OutLabelsU, options.OutT,options.OutLabelsT,options.computeMetrics) ;
 
end
