%  A. Faramarzi, M. Heidarinejad, S. Mirjalili, A.H. Gandomi, 
%  Marine Predators Algorithm: A Nature-inspired Metaheuristic
%  Expert Systems with Applications
%  DOI: doi.org/10.1016/j.eswa.2020.113377

function result = TMFSMTSmpa(frame, options)
    % Parameters
	tic
    
    lb = 0;
    ub = 1;
 
    threshold = 0.5; ; %  
	
 % Parameters specific to MPA
    beta = 1.5; % Levy component
    P = 0.5; % Constant
    FADs = 0.2; % Fish aggregating devices effect

    if isfield(options, 'N'), N = options.N; end
    if isfield(options, 'max_Iter'), max_Iter = options.max_Iter; end
    if isfield(options, 'maxEvals'), maxEvals = options.maxEvals; end

    
    % Dimensions of the feature set
    dim = size(frame, 2);
    FEs = 0;
    %% Initialization
    X = lb + (ub - lb) .* rand(N, dim); % Initialize population
    fitness = inf * ones(N, 1); % Fitness values for population
    Best_fitness = inf; % Best fitness
    Best_position = zeros(1, dim); % Best solution position
    % Convergence_curve = zeros(1, max_Iter); % Convergence curve
    Convergence_curve = []; % Convergence curve

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

        % Adaptive parameter (Eq. 14)
        CF = (1 - (t / max_Iter)) ^ (2 * (t / max_Iter));
        X_new = zeros(N, dim); % Initialize updated positions

        % First phase
        if t <= max_Iter / 3
            for i = 1:N
                RB = randn(1, dim); % Brownian random number
                R = rand();
                stepsize = RB .* (Best_position - RB .* X(i, :));
                X_new(i, :) = X(i, :) + P * R .* stepsize; % Eq. (12)
            end

        % Second phase
        elseif t > max_Iter / 3 && t <= 2 * max_Iter / 3
            for i = 1:N
                if i <= N / 2
                    RL = 0.05 * levy(beta, dim); % Levy random number
                    R = rand();
                    stepsize = RL .* (Best_position - RL .* X(i, :));
                    X_new(i, :) = X(i, :) + P * R .* stepsize; % Eq. (13)
                else
                    RB = randn(1, dim); % Brownian random number
                    stepsize = RB .* (RB .* Best_position - X(i, :));
                    X_new(i, :) = Best_position + P * CF .* stepsize; % Eq. (14)
                end
            end

        % Third phase
        else
            for i = 1:N
                RL = 0.05 * levy(beta, dim); % Levy random number
                stepsize = RL .* (RL .* Best_position - X(i, :));
                X_new(i, :) = Best_position + P * CF .* stepsize; % Eq. (15)
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

        % Eddy formation and FADs effect
        if rand() <= FADs
            for i = 1:N
                U = rand(1, dim) < FADs; % Uniform random number
                R = rand();
                X(i, :) = X(i, :) + CF .* (lb + R .* (ub - lb)) .* U; % Eq. (16)
            end
        end

        
        Convergence_curve(t) = Best_fitness;
		aP=determinePerformance(Best_position,options.OutU,options.OutLabelsU, options.OutT,options.OutLabelsT,options.computeMetrics);
		 fprintf('\n  Alg  %s Iteration %d Accuracy = %f  FBest %f',options.name, t,  aP(1) ,Convergence_curve(t)   ) 
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

%% Levy Distribution
function LF = levy(beta, dim)
    num = gamma(1 + beta) * sin(pi * beta / 2);
    deno = gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2);
    sigma = (num / deno) ^ (1 / beta);
    u = random('Normal', 0, sigma, 1, dim);
    v = random('Normal', 0, 1, 1, dim);
    LF = u ./ (abs(v) .^ (1 / beta));
end


% Copyright (c) 2020, Afshin Faramarzi, Seyedali Mirjalili
% All rights reserved.

% Redistribution and use in source, with or without
% modification, are permitted considering the conditions below:

    % * Redistributions of source code must retain the above copyright
      % notice, this list of conditions and the following disclaimer.

% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
