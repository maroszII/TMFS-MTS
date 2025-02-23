% Yiying Zhang, Zhigang Jin, Seyedali Mirjalili,
% Generalized normal distribution optimization and its applications in parameter extraction of photovoltaic models,
% Energy Conversion and Management,
% Volume 224,
% 2020,
% 113301,
% ISSN 0196-8904,
% https://doi.org/10.1016/j.enconman.2020.113301.


function result = TMFSMTSgndo(frame, options)
    % Parameters
 tic
  
lb = 0;
ub = 1;
 
threshold=0.5;% 
 
if isfield(options,'N'), N = options.N; end 
if isfield(options,'max_Iter'), max_Iter = options.max_Iter; end% 
if isfield(options,'maxEvals'), maxEvals = options.maxEvals; end% 
 

 if N<4
	 disp('GNDO N<4 ->  p1 = RN(1); p2 = RN(2); p3 = RN(3); and other problems; N is changed to 4') %marosz
	 N=4;
 end

    % Dimensions of the feature set
    dim = size(frame, 2);

    %% Initialization
    X = lb + (ub - lb) .* rand(N, dim); % Initialize population
    fitness = inf * ones(N, 1); % Fitness values for population
    Best_fitness = inf; % Best fitness
    Best_position = zeros(1, dim); % Best solution position
    % Convergence_curve = zeros(1, max_Iter); % Convergence curve
    Convergence_curve = []; % Convergence curve
	FEs = 0;
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
    V = zeros(N, dim); % Temporary population for updates
    t = 2;
    while t <= max_Iter
		if FEs >= maxEvals
			break;
		end
        % Compute mean position (Eq. 22)
        M = mean(X, 1);

        for i = 1:N
            alpha = rand();

            % Local exploitation (Eq. 18-21)
            if alpha > 0.5
                a = rand();
                b = rand();
                for d = 1:dim
                    mu = (1/3) * (X(i, d) + Best_position(d) + M(d)); % Eq. 19
                    delta = sqrt((1/3) * ((X(i, d) - mu)^2 + ...
                          (Best_position(d) - mu)^2 + (M(d) - mu)^2)); % Eq. 20

                    lambda1 = rand();
                    lambda2 = rand();
                    if a <= b
                        eta = sqrt(-log(lambda1)) * cos(2 * pi * lambda2);
                    else
                        eta = sqrt(-log(lambda1)) * cos(2 * pi * lambda2 + pi);
                    end
                    V(i, d) = mu + delta * eta; % Eq. 18
                end

            % Global exploitation (Eq. 23-25)
            else
                RN = randperm(N); RN(RN == i) = [];
                p1 = RN(1); p2 = RN(2); p3 = RN(3);

                beta = rand();
                lambda3 = randn();
                lambda4 = randn();

                if fitness(i) < fitness(p1)
                    v1 = X(i, :) - X(p1, :);
                else
                    v1 = X(p1, :) - X(i, :);
                end

                if fitness(p2) < fitness(p3)
                    v2 = X(p2, :) - X(p3, :);
                else
                    v2 = X(p3, :) - X(p2, :);
                end

                for d = 1:dim
                    V(i, d) = X(i, d) + beta * abs(lambda3) * v1(d) + ...
                              (1 - beta) * abs(lambda4) * v2(d); % Eq. 23
                end
            end

            % Boundary control
            V(i, :) = max(lb, min(ub, V(i, :)));
        end

        % Evaluate new population
        for i = 1:N
            new_fitness = TMFSMTSobjFun(V(i, :) > threshold, options);
			FEs = FEs + 1;
            if new_fitness < fitness(i)
                fitness(i) = new_fitness;
                X(i, :) = V(i, :);
                if new_fitness < Best_fitness
                    Best_fitness = new_fitness;
                    Best_position = V(i, :);
                end
            end
        end

        % Update convergence curve
        Convergence_curve(t) = Best_fitness;
		 aP = determinePerformance(Best_position,options.OutU,options.OutLabelsU, options.OutT,options.OutLabelsT,options.computeMetrics);
        fprintf('\n  Alg  %s Iteration %d Accuracy = %f  FBest %f',options.name, t,  aP(1),Convergence_curve(t)  ) 
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

% Copyright (c) 2023, Heming Jia,Raohonghua,Wen changsheng,Seyedali Mirjalili
% All rights reserved.

% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:

% * Redistributions of source code must retain the above copyright notice, this
  % list of conditions and the following disclaimer.

% * Redistributions in binary form must reproduce the above copyright notice,
  % this list of conditions and the following disclaimer in the documentation
  % and/or other materials provided with the distribution

% * Neither the name of contributor's University nor the names of its
  % contributors may be used to endorse or promote products derived from this
  % software without specific prior written permission.

% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
% FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
% DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
% SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
% OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

