%  A. Faramarzi, M. Heidarinejad, B. Stephens, S. Mirjalili, 
%  Equilibrium optimizer: A novel optimization algorithm
%  Knowledge-Based Systems
%  DOI: https://doi.org/10.1016/j.knosys.2019.105190

function result = TMFSMTSeo(frame, options)
 tic
 
lb = 0;
ub = 1;
 
threshold= 0.5;%%% 
 
if isfield(options,'N'), N = options.N; end%% 
if isfield(options,'max_Iter'), max_Iter = options.max_Iter; end%% 
if isfield(options,'maxEvals'), maxEvals = options.maxEvals; end%% 
 

    % EO-specific parameters
    a1 = 2; % constant
    a2 = 1; % constant
    GP = 0.5; % generation probability
    V = 1; % unit

  % Number of dimensions
dim = size(frame,2); 
% Initial (6)
X   = zeros(N,dim); 
for i = 1:N
	for d = 1:dim
    X(i,d) = lb + (ub - lb) * rand(); 
  end
end
% Set memory
Xmb  = zeros(N,dim); 
fitM = ones(1,N);
% Pre
fitE1 = inf;
fitE2 = inf; 
fitE3 = inf;
fitE4 = inf; 
Xeq1  = zeros(1,dim);
Xeq2  = zeros(1,dim); 
Xeq3  = zeros(1,dim);
Xeq4  = zeros(1,dim);
Xave  = zeros(1,dim);
fit   = zeros(1,N); 

% curve = inf;
curve = [];
t = 1;
FEs =0;
% Iteration
while t <= max_Iter
	if FEs >= maxEvals
        break;
    end
  % Fitness
  for i = 1:N
    fit(i) =TMFSMTSobjFun(X(i, :) > threshold, options);
	FEs = FEs + 1;
    % Equilibrium update
    if fit(i) < fitE1
      fitE1 = fit(i); 
      Xeq1  = X(i,:);
    elseif fit(i) > fitE1 && fit(i) < fitE2
      fitE2 = fit(i);
      Xeq2  = X(i,:);
    elseif fit(i) > fitE1 && fit(i) > fitE2 && fit(i) < fitE3
      fitE3 = fit(i);
      Xeq3  = X(i,:);
    elseif fit(i) > fitE1 && fit(i) > fitE2 && fit(i) > fitE3 && ...
        fit(i) < fitE4
      fitE4 = fit(i);
      Xeq4  = X(i,:);
    end
  end
  % Memory update
  for i = 1:N
    if fitM(i) < fit(i)
      fit(i) = fitM(i);
      X(i,:) = Xmb(i,:);
    end
  end
  % Store memory
  Xmb  = X; 
  fitM = fit; 
  % Compute average candidate 
  for d = 1:dim
    Xave(d) = (Xeq1(d) + Xeq2(d) + Xeq3(d) + Xeq4(d)) / 4;
  end
  % Make an equilibrium pool (7)
  Xpool = [Xeq1; Xeq2; Xeq3; Xeq4; Xave];
  % Compute function tt (9)
  T     = (1 - (t / max_Iter)) ^ (a2 * (t / max_Iter));
  % Update
  for i = 1:N
    % Generation rate control parameter (15)
    r1 = rand(); 
    r2 = rand();
    if r2 >= GP
      GCP = 0.5 * r1;
    else
      GCP = 0;
    end
    % Random one solution from Xpool
    eq = randi([1,5]);
    for d = 1:dim
      % Random in [0,1]
      r = rand();
      % Lambda in [0,1]
      lambda = rand();
      % Substitution (11)
      F  = a1 * sign(r - 0.5) * (exp(-lambda * T) - 1);
      % Compute G0 (14)
      G0 = GCP * (Xpool(eq,d) - lambda * X(i,d));
      % Compute G (13)
      G  = G0 * F;
      % Update (16)
      X(i,d) = Xpool(eq,d) + (X(i,d) - Xpool(eq,d)) * F + ...
        (G / (lambda * V)) * (1 - F);
    end
    % Boundary
    XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    X(i,:) = XB;
  end 
  curve(t) = fitE1;
  aP =determinePerformance(Xeq1,options.OutU,options.OutLabelsU, options.OutT,options.OutLabelsT,options.computeMetrics);
  
         fprintf('\n  Alg  %s Iteration %d Accuracy = %f  FBest %f',options.name, t, aP(1),curve(t)   ) 
  t = t + 1;
end

    %% Result Struct
    result.fbest = fitE1; % Best fitness
    result.xbest = Xeq1; % Best solution
    result.fnumber = sum((Xeq1 > threshold) == 1); % Number of selected features
    result.ccurve = curve; % Convergence curve
	result.elapsedTime=toc;
    result.acc=determinePerformance(result.xbest,options.OutU,options.OutLabelsU, options.OutT,options.OutLabelsT,options.computeMetrics) ;
 
end
