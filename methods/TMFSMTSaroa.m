 %%Karol Cymerys, Mariusz Oszust, Attraction–Repulsion Optimization Algorithm for Global Optimization Problems, Swarm and Evolutionary Computation, Volume 84, 2024, 101459, ISSN 2210-6502, https://doi.org/10.1016/j.swevo.2023.101459.. 

function result = TMFSMTSaroa(frame,options)

 tic %for measuring runtime
 
 % Algorithm parameters definition
    c = 0.95;
    fr1 = 0.15;
    fr2 = 0.6;
    p1 = 0.2;  
    p2 = 0.8; 
    Ef = 0.4;
    tr1 = 0.9;
    tr2 = 0.85; 
    tr3 = 0.9;
    % Algorithm parameters definition
	
 
 
if isfield(options,'N'), N = options.N; end
if isfield(options,'max_Iter'), max_Iter = options.max_Iter; end
if isfield(options,'maxEvals'), maxEvals = options.maxEvals; end
 
 
 % Parameters
 lb    = 0;
 ub    = 1; 
 threshold = 0.5;  

dim = size(frame,2); 
% Initial (9)


    
    tmax = ceil((maxEvals - N)/(2*N));
    evalCounter = 0;

    Convergence_curve = zeros(1,tmax);
    Xmin = repmat(ones(1,dim).*lb,N,1);
    Xmax = repmat(ones(1,dim).*ub,N,1);

 

    % random initialization - Eq (3)
    X = rand(N,dim).*(ub-lb) + lb;
    [X, F, evalCounter] = evaluate_population(X,   ub, lb, evalCounter, maxEvals,  options, threshold);
    [fbest, ibest] = min(F);
    xbest = X(ibest,:);
    % random initialization - Eq (3)

    X_memory = X;
    F_memory = F;

    % Main loop
    for t=1:tmax	
        D = squareform(pdist(X, 'squaredeuclidean'));  % Eq (4) 
        m = tanh(t, tmax, [-2, 7]);   % Eq (11)    

        for i=1:N
           Dimax = max(D(i,:));
           k = floor((1-t/tmax)*N)+1;  % Eq (9)
           [~, neighbors] = sort(D(i,:));
           
           % Attraction-Repulsion operator % Eq (6)
           delta_ni = zeros(1,dim);
           for j=neighbors(1:k)
                I = 1 - (D(i,j)/Dimax);  % Eq (7)
                s = sign(F(j)-F(i));  % Eq (8)
                delta_ni = delta_ni + c*(X_memory(i,:)-X_memory(j,:))*I*s;
           end
           ni = delta_ni/N;
           % Attraction-Repulsion operator % Eq (6)

           % Attraction to best solusion Eq (10)
           if rand < p1
               bi = m*c.*(rand(1,dim).*xbest - X_memory(i,:));
           else
               bi = m*c.*(xbest - X_memory(i,:));
           end
           % Attraction to best solusion Eq (10)

           % Local search operators Eq (15)
           if rand < p2
               if rand > 0.5*t/tmax + 0.25
                   u1 = rand(1, dim) > tr1;
                   ri = u1.*random('Normal', zeros(1,dim), fr1*(1-t/tmax)*(ub-lb));  % Eq (12)
               else
                   % Eq (13)
                   u2 = rand(1,dim) > tr2;
                   w = index_roulette_wheel_selection(F, k);
                   Xw = X_memory(w,:);
                   if rand < 0.5
                       ri = fr2*u2.*(1-t/tmax).*sin(2*pi*rand(1,dim)).*abs(rand(1,dim).*Xw-X_memory(i,:));
                   else
                       ri = fr2*u2.*(1-t/tmax).*cos(2*pi*rand(1,dim)).*abs(rand(1,dim).*Xw-X_memory(i,:));
                   end
                   % Eq (13)
               end
           else
               u3 = rand(1,dim) > tr3;
               ri = u3.*(2*rand(1,dim)-ones(1,dim)) .* (ub-lb);  % Eq (14)
           end
           % Local search operators Eq (15)

           X(i,:) = X(i,:) + ni + bi + ri;  % Eq(16)
		   
        end
        
        [X, F, evalCounter] =evaluate_population(X,   ub, lb, evalCounter, maxEvals,  options, threshold);
        [fbest_candidate, ibest_candidate] = min(F);

        if fbest_candidate < fbest
            fbest = fbest_candidate;
            xbest = X(ibest_candidate, :);
        end

	    [X, F] = memory_operator(X, F, X_memory, F_memory);  % Eq (18)
        X_memory = X;
        F_memory = F; 
			
        % Eq (17)			 
        CF=(1-t/tmax)^3 ;			 
        if rand < Ef
            u4 = rand(N,dim) < Ef;                                                                                              
            X = X + CF*(u4.*(rand(N,dim).*(Xmax-Xmin) + Xmin));
        else
            r7 = rand();		
            X = X + (CF*(1-r7) + r7)*(X(randperm(N),:) - X(randperm(N),:));
        end
		% Eq (17)

        [X, F, evalCounter] = evaluate_population(X,   ub, lb, evalCounter, maxEvals,  options, threshold);
        [fbest_candidate, ibest_candidate] = min(F);

        if fbest_candidate < fbest
            fbest = fbest_candidate;
            xbest = X(ibest_candidate, :);
        end
	 
        [X, F] = memory_operator(X, F, X_memory, F_memory);  % Eq (18)
        X_memory = X;
        F_memory = F; 

  	    Convergence_curve(t) = fbest;
		aP=determinePerformance(xbest,options.OutU,options.OutLabelsU, options.OutT,options.OutLabelsT,options.computeMetrics);
			 fprintf('\n  Alg  %s Iteration %d Accuracy = %f  FBest %f',options.name,  t, aP(1),Convergence_curve( t)   )  
    end	

 
  result.fbest=fbest;
  result.xbest=xbest;
  result.fnumber= sum((xbest > threshold) == 1);
  result.ccurve= Convergence_curve;
  result.elapsedTime=toc;
  result.acc=determinePerformance(result.xbest,options.OutU,options.OutLabelsU, options.OutT,options.OutLabelsT,options.computeMetrics) ;
 
 

	
end

function [X, F, evalCounter] = evaluate_population(X,   ub, lb, evalCounter, maxEvals,  options, threshold)
    N = size(X,1);
    F = Inf(N,1);
    X = max(lb, min(ub, X)); % Check space bounds    
	
	 
	
    for i=1:N
        if evalCounter >= maxEvals
            break
        end
       
	        F(i) =TMFSMTSobjFun(X(i,:) > threshold,options);
		
        evalCounter = evalCounter + 1;
    end
end

function [X, F] = memory_operator(X, F, X_memory, F_memory)
    dim = size(X, 2);
    Inx = F_memory < F;
    Indx = repmat(Inx,1,dim);
    X = Indx.*X_memory + ~Indx.*X;
    F = Inx.*F_memory + ~Inx.*F;
end

function [y] = tanh(t, tmax, range)
    z = 2*(t/tmax*(range(2)-range(1)) + range(1));
    y = 0.5*((exp(z)-1)/(exp(z)+1) + 1);
end


function [selected_index] = index_roulette_wheel_selection(F, k)
    fitness = F(1:k);
    weights = max(fitness) - fitness;
    weights = cumsum(weights/sum(weights));
    
    selected_index = roulette_wheel_selection(weights);
end

function [selected_index] = roulette_wheel_selection(weights)
    r = rand();
    selected_index = 1;
    for index=size(weights,1)
        if r <= weights(index)
            selected_index = index;
            break;
        end
    end
end

 