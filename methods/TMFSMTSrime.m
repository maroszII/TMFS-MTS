% Hang Su, Dong Zhao, Ali Asghar Heidari, Lei Liu, Xiaoqin Zhang, Majdi Mafarja, Huiling Chen  
% RIME: A physics-based optimization
% Neurocomputing,ELSEVIER- 2023 
function result = TMFSMTSrime(frame, options)
tic
 
lb = 0;
ub = 1;
threshold = 0.5;
 
 
if isfield(options,'N'), N = options.N; end% 
if isfield(options,'max_Iter'), max_Iter = options.max_Iter; end% 
if isfield(options,'maxEvals'), maxEvals = options.maxEvals; end% 
Max_iter= max_Iter;  %% 

 
dim = size(frame,2); 
% Initial 
Rimepop   = zeros(N,dim); 
for i = 1:N
  for d = 1:dim
    Rimepop(i,d) = lb + (ub - lb) * rand();
  end
end  

FEs = 0; % Function evaluations counter


% initialize position
Best_rime=zeros(1,dim);
Best_rime_rate=inf;%change this to -inf for maximization problems
Lb=lb.*ones(1,dim);% lower boundary 
Ub=ub.*ones(1,dim);% upper boundary
% Convergence_curve=zeros(1,Max_iter);
Convergence_curve=[];
Rime_rates=zeros(1,N);
newRime_rates=zeros(1,N);
W = 5;


for i=1:N
    
    Rime_rates(1,i)=TMFSMTSobjFun(Rimepop(i,:) > threshold,options);
    FEs = FEs + 1;
    if Rime_rates(1,i)<Best_rime_rate
        Best_rime_rate=Rime_rates(1,i);
        Best_rime=Rimepop(i,:);
    end
end
Convergence_curve(1) = Best_rime_rate;
newRime_rates = Rime_rates;
it = 2;
% Main loop
while it <= Max_iter
	if FEs >= maxEvals
        break;
    end
     %Parameters of Eq.(3),(4),(5)
    r1 = (rand-0.5)*2; %[-1,1]
    Sita = (pi*it/(Max_iter/10));
    Beta = (1-round(it*W/Max_iter)/W);
    E =sqrt(it/Max_iter);%Eq.(6)
    RimeFactor = r1 * cos(Sita) * Beta;    
    newRimepop = Rimepop;%Recording new populations
    normalized_rime_rates=normr(Rime_rates);%Parameters of Eq.(7)
    for i=1:N
        for j=1:dim
            
            r1=rand();
            if r1< E
                newRimepop(i,j)=Best_rime(1,j)+RimeFactor*((Ub(j)-Lb(j))*rand+Lb(j));%Eq.(3)
            end
            
            r2=rand();
            if r2<normalized_rime_rates(i)
                newRimepop(i,j)=Best_rime(1,j);%Eq.(7)
            end
        end
    end
    for i=1:N
      
        Flag4ub=newRimepop(i,:)>ub;
        Flag4lb=newRimepop(i,:)<lb;
        newRimepop(i,:)=(newRimepop(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
      
        newRime_rates(1,i)= TMFSMTSobjFun(newRimepop(i,:)> threshold,options);
        FEs = FEs + 1;
        if newRime_rates(1,i)<Rime_rates(1,i)
            Rime_rates(1,i) = newRime_rates(1,i);
            Rimepop(i,:) = newRimepop(i,:);
            if newRime_rates(1,i)< Best_rime_rate
               Best_rime_rate=Rime_rates(1,i);
               Best_rime=Rimepop(i,:);
            end
        end
    end

    Convergence_curve(it)=Best_rime_rate;
    aP=determinePerformance(Best_rime,options.OutU,options.OutLabelsU, options.OutT,options.OutLabelsT,options.computeMetrics);
	fprintf('\n  Alg  %s Iteration %d Accuracy = %f  FBest %f',options.name, it,aP(1),Convergence_curve(it)   )  
    it=it+1;
end
 

  result.fbest=Best_rime_rate;
  result.xbest=Best_rime;
  result.fnumber= sum((result.xbest > threshold) == 1);
  result.ccurve= Convergence_curve;
  result.elapsedTime=toc;
  result.acc=determinePerformance(result.xbest,options.OutU,options.OutLabelsU, options.OutT,options.OutLabelsT,options.computeMetrics) ;
 
 
end



