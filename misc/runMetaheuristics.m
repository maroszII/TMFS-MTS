% Initialize result storage
Results = struct('fbest', [], 'xbest', [], 'fnumber', [], 'ccurve', [], 'elapsedTime', [], 'acc', []);
 
% Main loop through methods and iterations
Num_Methods=10;
 for i = 1:Num_Methods % 10 methods 
    for j = 1:Num_Runs
        fprintf('\nRun %d for method %d...', j, i); 
        
        % Switch for selecting the feature selection method
        switch i
            case 1
                options.name = 'AROA';
                result = TMFSMTSaroa(frame, options);
            case 2
                options.name = 'RIME';
                result = TMFSMTSrime(frame, options);
            case 3
                options.name = 'PLO';
                result = TMFSMTSplo(frame, options);
            case 4
                options.name = 'COA';
                result = TMFSMTScoa(frame, options);
            case 5
                options.name = 'MPA';
                result = TMFSMTSmpa(frame, options);
            case 6
                options.name = 'EO';
                result = TMFSMTSeo(frame, options);
            case 7
                options.name = 'GNDO';
                result = TMFSMTSgndo(frame, options);
            case 8
                options.name = 'SCA';
                result = TMFSMTSsca(frame, options);
            case 9
                options.name = 'HHO';
                result = TMFSMTShho(frame, options);
            case 10
                options.name = 'AO';
                result = TMFSMTSao(frame, options);
        end

        % Store results
        Results(i, j).fbest = result.fbest;
        Results(i, j).xbest = result.xbest;
        Results(i, j).fnumber = result.fnumber;
        Results(i, j).ccurve = result.ccurve;
        Results(i, j).elapsedTime = result.elapsedTime;
        Results(i, j).acc = result.acc; 
    end
	
	   
end
 
 % Generate timestamped filename
        timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
        filename = sprintf('%s-TMFS-MTS.mat', timestamp );
        
 % Save results
        save(filename );
 	
