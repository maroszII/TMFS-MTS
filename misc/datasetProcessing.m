% Dataset selection and feature extraction
% Datasets are divided into training, validation, and testing subsets
% Feature selection is performed only on the training and validation subsets
% i.e., objective function TMFSMTSobjFun uses only them
% Testing subset is only used to report the performance of a selected solution 
% i.e., as result.acc of result.X or local print 
% Datasets:
% % 'FLORENCE' - https://www.micc.unifi.it/resources/datasets/florence-3d-actions-dataset/   
  % 'MSRA' - https://sites.google.com/view/wanqingli/data-sets/msr-action3d 
  % 'UTD' - https://personal.utdallas.edu/~kehtar/UTD-MHAD.html
  % 'UTK' - http://cvrc.ece.utexas.edu/KinectDatasets/HOJ3D.html
 

if selectDataset == 1  
    % Load MSRA datasets (Set I)
    load('features\DD_PPD_3_sets.mat');
    % Choose a dataset (Set I)
    features = mergedFeaturesAllPersonsSetI;
    % Dataset division into training, validation, and testing sets    
    trainingData = cat(2, features{1}, features{3}, features{5}, features{7});
    testingData = cat(2, features{2}, features{4}, features{6}, features{8}, features{10});
	validationData = cat(2, features{9});
	
end

if selectDataset== 2    
	% Load MSRA datasets (Set II)
    load('features\DD_PPD_3_sets.mat');
	features = mergedFeaturesAllPersonsSetII;  
    % Dataset division into training and testing sets     % Dataset division into training, validation, and testing sets     
	trainingData = cat(2, features{1}, features{3}, features{5}, features{7});
    testingData = cat(2, features{2}, features{4}, features{6}, features{8}, features{10});
	validationData = cat(2, features{9});
	
end

if selectDataset == 3
    % Load MSRA datasets (Set III)
    load('features\DD_PPD_3_sets.mat');
    % Choose a dataset (Set III in this case)
    features = mergedFeaturesAllPersonsSetIII; 
    % Dataset division into training, validation, and testing sets    
    trainingData = cat(2, features{1}, features{3}, features{5}, features{7});
    testingData = cat(2, features{2}, features{4}, features{6}, features{8}, features{10});
	validationData = cat(2, features{9});
end  

if selectDataset == 4 % UTD  
    % Load feature data from .mat files
	load('features\UTD1.mat');% Load the first part
	load('features\UTD2.mat');% Load the second part

	% Process features for each person

	% Person 1: Merge distance-based and point-pair features, assign labels
	for i = 1 : size(pointPairFeaturesPerson_1,1)
		mergedFeaturesPerson_1{i}.features = [distanceFeaturesPerson_1{i,1} pointPairFeaturesPerson_1{i,1}];
		mergedFeaturesPerson_1{i}.label = pointPairFeaturesPerson_1{i,2};
	end
	clear distanceFeaturesPerson_1 pointPairFeaturesPerson_1; % Free memory

	% Person 2: Merge features and assign labels
	for i = 1 : size(pointPairFeaturesPerson_2,1)
		mergedFeaturesPerson_2{i}.features = [distanceFeaturesPerson_2{i,1} pointPairFeaturesPerson_2{i,1}];
		mergedFeaturesPerson_2{i}.label = pointPairFeaturesPerson_2{i,2};
	end
	clear distanceFeaturesPerson_2 pointPairFeaturesPerson_2;

	% Person 3: Merge features and assign labels
	for i = 1 : size(pointPairFeaturesPerson_3,1)
		mergedFeaturesPerson_3{i}.features = [distanceFeaturesPerson_3{i,1} pointPairFeaturesPerson_3{i,1}];
		mergedFeaturesPerson_3{i}.label = pointPairFeaturesPerson_3{i,2};
	end
	clear distanceFeaturesPerson_3 pointPairFeaturesPerson_3;

	% Person 4: Merge features and assign labels
	for i = 1 : size(pointPairFeaturesPerson_4,1)
		mergedFeaturesPerson_4{i}.features = [distanceFeaturesPerson_4{i,1} pointPairFeaturesPerson_4{i,1}];
		mergedFeaturesPerson_4{i}.label = pointPairFeaturesPerson_4{i,2};
	end
	clear distanceFeaturesPerson_4 pointPairFeaturesPerson_4;

	% Person 5: Merge features and assign labels
	for i = 1 : size(pointPairFeaturesPerson_5,1)
		mergedFeaturesPerson_5{i}.features = [distanceFeaturesPerson_5{i,1} pointPairFeaturesPerson_5{i,1}];
		mergedFeaturesPerson_5{i}.label = pointPairFeaturesPerson_5{i,2};
	end
	clear distanceFeaturesPerson_5 pointPairFeaturesPerson_5;

	% Person 6: Merge features and assign labels
	for i = 1 : size(pointPairFeaturesPerson_6,1)
		mergedFeaturesPerson_6{i}.features = [distanceFeaturesPerson_6{i,1} pointPairFeaturesPerson_6{i,1}];
		mergedFeaturesPerson_6{i}.label = pointPairFeaturesPerson_6{i,2};
	end
	clear distanceFeaturesPerson_6 pointPairFeaturesPerson_6;

	% Person 7: Merge features and assign labels
	for i = 1 : size(pointPairFeaturesPerson_7,1)
		mergedFeaturesPerson_7{i}.features = [distanceFeaturesPerson_7{i,1} pointPairFeaturesPerson_7{i,1}];
		mergedFeaturesPerson_7{i}.label = pointPairFeaturesPerson_7{i,2};
	end
	clear distanceFeaturesPerson_7 pointPairFeaturesPerson_7;

	% Person 8: Merge features and assign labels
	for i = 1 : size(pointPairFeaturesPerson_8,1)
		mergedFeaturesPerson_8{i}.features = [distanceFeaturesPerson_8{i,1} pointPairFeaturesPerson_8{i,1}];
		mergedFeaturesPerson_8{i}.label = pointPairFeaturesPerson_8{i,2};
	end
	clear distanceFeaturesPerson_8 pointPairFeaturesPerson_8;

    % Concatenate data from different persons for training and testing
    trainingData = cat(2, mergedFeaturesPerson_1, mergedFeaturesPerson_3, mergedFeaturesPerson_5 );
    validationData = cat(2,  mergedFeaturesPerson_7);
    testingData = cat(2, mergedFeaturesPerson_2, mergedFeaturesPerson_4, mergedFeaturesPerson_6, mergedFeaturesPerson_8);

    % Remove unwanted features from data
    for i = 1:length(testingData)
        testingData{i}.features(:, [81, 78, 75, 72, 69, 66, 63, 60, 57, 54, 51, 48]) = [];
    end
	for i = 1:length(validationData)
        validationData{i}.features(:, [81, 78, 75, 72, 69, 66, 63, 60, 57, 54, 51, 48]) = [];
    end
    for i = 1:length(trainingData)
        trainingData{i}.features(:, [81, 78, 75, 72, 69, 66, 63, 60, 57, 54, 51, 48]) = [];
    end
end

if selectDataset == 5 % Florence dataset
   % Load feature data from .mat files
		load('features\Florence1.mat');% Load the first part
		load('features\Florence2.mat');% Load the second part

		% Process features for all persons
		for s = 1 : length(pointPairFeaturesAllPersons) % Loop over all subjects
			for i = 1 : length(pointPairFeaturesAllPersons{s}) % Loop over feature sets for each subject
				% Merge distance-based and point-pair features
				mergedFeaturesAllPersons{s}{i}.features = [distanceFeaturesAllPersons{s}{i,1} pointPairFeaturesAllPersons{s}{i,1}];
				% Assign the corresponding label
				mergedFeaturesAllPersons{s}{i}.label = pointPairFeaturesAllPersons{s}{i,2};
			end	
		end

		% Free memory by clearing unnecessary variables
		clear distanceFeaturesAllPersons pointPairFeaturesAllPersons;

      
        % Concatenate data from different persons for training and testing
        testingData = cat(2, mergedFeaturesAllPersons{1:4});
        validationData = cat(2, mergedFeaturesAllPersons{5});
        trainingData = cat(2, mergedFeaturesAllPersons{6:10});
        
        % Remove unwanted features from data
		for i = 1:length(validationData)
            validationData{i}.features(:, [69, 66, 63, 60, 57, 54, 51, 48]) = [];
        end
        for i = 1:length(testingData)
            testingData{i}.features(:, [69, 66, 63, 60, 57, 54, 51, 48]) = [];
        end
        for i = 1:length(trainingData)
            trainingData{i}.features(:, [69, 66, 63, 60, 57, 54, 51, 48]) = [];
        end
     
end

if selectDataset == 6 % UTK dataset
    % Load feature data from .mat files
	load('features\UTK1.mat'); % Load the first part
	load('features\UTK2.mat'); % Load the second part

	% Process features for all persons
	for s = 1 : length(pointPairFeaturesAllPersons) % Iterate over all subjects
		for i = 1 : length(pointPairFeaturesAllPersons{s}) % Iterate over feature sets for each subject
			% Merge distance-based features and point-pair features into a single structure
			mergedFeaturesAllPersons{s}{i}.features = [distanceFeaturesAllPersons{s}{i,1} pointPairFeaturesAllPersons{s}{i,1}];
			% Assign corresponding label to the merged feature set
			mergedFeaturesAllPersons{s}{i}.label = pointPairFeaturesAllPersons{s}{i,2};
		end	
	end

	% Free memory by clearing unnecessary variables
	clear distanceFeaturesAllPersons pointPairFeaturesAllPersons;
  
     % Concatenate data from different persons for training and testing
    testingData = cat(2, mergedFeaturesAllPersons{1:4});
    validationData = cat(2, mergedFeaturesAllPersons{5});
    trainingData = cat(2, mergedFeaturesAllPersons{6:10});
    
    % Remove unwanted features from data
	for i = 1:length(validationData)
        validationData{i}.features(:, [81, 78, 75, 72, 69, 66, 63, 60, 57, 54, 51, 48]) = [];
    end
    for i = 1:length(testingData)
        testingData{i}.features(:, [81, 78, 75, 72, 69, 66, 63, 60, 57, 54, 51, 48]) = [];
    end
    for i = 1:length(trainingData)
        trainingData{i}.features(:, [81, 78, 75, 72, 69, 66, 63, 60, 57, 54, 51, 48]) = [];
    end
end

%%% Prepare the training and testing data for further processing
frame = trainingData{1}.features; % Sample frame as metaheuristics do not work with temporal data while selecting features 
TRAIN_Y = [];
VALIDATION_Y = [];
TEST_Y = [];

% Organize training data
for i = 1:length(trainingData)
    TRAIN_X{i} = trainingData{i}.features';
    TRAIN_Y = [TRAIN_Y; str2num(trainingData{i}.label)];
end

% Organize testing data
for i = 1:length(testingData)
    TEST_X{i} = testingData{i}.features';
    TEST_Y = [TEST_Y; str2num(testingData{i}.label)];
end

% Organize validation data
for i = 1:length(validationData)
    VALIDATION_X{i} = validationData{i}.features';
    VALIDATION_Y = [VALIDATION_Y; str2num(validationData{i}.label)];
end

% Prepare outputs for model
OutT = TEST_X';
OutU = TRAIN_X';
OutV = VALIDATION_X';
OutLabelsU = TRAIN_Y;
OutLabelsT = TEST_Y;
OutLabelsV = VALIDATION_Y ;

% Store the processed data in options
options.OutU = OutU;
options.OutLabelsU = OutLabelsU;
options.OutT = OutT;
options.OutLabelsT = OutLabelsT;
options.OutV = OutV;
options.OutLabelsV = OutLabelsV;
% Note that it is assumed that values of feature vectors in time series are normalized



