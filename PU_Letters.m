runCode = true;
% load('CurrentSeed.mat');
% rng(seedVal);
kernelType = 'Gaussian';
sigmaVal = 5;
name = 'Letters';
allLabels = 'A':'Z';
numLabels = length(allLabels);
setDiffStopTh = 0.00000000001;
% Level of sparsity
m = 300;
labelPosToRun = 1:numLabels;
numDiffLablesToRun = length(labelPosToRun);
propOfCurrLabelInP = [50];
propOfOtherLabelsInU = 100;
propOfTestData = 20;
%% Read data
[complete_data, complete_labels] = readUCILetters_2;
train = complete_data(:,1:16000);
labels_train = complete_labels(1:16000);
test = complete_data(:,16001:end);
labels_test = complete_labels(16001:end);
test_size = numel(labels_test);
train_size = numel(labels_train);

%%
numSamples = numel(labels_train);
%%
plotFigure = true; 
saveOutput = true;
%%
outputFileName = 'PU';
outputFileName = strcat(outputFileName,'_',name,'_m',num2str(m),'_I',num2str(numSamples),'_K_',kernelType);
if(strcmp(kernelType,'Gaussian'))
    outputFileName = strcat(outputFileName,'_sigma',num2str(sigmaVal));
end
fprintf('Output file: %s\n',outputFileName);
%%
if(runCode)
    numIndividualLabels = zeros(numLabels,1);
    individualLabelData = cell(numLabels,1);
    for labelPos = 1:numLabels
        labelName = allLabels(labelPos);
        dataPos = labels_train==labelName;
        numIndividualLabels(labelPos) = sum(dataPos);
        individualLabelData{labelPos} = train(:,dataPos);
    end
    %%
    TypesOfClassication = {'UseLabelledInfo','SelectPrototypes','SetAllUtoClassNeg'};
    accuracyPercentage = zeros(numDiffLablesToRun,numel(TypesOfClassication));
    selectedPrototypeIndices = zeros(numDiffLablesToRun,m);
    protoLabels = zeros(numDiffLablesToRun,m);
    setSizes = zeros(numDiffLablesToRun,1);
    %%
    for labelCount = 1:numDiffLablesToRun
        labelPos = labelPosToRun(labelCount);
        labelName = allLabels(labelPos);
        propValue = propOfCurrLabelInP(1);
        P_size = floor(numIndividualLabels(labelPos)*propValue/100);
        sampleNum = randperm(numIndividualLabels(labelPos));
        P_data= individualLabelData{labelPos}(:,sampleNum(1:P_size));
        U_data = individualLabelData{labelPos}(:,sampleNum(P_size+1:end));
        U_size = numIndividualLabels(labelPos)-P_size;
        true_labels_U = repmat(allLabels(labelPos),1,U_size);
        locOfPinU = 1:U_size;
        for otherLabels = 1:numLabels
            if(otherLabels~=labelPos)
                numOtherLabelSamples = ceil((propOfOtherLabelsInU/100)*numIndividualLabels(otherLabels));
                sampleNum = randperm(numIndividualLabels(otherLabels));
                U_data = horzcat(U_data,individualLabelData{otherLabels}(:,sampleNum(1:numOtherLabelSamples)));
                true_labels_U = horzcat(true_labels_U,repmat(allLabels(otherLabels),1,numOtherLabelSamples));
                U_size = U_size + numOtherLabelSamples;
            end
        end
        fprintf('Size of P data = %d\n',P_size);
        fprintf('Size of U data = %d\n',U_size);
        %PUData(labelCount).P = P_data;
        %PUData(labelCount).U = U_data;
        %% Determine weights for the data in P class
        distP = pdist2(P_data',P_data');
        KP = exp(-distP.^2/(2*sigmaVal^2));
        meanInnerProductP = sum(KP,2)/P_size;
        fprintf('Computing weights for the P class\n');
        [P_weights,~] = runOptimiser(KP,meanInnerProductP,[],zeros(P_size,1));
        %% Select prototypes from U class that best represents P
        fprintf('Computing meanInnerProductP...\n');
        meanInnerProductP = computeMeanInnerProductX(P_data,U_data,kernelType,sigmaVal,'faster');
        fprintf('Running ProtoDash for across prototype selection...\n');
        [proto_weights,STemp,setValue] = ProtoDash_Withepsilon(P_data,U_data,m,setDiffStopTh,kernelType,sigmaVal,meanInnerProductP);
        [~,IX] = sort(proto_weights,'descend');
        S = STemp(IX);
        setSizes(labelCount) = min(m,numel(S));
        protoLabels(labelCount,1:setSizes(labelCount)) = true_labels_U(S(1:setSizes(labelCount)));
        selectedPrototypeIndices(labelCount,1:setSizes(labelCount)) = S(1:setSizes(labelCount));
        %% Perform different types of classification
        for classType = 1:numel(TypesOfClassication)
            switch TypesOfClassication{classType}
                case 'UseLabelledInfo'
                    class_1_data  = horzcat(P_data,U_data(:,locOfPinU));
                    class_1_size = size(class_1_data,2);
                    remainingElements = setdiff(1:U_size,locOfPinU);
                    class_0_data = U_data(:,remainingElements);
                    class_0_size = size(class_0_data,2);
                    class_1_weights = (1/(class_1_size+class_0_size))*ones(1,class_1_size);
                    class_0_weights = (1/(class_1_size+class_0_size))*ones(1,class_0_size);
                case 'SelectPrototypes'
                    class_1_data  = horzcat(P_data,U_data(:,S));
                    class_1_weights = horzcat(P_weights(:)',proto_weights(:)');
                    remainingElements = setdiff(1:U_size,S);
                    class_0_data = U_data(:,remainingElements);
                    class_0_size = size(class_0_data,2);
                    class_0_weights = median(class_1_weights)*ones(1,class_0_size);
                case 'SetAllUtoClassNeg'
                    class_1_data  = P_data;
                    class_1_size = size(class_1_data,2);
                    class_0_data = U_data;
                    class_0_size = size(class_0_data,2);
                    class_1_weights = (1/(class_1_size+class_0_size))*ones(1,class_1_size);
                    class_0_weights = (1/(class_1_size+class_0_size))*ones(1,class_0_size);
            end
            class_1_labels = ones(1,numel(class_1_weights));
            class_0_labels = -1*ones(1,numel(class_0_weights));
            %% Train an SVM classifier
            all_data = horzcat(class_0_data,class_1_data)';
            all_weights = horzcat(class_0_weights,class_1_weights)';
            all_labels = horzcat(class_0_labels,class_1_labels)';
            fprintf('Training a linear SVM classifier for classification type: %s\n',TypesOfClassication{classType});
            SVMModel = fitcsvm(all_data,all_labels,'KernelFunction','linear','KernelScale','auto',...
            'Standardize',true,'ClassNames',[-1,1],'Weights',all_weights);
            %% Predict the test data using SVM classifier
            fprintf('Evaluating the performance of the classifier for classification type: %s\n',TypesOfClassication{classType});
            ground_truth = 2*int8(labels_test==labelName)-1;
            ground_truth = ground_truth(:);
            [predicted_labels,score] = predict(SVMModel,test');
            %CP = classperf(ground_truth, predicted_labels);
            accuracy = sum(ground_truth == predicted_labels) / numel(ground_truth);
            accPercent = 100*accuracy;
            accuracyPercentage(labelCount,classType) = accPercent;
            fprintf('Accuracy for classification type: %s for label %d = %f (%%) \n',TypesOfClassication{classType},allLabels(labelPos),accPercent);
        end   
        if(saveOutput)
            deleteFileName = strcat('Variables_',outputFileName,'.mat');
            delete(deleteFileName);
            save(strcat('Variables_',outputFileName),'accuracyPercentage','selectedPrototypeIndices',...
                'protoLabels','setSizes');
        end
    end
else
    load(strcat('Variables_',outputFileName));
end
if(plotFigure)
    averageAcc = mean(accuracyPercentage,1);
    fprintf('Accuracy when using labelled Information = %f\n',averageAcc(1));
    fprintf('Accuracy after selecting  prototypes = %f\n',averageAcc(2));
    fprintf('Accuracy from setting entire U to negative class = %f\n',averageAcc(3));
end