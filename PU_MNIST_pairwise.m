runCode = false;
% load('CurrentSeed.mat');
% rng(seedVal);
kernelType = 'Gaussian';
sigmaVal = 5;
name = 'MNIST';
allLabels = 0:9;
numLabels = length(allLabels);
setDiffStopTh = 0.00000000001;
% Level of sparsity
m = 1500;
pairPosToRun = [1,8; 2,8; 3,10; 4,7; 1,10; 5,6; 8,9];
numDiffPairsToRun = size(pairPosToRun,1);
propOfCurrLabelInP = [50];
propOfOtherLabelsInU = 100;
propOfTestData = 20;
%% Read data
[train_T, labels_train_T,test_T,labels_test_T] = readMNIST(60000);
complete_data = horzcat(train_T,test_T);
complete_labels = horzcat(labels_train_T(:)',labels_test_T(:)');
test_size = ceil((propOfTestData/100) * numel(complete_labels));
train_size = numel(complete_labels) - test_size;
sampleNum = randperm(numel(complete_labels));
train = complete_data(:,sampleNum(1:train_size));
labels_train = complete_labels(sampleNum(1:train_size));
test_all = complete_data(:,sampleNum(train_size+1:end));
labels_test_all = complete_labels(sampleNum(train_size+1:end));
%%
numSamples = numel(labels_train);
%%
plotFigure = true; 
saveOutput = true;
%%
outputFileName = 'PU_Pairwise';
outputFileName = strcat(outputFileName,'_',name,'_m',num2str(m),'_I',num2str(numSamples),'_K_',kernelType);
if(strcmp(kernelType,'Gaussian'))
    outputFileName = strcat(outputFileName,'_sigma',num2str(sigmaVal));
end
fprintf('Output file: %s\n',outputFileName);
%%
if(runCode)
    numIndividualLabels = zeros(numLabels,1);
    individualLabelData = cell(numLabels,1);
    numIndLabelsInTest = zeros(numLabels,1);
    individualLabelDataInTest = cell(numLabels,1);
    for labelPos = 1:numLabels
        labelName = allLabels(labelPos);
        dataPos = labels_train==labelName;
        numIndividualLabels(labelPos) = sum(dataPos);
        individualLabelData{labelPos} = train(:,dataPos);
        
        dataPos = labels_test_all==labelName;
        numIndLabelsInTest(labelPos) = sum(dataPos);
        individualLabelDataInTest{labelPos} = test_all(:,dataPos);
    end
    %%
    TypesOfClassication = {'UseLabelledInfo','SelectPrototypes','SetAllUtoClassNeg'};
    accuracyPercentage = zeros(numDiffPairsToRun,numel(TypesOfClassication));
    selectedPrototypeIndices = zeros(numDiffPairsToRun,m);
    protoLabels = zeros(numDiffPairsToRun,m);
    setSizes = zeros(numDiffPairsToRun,1);
    %%
    for labelCount = 1:numDiffPairsToRun
        labelPos = pairPosToRun(labelCount,1);
        labelName = allLabels(labelPos);
        propValue = propOfCurrLabelInP(1);
        P_size = floor(numIndividualLabels(labelPos)*propValue/100);
        sampleNum = randperm(numIndividualLabels(labelPos));
        P_data= individualLabelData{labelPos}(:,sampleNum(1:P_size));
        U_data = individualLabelData{labelPos}(:,sampleNum(P_size+1:end));
        U_size = numIndividualLabels(labelPos)-P_size;
        true_labels_U = repmat(allLabels(labelPos),1,U_size);
        locOfPinU = 1:U_size;
        %%
        test = individualLabelDataInTest{labelPos};
        ground_truth = ones(1,numIndLabelsInTest(labelPos));
        %%
        for otherLabelCount = 2:size(pairPosToRun,2)
            otherLabelPos = pairPosToRun(labelCount,otherLabelCount);
            if(otherLabelPos~=labelPos)
                numOtherLabelSamples = ceil((propOfOtherLabelsInU/100)*numIndividualLabels(otherLabelPos));
                sampleNum = randperm(numIndividualLabels(otherLabelPos));
                U_data = horzcat(U_data,individualLabelData{otherLabelPos}(:,sampleNum(1:numOtherLabelSamples)));
                true_labels_U = horzcat(true_labels_U,repmat(allLabels(otherLabelPos),1,numOtherLabelSamples));
                U_size = U_size + numOtherLabelSamples;
                test = horzcat(test, individualLabelDataInTest{otherLabelPos});
                ground_truth = horzcat(ground_truth, -1*ones(1,numIndLabelsInTest(otherLabelPos)));
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