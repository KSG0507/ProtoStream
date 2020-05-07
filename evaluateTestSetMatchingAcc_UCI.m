kernelType = 'Gaussian';
sigmaVal = 5;
name = 'UCI';
allLabels = 'A':'Z';
numLabels = length(allLabels);
% Level of sparsity
m = 100;
kValues = [1 m];
fixTestsetSize = false;
numkValues = length(kValues);
startSparsity = 50;
numLevelsSparsity = m-startSparsity+1;
incrementLevels4KM = 50;
sparsityLevels4KM = startSparsity:incrementLevels4KM:m; 
numLevelsSparsity4KM = length(sparsityLevels4KM);
%proportions = [10 20 30 40 50 60 70 80 90 100];
proportions = [4 10 30 50 70 90 100];
labelPosToRun = [4 7 12 20 25];
numDiffLablesToRun = length(labelPosToRun);
numRuns = length(proportions)*numDiffLablesToRun;
%% Read data
[test,labels_test,Y,labels_Y] = readUCILetters;
numSamples = numel(labels_Y);
%%
plotFigure = false; plotBudget = false;
runBudget = true;
runGreedy = true;
runStreaming = true;
saveOutput = true;
%%
if(runStreaming)
    outputFileName = 'TestSetMatchingAccuracy_Streaming_';
else
    outputFileName = 'TestSetMatchingAccuracy';
end
if(runGreedy)
    outputFileName = strcat(outputFileName,'WithGreedyAndAdaptedL2C');
else
     outputFileName = strcat(outputFileName,'WithAdaptedL2C');
end
outputFileName = strcat(outputFileName,'_',name,'_m',num2str(m),'_k','Extremes','_I',num2str(numSamples),'_K_',kernelType);
if(strcmp(kernelType,'Gaussian'))
    outputFileName = strcat(outputFileName,'_sigma',num2str(sigmaVal));
end
fprintf('Output file: %s\n',outputFileName);
%%
numTest = length(labels_test);
numIndividualLabels = zeros(numLabels,1);
individualLabelData = cell(numLabels,1);
for labelPos = 1:numLabels
    labelName = allLabels(labelPos);
    dataPos = labels_test==labelName;
    numIndividualLabels(labelPos) = sum(dataPos);
    individualLabelData{labelPos} = test(:,dataPos);
end
minNumIndividualLabels = min(numIndividualLabels);
if(fixTestsetSize)
    testsetSize = minNumIndividualLabels;
end
%% Run other algorithms exactly once
fprintf('Computing meanInnerProductY...\n');
meanInnerProductY = computeMeanInnerProductX(Y,Y,kernelType,sigmaVal,'faster');
fprintf('Running L2C...\n');
[w_L,S_L,setValues_L] = Learn2CriticizeSetSelection(Y,Y,m,kernelType,sigmaVal,meanInnerProductY);
fprintf('Running Random\n');
[w_RU,setValues_RU,w_RE,setValues_RE,S_R] = RandomSetSelection(Y,Y,m,kernelType,sigmaVal,meanInnerProductY);
fprintf('Running K-medoids\n');
[w_MU,setValues_MU,w_ME,setValues_ME,S_M] = KmedoidsSetSelection(Y,Y,m,kernelType,sigmaVal,startSparsity,incrementLevels4KM,'faster');
%%
a_HX = zeros(numRuns,numLevelsSparsity,numkValues);
a_HS = zeros(numRuns,numLevelsSparsity);
setSize_PDS = zeros(1,numRuns);
a_L = zeros(numRuns,numLevelsSparsity);
a_LA = zeros(numRuns,numLevelsSparsity);
a_R = zeros(numRuns,numLevelsSparsity);
a_M = zeros(numRuns,numLevelsSparsity4KM);
if(runBudget)
    a_B = zeros(numRuns,numLevelsSparsity);
end
if(runGreedy)
    a_G = zeros(numRuns,numLevelsSparsity);
end
r = 0;
for labelCount = 1:numDiffLablesToRun
    labelPos = labelPosToRun(labelCount);
    for prop = 1:length(proportions)
        propValue = proportions(prop);
        propOfOtherLabels = (100-propValue)/(numLabels-1);
        fprintf('Creating test set for label %s with proportion %f\n',allLabels(labelPos),propValue);
        r = r+1;
        fprintf('Run number:%d\n',r);
        %% define the training data set X with right percentage
        fprintf('Creating the test data with the right set of proportions....\n');
        if(~fixTestsetSize)
            tempVal1 = floor(numIndividualLabels(labelPos)*100/propValue);
            if(propOfOtherLabels==0)
                tempVal2 = inf;
            else
                tempVal2 = floor(minNumIndividualLabels*100/propOfOtherLabels);
            end
            testsetSize = min(tempVal1,tempVal2);
        end
        numCurrentLabelSamples = ceil((propValue/100)*testsetSize);
        sampleNum = randperm(numIndividualLabels(labelPos));
        X = individualLabelData{labelPos}(:,sampleNum(1:numCurrentLabelSamples));
        labels_X = repmat(allLabels(labelPos),1,numCurrentLabelSamples);
        numX = numCurrentLabelSamples;
        if(propOfOtherLabels > 0)
            for otherLabels = 1:numLabels
                if(otherLabels~=labelPos)
                    numOtherLabelSamples = ceil((propOfOtherLabels/100)*testsetSize);
                    sampleNum = randperm(numIndividualLabels(otherLabels));
                    X = horzcat(X,individualLabelData{otherLabels}(:,sampleNum(1:numOtherLabelSamples)));
                    labels_X = horzcat(labels_X,repmat(allLabels(otherLabels),1,numOtherLabelSamples));
                    numX = numX + numOtherLabelSamples;
                end
            end
        end
        fprintf('Size of test data = %d\n',numX);
        %% Computing mean inner product with different data sets
        fprintf('Computing meanInnerProductX...\n');
        meanInnerProductX = computeMeanInnerProductX(X,Y,kernelType,sigmaVal,'faster');
        %%
        if(runBudget)
            fprintf('Running Budget\n');
            for l1bound = 0.45:0.01:0.45
                individualMaxVal = l1bound/m;
                [w_B,S_B,setValues_B,allw_B,numNonZero] = SVMBudgetSetSelection(X,Y,m,kernelType,individualMaxVal,sigmaVal,meanInnerProductX ,'Incremental');
                fprintf('l1bound = %f\tLength = %d\tNum nonzero=%d\n',l1bound,length(S_B),numNonZero);
            end
            fprintf('Computing classification accuracy for Budget at all levels of sparsity...\n');
            classifiedLabels_B = NNC(X,Y(:,S_B),labels_Y(S_B),startSparsity);
            numLevelsSparsity4B = size(classifiedLabels_B,2);
        end 
        %% Run ProtoDash to choose prototypes from Y that best represents X
        classifiedLabels_HX = zeros(size(X,2),numLevelsSparsity,numkValues);
        fprintf('Running ProtoDash for across prototype selection...\n');
        for kcount = 1:numkValues
            k = kValues(kcount);
            if(k==m)
                [~,S_HX,~,~] = ProtoDashStreaming(X,Y,m,kernelType,sigmaVal,meanInnerProductX);
            else
                [~,S_HX,~,~] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX,k);
            end
            fprintf('Computing classification accuracy for ProtoDash across at all levels of sparsity...\n');
            classifiedLabels_HX(:,:,kcount) = NNC(X,Y(:,S_HX(1:m)),labels_Y(S_HX(1:m)),startSparsity);
        end
        %% Run ProtoDash streaming to choose prototypes from Y that best represents X
        if(runStreaming)
            epsilon = 0.4;
            [W_HS, S_HS, setValue_HS] = ProtoDashStreamingWithMultipleThreshold(X,Y,m,kernelType,epsilon,sigmaVal,meanInnerProductX);
            setSize_PDS(r) = min(m,numel(S_HS));
            classifiedLabels_HS = NNC(X,Y(:,S_HS(1:setSize_PDS(r))),labels_Y(S_HS(1:setSize_PDS(r))),startSparsity);
        end
        %% Run L2C adapted to choose prototypes from Y that best represents X
        fprintf('Running L2C adapted for across prototype selection...\n');
        [w_LA,S_LA,setValues_LA] = Learn2CriticizeSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX);
        fprintf('Computing classification accuracy for L2C across at all levels of sparsity...\n');
        classifiedLabels_LA = NNC(X,Y(:,S_LA),labels_Y(S_LA),startSparsity);
        %%
        fprintf('Computing classification accuracy for L2C at all levels of sparsity...\n');
        classifiedLabels_L = NNC(X,Y(:,S_L),labels_Y(S_L),startSparsity);
        %%
        fprintf('Computing classification accuracy for random at all levels of sparsity...\n');
        classifiedLabels_R = NNC(X,Y(:,S_R),labels_Y(S_R),startSparsity);
        %%
        fprintf('Computing classification accuracy for K-medoids at given levels of sparsity...\n');
        classifiedLabels_M = zeros(numX,numLevelsSparsity4KM);
        for levelNum = 1:numLevelsSparsity4KM
            spLevel = sparsityLevels4KM(levelNum);
            protoPos = S_M(1:spLevel,levelNum);
            classifiedLabels_M(:,levelNum) = NNC(X,Y(:,protoPos),labels_Y(protoPos),spLevel);
            a_M(r,levelNum) = (sum(classifiedLabels_M(:,levelNum)==labels_X(:))/numX)*100;
        end
        %%
        if(runGreedy)
            fprintf('Running ProtoGreedy for across prototype selection...\n');
            [w_G,S_G,setValues_G,stageWeights_G] = GreedySetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX);
            fprintf('Computing classification accuracy for ProtoGreedy across at all levels of sparsity...\n');
            classifiedLabels_G = NNC(X,Y(:,S_G),labels_Y(S_G),startSparsity);
        end
        for levelNum = 1:numLevelsSparsity
            for kcount = 1:numkValues
                a_HX(r,levelNum,kcount) = (sum(classifiedLabels_HX(:,levelNum,kcount)==labels_X(:))/numX)*100;
            end
            if(runStreaming)
                if(levelNum <= setSize_PDS(r)-startSparsity+1)
                    a_HS(r,levelNum) = (sum(classifiedLabels_HS(:,levelNum)==labels_X(:))/numX)*100;
                end
            end
            a_L(r,levelNum) = (sum(classifiedLabels_L(:,levelNum)==labels_X(:))/numX)*100;
            a_LA(r,levelNum) = (sum(classifiedLabels_LA(:,levelNum)==labels_X(:))/numX)*100;
            a_R(r,levelNum) = (sum(classifiedLabels_R(:,levelNum)==labels_X(:))/numX)*100;
            if(runBudget)
                if(levelNum <= numLevelsSparsity4B)
                    a_B(r,levelNum) = (sum(classifiedLabels_B(:,levelNum)==labels_X(:))/numX)*100;
                end
            end
            if(runGreedy)
                a_G(r,levelNum) = (sum(classifiedLabels_G(:,levelNum)==labels_X(:))/numX)*100;
            end
        end
        %%
        if(saveOutput)
            deleteFileName = strcat('Variables_',outputFileName,'.mat');
            delete(deleteFileName);
            if(runBudget)
                if(runGreedy)
                    save(strcat('Variables_',outputFileName),'setSize_PDS','a_HX','a_HS','a_L','a_LA','a_R','a_M','a_B','a_G');
                else
                    save(strcat('Variables_',outputFileName),'setSize_PDS','a_HX','a_HS','a_L','a_LA','a_R','a_M','a_B');
                end
            else
                if(runGreedy)
                    save(strcat('Variables_',outputFileName),'setSize_PDS','a_HX','a_HS','a_L','a_LA','a_R','a_M','a_G');
                else
                    save(strcat('Variables_',outputFileName),'setSize_PDS','a_HX','a_HS','a_L','a_LA','a_R','a_M');
                end
            end
        end
    end
end
if(plotFigure)
    a_HXMean = zeros(numLevelsSparsity,numkValues);
    for kcount = 1:numkValues
        a_HXMean(:,kcount) = mean(a_HX(:,:,kcount),1)';
    end
    if(runStreaming)
        a_HSMean = zeros(1,numLevelsSparsity);
        for i = 1:numLevelsSparsity
            nonZeroLoc = a_HS(:,i)~=0;
            a_HSMean(i) = mean(a_HS(nonZeroLoc,i));
        end
    end
    a_LMean = mean(a_L,1);
    a_LAMean = mean(a_LA,1);
    a_RMean = mean(a_R,1);
    a_MMean = mean(a_M,1);
    if(runBudget)
        a_BMean = zeros(1,numLevelsSparsity);
        for i = 1:numLevelsSparsity
            nonZeroLoc = a_B(:,i)~=0;
            a_BMean(i) = mean(a_B(nonZeroLoc,i));
        end
    end
    if(runGreedy)
        a_GMean = mean(a_G,1);
    end
    figure(101);
    plot(startSparsity:m,a_HXMean(:,1),'g--','Linewidth',2);
    hold on;
    plot(startSparsity:m,a_HXMean(:,2),'g-.','Linewidth',2);
    plot(startSparsity:m,a_HXMean(:,3),'g-','Linewidth',2);
    plot(startSparsity:m,a_HXMean(:,4),'g:','Linewidth',2);
    plot(startSparsity:m,a_HXMean(:,5),'m--','Linewidth',2);
    plot(startSparsity:m,a_HXMean(:,6),'m-.','Linewidth',2);
    if(runStreaming)
        plot(startSparsity:m,a_HSMean,'m-','Linewidth',2);
    end
    %plot(startSparsity:m,a_HX200Mean,'m-','Linewidth',2);
    plot(startSparsity:m,a_LMean,'b-.','Linewidth',2);
    plot(startSparsity:m,a_LAMean,'b--','Linewidth',2);
    plot(startSparsity:m,a_RMean,'k-','Linewidth',2);
    plot(sparsityLevels4KM,a_MMean,'r-.s','Linewidth',2,'MarkerSize',10);
    if(plotBudget && runBudget)
        plot(startSparsity:m,a_BMean,'c-','Linewidth',2,'MarkerSize',10);
    end
    if(runGreedy)
        plot(startSparsity:m,a_GMean,'m--','Linewidth',2,'MarkerSize',10);
    end
    hold off;
    titleString = sprintf('Dataset: %s',name);
    title(titleString,'fontsize',24,'fontweight','bold');
    xlabel('Sparsity level','fontsize',20,'fontweight','bold');
    ylabel('Classification Accuracy','fontsize',20,'fontweight','bold');
    if(plotBudget && runBudget)
        if(runGreedy)
            legend('ProtoDash','L2C','L2C Adpated','Random','K-Medoids','P-Lasso','ProtoGreedy');
        else
            legend('PrDash (k=1)','PrDash (k=5)','PrDash (k=10)','PrDash (k=20)','PrDash (k=50)','PrDash (k=100)','PrDash Streaming','L2C','L2C-A','RndW','K-Med','P-Las');
        end
    else
        if(runGreedy)
            legend('ProtoDash','L2C','L2C Adpated','Random','K-Medoids','ProtoGreedy');
        else
            legend('PrDash (k=1)','PrDash (k=5)','PrDash (k=10)','PrDash (k=20)','PrDash (k=50)','PrDash (k=100)','PrDash Streaming','L2C','L2C-A','RndW','K-Med');
        end
    end
    set(gca,'fontsize',20,'fontweight','bold');
    if(saveOutput)
        saveas(gcf,outputFileName,'jpeg');
        saveas(gcf,outputFileName);
    end
end
