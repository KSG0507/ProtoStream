runCode = false;
kernelType = 'Gaussian';
sigmaVal = 5;
name = 'MNIST';
% Level of sparsity
m = 200;
kValues = [1 5 10 20 50 100 200];
numkValues = length(kValues);
startSparsity = 50;
numLevelsSparsity = m-startSparsity+1;
incrementLevels4KM = 50;
sparsityLevels4KM = startSparsity:incrementLevels4KM:m; 
numLevelsSparsity4KM = length(sparsityLevels4KM);
%proportions = [10 20 30 40 50 60 70 80 90 100];
proportions = [10 30 50 70 90 100];
numProp = length(proportions);
numRuns = length(proportions)*10;
%% Read data
[test,labels_test,YT,labels_YT] = readMNIST(60000);
%% Randomly choose the training data set
numSamples = min(5000,length(labels_YT));
sampleNum = randperm(length(labels_YT));
Y = YT(:,sampleNum(1:numSamples));
labels_Y = labels_YT(sampleNum(1:numSamples));
%%
plotFigure = true; plotBudget = true;
runBudget = true;
runGreedy = false;
saveOutput = false;
%%
outputFileName = 'PrototypeDistribution';
if(runGreedy)
    outputFileName = strcat(outputFileName,'WithGreedyAndAdaptedL2C');
else
     outputFileName = strcat(outputFileName,'WithAdaptedL2C');
end
outputFileName = strcat(outputFileName,'_',name,'_m',num2str(m),'_k','Varying','_I',num2str(numSamples),'_K_',kernelType);
if(strcmp(kernelType,'Gaussian'))
    outputFileName = strcat(outputFileName,'_sigma',num2str(sigmaVal));
end
fprintf('Output file: %s\n',outputFileName);
%%
if(runCode)
    numTest = length(labels_test);
    numIndividualLabels = zeros(10,1);
    individualLabelData = cell(10,1);
    for labelNum = 0:9
        dataPos = labels_test==labelNum;
        numIndividualLabels(labelNum+1) = sum(dataPos);
        individualLabelData{labelNum+1} = test(:,dataPos);
    end
    testsetSize = min(numIndividualLabels);
    %%
    protoLabels_HX = zeros(numRuns,m,numkValues);
    protoLabels_L = zeros(1,m);
    protoLabels_LA = zeros(numRuns,m);
    protoLabels_R = zeros(1,m);
    protoLabels_M = zeros(1,m);
    if(runBudget)
        protoLabels_B = zeros(numRuns,m);
    end
    if(runGreedy)
        protoLabels_G = zeros(numRuns,m);
    end
    %% Run other algorithms exactly once
    fprintf('Computing meanInnerProductY...\n');
    meanInnerProductY = computeMeanInnerProductX(Y,Y,kernelType,sigmaVal,'faster');
    fprintf('Running L2C...\n');
    [w_L,S_L,setValues_L] = Learn2CriticizeSetSelection(Y,Y,m,kernelType,sigmaVal,meanInnerProductY);
    protoLabels_L(1:m) = labels_Y(S_L);
    fprintf('Running Random\n');
    [w_RU,setValues_RU,w_RE,setValues_RE,S_R] = RandomSetSelection(Y,Y,m,kernelType,sigmaVal,meanInnerProductY);
    protoLabels_R(1:m) = labels_Y(S_R);
    fprintf('Running K-medoids\n');
    [w_MU,setValues_MU,w_ME,setValues_ME,S_M] = KmedoidsSetSelection(Y,Y,m,kernelType,sigmaVal,startSparsity,incrementLevels4KM,'faster');
    protoLabels_M(1:m) = labels_Y(S_M(1:m,end));
    %%
    r = 0;
    for labelNum = 0:9
        for prop = 1:length(proportions)
            propValue = proportions(prop);
            fprintf('Creating test set for label %d with proportion %f\n',labelNum,propValue);
            r = r+1;
            fprintf('Run number:%d\n',r);
            %% define the training data set X with right percentage
            fprintf('Creating the test data with the right set of proportions....\n');
            numCurrentLabelSamples = ceil((propValue/100)*testsetSize);
            sampleNum = randperm(numIndividualLabels(labelNum+1));
            X = individualLabelData{labelNum+1}(:,sampleNum(1:numCurrentLabelSamples));
            labels_X = labelNum*ones(1,numCurrentLabelSamples);
            numX = numCurrentLabelSamples;
            propOfOtherLabels = (100-propValue)/9;
            if(propOfOtherLabels > 0)
                for otherLabels = 0:9
                    if(otherLabels~=labelNum)
                        numOtherLabelSamples = ceil((propOfOtherLabels/100)*testsetSize);
                        sampleNum = randperm(numIndividualLabels(otherLabels+1));
                        X = horzcat(X,individualLabelData{otherLabels+1}(:,sampleNum(1:numOtherLabelSamples)));
                        labels_X = horzcat(labels_X,otherLabels*ones(1,numOtherLabelSamples));
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
                for l1bound = 0.45:0.1:0.45
                    individualMaxVal = l1bound/m;
                    [w_B,S_B,setValues_B,allw_B,numNonZero] = SVMBudgetSetSelection(X,Y,m,kernelType,individualMaxVal,sigmaVal,meanInnerProductX);
                    protoLabels_B(r,1:length(S_B)) = labels_Y(S_B);
                    fprintf('l1bound = %f\tLength = %d\tNum nonzero=%d\n',l1bound,length(S_B),numNonZero);
                end
            end 
            %% Run ProtoDash to choose prototypes from Y that best represents X
            fprintf('Running ProtoDash for across prototype selection...\n');
            for kcount = 1:numkValues
                k = kValues(kcount);
                [~,S_HX,~,~] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX,k);
                protoLabels_HX(r,1:m,kcount) = labels_Y(S_HX(1:m));
            end
            %% Run L2C adapted to choose prototypes from Y that best represents X
            fprintf('Running L2C adapted for across prototype selection...\n');
            [w_LA,S_LA,setValues_LA] = Learn2CriticizeSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX);
            protoLabels_LA(r,1:m) = labels_Y(S_LA);
            %%
            if(runGreedy)
                fprintf('Running ProtoGreedy for across prototype selection...\n');
                [w_G,S_G,setValues_G,stageWeights_G] = GreedySetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX);
                protoLabels_G(r,1:m) = labels_Y(S_G);
            end
            if(saveOutput)
                deleteFileName = strcat('Variables_',outputFileName,'.mat');
                delete(deleteFileName);
                if(runBudget)
                    if(runGreedy)
                        save(strcat('Variables_',outputFileName),'protoLabels_HX','protoLabels_L','protoLabels_LA','protoLabels_R','protoLabels_M','protoLabels_B','protoLabels_G');
                    else
                        save(strcat('Variables_',outputFileName),'protoLabels_HX','protoLabels_L','protoLabels_LA','protoLabels_R','protoLabels_M','protoLabels_B');
                    end
                else
                    if(runGreedy)
                        save(strcat('Variables_',outputFileName),'protoLabels_HX','protoLabels_L','protoLabels_LA','protoLabels_R','protoLabels_M','protoLabels_G');
                    else
                        save(strcat('Variables_',outputFileName),'protoLabels_HX','protoLabels_L','protoLabels_LA','protoLabels_R','protoLabels_M');
                    end
                end
            end
        end
    end
else
    load(strcat('Variables_',outputFileName));
end
if(plotFigure)
    plottingLabel = 8;
    axesFontSize = 20;
    for p = 1:numProp
        fh = figure(p);
        titleString = sprintf('Label distribution for %s %% skew',num2str(proportions(p)));
        annotation(fh,'textbox',[0.25 0.96 0.56 0.06],'String',titleString,...
        'LineStyle','none',...
        'FontWeight','bold',...
        'FontSize',26,...
        'FitBoxToText','off');
        for kcount = 1:numkValues
            subplot(3,3,kcount);
            histogram(categorical(protoLabels_HX(p+plottingLabel*numProp,1:m,kcount)));
            titleString = sprintf('k = %s',num2str(kValues(kcount)));
            title(titleString,'fontsize',axesFontSize,'fontweight','bold');
            axis tight;
            %set(gca,'fontsize',15,'fontweight','bold');
%             xlabel('Labels','fontsize',axesFontSize,'fontweight','bold');
%             ylabel('Count','fontsize',axesFontSize,'fontweight','bold');
        end
        if(saveOutput)
            saveFigFileName = strcat(outputFileName,'_p',num2str(proportions(p)));
            saveas(gcf,saveFigFileName,'jpeg');
        end
    end
end