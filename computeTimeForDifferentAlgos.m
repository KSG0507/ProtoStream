runCode = false;
kernelType = 'Gaussian';
sigmaVal = 5;
name = 'MNIST';
% Level of sparsity
m = 200;
startSparsity = m;
numLevelsSparsity = m-startSparsity+1;
incrementLevels4KM = 50;
sparsityLevels4KM = startSparsity:incrementLevels4KM:m; 
numLevelsSparsity4KM = length(sparsityLevels4KM);
%proportions = [10 20 30 40 50 60 70 80 90 100];
proportions = [10 30 50 70 90 100];
numRuns = length(proportions)*10;
%% Read data
[test,labels_test,YT,labels_YT] = readMNIST(60000);
%% Randomly choose the training data set
numSamples = min(1500,length(labels_YT));
sampleNum = randperm(length(labels_YT));
Y = YT(:,sampleNum(1:numSamples));
labels_Y = labels_YT(sampleNum(1:numSamples));
%%
runBudget = true;
runGreedy = true;
%%
outputFileName = 'ComputeTime';
if(runGreedy)
    outputFileName = strcat(outputFileName,'WithGreedy');
end
outputFileName = strcat(outputFileName,'_',name,'_m',num2str(m),'_I',num2str(numSamples),'_K_',kernelType);
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
    %% Run other algorithms exactly once
    fprintf('Computing meanInnerProductY...\n');
    meanInnerProductY = computeMeanInnerProductX(Y,Y,kernelType,sigmaVal,'faster');
    fprintf('Running L2C...\n');
    startL = tic;
    [w_L,S_L,setValues_L] = Learn2CriticizeSetSelection(Y,Y,m,kernelType,sigmaVal,meanInnerProductY);
    tL = toc(startL);
    fprintf('Time taken to run L2C = %f secs\n',tL);
    fprintf('Running Random\n');
    startR = tic;
    [w_RU,setValues_RU,w_RE,setValues_RE,S_R] = RandomSetSelection(Y,Y,m,kernelType,sigmaVal,meanInnerProductY);
    tR = toc(startR);
    fprintf('Time taken to run Random = %f secs\n',tR);
    fprintf('Running K-medoids\n');
    startK = tic;
    [w_MU,setValues_MU,w_ME,setValues_ME,S_M] = KmedoidsSetSelection(Y,Y,m,kernelType,sigmaVal,startSparsity,incrementLevels4KM,'faster');
    tK = toc(startK);
    fprintf('Time taken to run K-Medoids = %f secs\n',tK);
    r = 0;
    for labelNum = 0:0
        for prop = 1:1
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
                    startB = tic;
                    [w_B,S_B,setValues_B,allw_B,numNonZero] = SVMBudgetSetSelection(X,Y,m,kernelType,individualMaxVal,sigmaVal,meanInnerProductX);
                    tB = toc(startB);
                    fprintf('l1bound = %f\tLength = %d\tNum nonzero=%d\n',l1bound,length(S_B),numNonZero);
                    fprintf('Time taken to run budget = %f secs\n',tB);
                end
            end 
            %% Run ProtoDash to choose prototypes from Y that best represents X
            fprintf('Running ProtoDash for across prototype selection...\n');
            startH = tic;
            [w_HX,S_HX,setValues_HX,stageWeights_HX] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX);
            tH = toc(startH);
            fprintf('Time taken to run ProtoDash = %f secs\n',tH);
            %%
            if(runGreedy)
                fprintf('Running ProtoGreedy for across prototype selection...\n');
                startG = tic;
                [w_G,S_G,setValues_G,stageWeights_G] = GreedySetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX);
                tG = toc(startG);
                fprintf('Time taken to run ProtoGreedy = %f secs\n',tG);
            end
        end
    end
    tValues = [tH,tL,tR,tK,tB,tG];
    save(outputFileName,'tValues');
else
    load(outputFileName);
end
algos = {'PrDash','L2C','RndW','K-Med','P-Las','PrGrdy'};
axesFontSize = 30;
p = axes;
han = bar(p,tValues,0.4);
x_loc = get(han, 'XData');
y_height = get(han, 'YData');
%arrayfun(@(x,y) text(x, y+0.2,num2str(y','%0.2f'), 'Color', 'r','HorizontalAlignment','center',...
%  'VerticalAlignment','bottom','fontsize',20,'fontweight','bold'), x_loc, y_height);
titleString = sprintf('b) MNIST time');
title(titleString,'fontsize',axesFontSize,'fontweight','bold');
ylabel('Computation time (secs)','fontsize',axesFontSize,'fontweight','bold');
set(gca,'fontsize',axesFontSize,'fontweight','bold');
set(gca,'XTickLabels',algos);
set(gca,'YScale','log');
%saveas(gcf,outputFileName,'jpeg');
%saveas(gcf,outputFileName);

