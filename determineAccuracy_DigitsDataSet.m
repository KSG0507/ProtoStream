kernelType = 'Gaussian';
sigmaVal = 5;
name = 'MNIST';
% Level of sparsity
m = 200;
startSparsity = 50;
numLevelsSparsity = m-startSparsity+1;
incrementLevels4KM = 50;
sparsityLevels4KM = startSparsity:incrementLevels4KM:m; 
numLevelsSparsity4KM = length(sparsityLevels4KM);
numSamples = 1500;
numRuns = 20;
zstar = 2.086/sqrt(numRuns);
outputFileName = strcat('ClassificationAccuracy_',name,'_m',num2str(m),'_I',num2str(numSamples),'_K_',kernelType);
if(strcmp(kernelType,'Gaussian'))
    outputFileName = strcat(outputFileName,'_sigma',num2str(sigmaVal));
end
plotFigure = true; plotBudget = true;
runBudget = true;
saveOutput = true;
leastNumLevelsSparsity4B = numLevelsSparsity;
%%
[YT,labels_YT,X,labels_X] = readMNIST(60000);
numX = size(X,2);
a_HY = zeros(numRuns,numLevelsSparsity);
a_HX = zeros(numRuns,numLevelsSparsity);
a_HX2 = zeros(numRuns,numLevelsSparsity);
a_L = zeros(numRuns,numLevelsSparsity);
a_R = zeros(numRuns,numLevelsSparsity);
a_M = zeros(numRuns,numLevelsSparsity4KM);
if(runBudget)
    a_B = zeros(numRuns,numLevelsSparsity);
end
for r = 1:numRuns
    fprintf('Run number:%d\n',r);
    sampleNum = randperm(60000);
    Y = YT(:,sampleNum(1:numSamples));
    labels_Y = labels_YT(sampleNum(1:numSamples));
    %% Computing mean inner product with different data sets
    fprintf('Computing meanInnerProductX...\n');
    meanInnerProductX = computeMeanInnerProductX(X,Y,kernelType,sigmaVal,'faster');
    fprintf('Computing meanInnerProductY...\n');
    meanInnerProductY = computeMeanInnerProductX(Y,Y,kernelType,sigmaVal,'faster');
    %%
    if(runBudget)
        fprintf('Running Budget\n');
        for l1bound = 0.45:0.1:0.45
            individualMaxVal = l1bound/m;
            [w_B,S_B,setValues_B,allw_B,numNonZero] = SVMBudgetSetSelection(X,Y,m,kernelType,individualMaxVal,sigmaVal,meanInnerProductX);
            fprintf('l1bound = %f\tLength = %d\tNum nonzero=%d\n',l1bound,length(S_B),numNonZero);
        end
        fprintf('Computing classification accuracy for Budget at all levels of sparsity...\n');
        classifiedLabels_B = NNC(X,Y(:,S_B),labels_Y(S_B),startSparsity);
        numLevelsSparsity4B = size(classifiedLabels_B,2);
        if(numLevelsSparsity4B < leastNumLevelsSparsity4B)
            leastNumLevelsSparsity4B = numLevelsSparsity4B;
        end
    end
    %% Run L2C to choose prototypes from Y that best represents Y
    fprintf('Running L2C...\n');
    [w_L,S_L,setValues_L] = Learn2CriticizeSetSelection(Y,Y,m,kernelType,sigmaVal,meanInnerProductY);
    fprintf('Computing classification accuracy for L2C at all levels of sparsity...\n');
    classifiedLabels_L = NNC(X,Y(:,S_L),labels_Y(S_L),startSparsity);
    %% Run ProtoDash to choose prototypes from Y that best represents Y
    fprintf('Running ProtoDash for within prototype selection...\n');
    [w_HY,S_HY,setValues_HY,stageWeights_HY] = HeuristicSetSelection(Y,Y,m,kernelType,sigmaVal,meanInnerProductY);
    fprintf('Computing classification accuracy for ProtoDash within at all levels of sparsity...\n');
    classifiedLabels_HY = NNC(X,Y(:,S_HY),labels_Y(S_HY),startSparsity);
    %% Run ProtoDash to choose prototypes from Y that best represents X
    fprintf('Running ProtoDash for across prototype selection...\n');
    [w_HX,S_HX,setValues_HX,stageWeights_HX] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX);
    fprintf('Computing classification accuracy for ProtoDash across at all levels of sparsity...\n');
    classifiedLabels_HX = NNC(X,Y(:,S_HX(1:m)),labels_Y(S_HX(1:m)),startSparsity);
%     S_HT2 = S_HX(1:2*m); [~,IX] = sort(stageWeights_HX(1:2*m,2*m),'descend');S_HX2 = S_HT2(IX(1:m));
%     fprintf('Computing classification accuracy for ProtoDash across at all levels of sparsity using top weighted prototypes...\n');
%     classifiedLabels_HX2 = NNC(X,Y(:,S_HX2),labels_Y(S_HX2),startSparsity);
    %%
    fprintf('Running Random\n');
    [w_RU,setValues_RU,w_RE,setValues_RE,S_R] = RandomSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX);
    fprintf('Computing classification accuracy for random at all levels of sparsity...\n');
    classifiedLabels_R = NNC(X,Y(:,S_R),labels_Y(S_R),startSparsity);
    %%
    fprintf('Running K-medoids\n');
    [w_MU,setValues_MU,w_ME,setValues_ME,S_M] = KmedoidsSetSelection(X,Y,m,kernelType,sigmaVal,startSparsity,incrementLevels4KM,'faster');
    classifiedLabels_M = zeros(numX,numLevelsSparsity4KM);
    for levelNum = 1:numLevelsSparsity4KM
        spLevel = sparsityLevels4KM(levelNum);
        protoPos = S_M(1:spLevel,levelNum);
        classifiedLabels_M(:,levelNum) = NNC(X,Y(:,protoPos),labels_Y(protoPos),spLevel);
        a_M(r,levelNum) = (sum(classifiedLabels_M(:,levelNum)==labels_X(:))/numX)*100;
    end
    %%
    for levelNum = 1:numLevelsSparsity
        a_HX(r,levelNum) = (sum(classifiedLabels_HX(:,levelNum)==labels_X(:))/numX)*100;
        a_HY(r,levelNum) = (sum(classifiedLabels_HY(:,levelNum)==labels_X(:))/numX)*100;
        a_L(r,levelNum) = (sum(classifiedLabels_L(:,levelNum)==labels_X(:))/numX)*100;
        a_R(r,levelNum) = (sum(classifiedLabels_R(:,levelNum)==labels_X(:))/numX)*100;
        if(runBudget)
            if(levelNum <= numLevelsSparsity4B)
                a_B(r,levelNum) = (sum(classifiedLabels_B(:,levelNum)==labels_X(:))/numX)*100;
            end
        end
        %a_HX2(r,levelNum) = (sum(classifiedLabels_HX2(:,levelNum)==labels_X(:))/numX)*100;
    end
end
a_HXMean = mean(a_HX,1);
a_HYMean = mean(a_HY,1);
a_LMean = mean(a_L,1);
a_RMean = mean(a_R,1);
a_MMean = mean(a_M,1);
if(runBudget)
    a_BMean = zeros(1,numLevelsSparsity);
    for i = 1:numLevelsSparsity
        nonZeroLoc = a_B(:,i)~=0;
        a_BMean(i) = mean(a_B(nonZeroLoc,i));
    end
end
%a_HX2Mean = mean(a_HX2,1);
% a_HXCI = zstar*std(a_HX,1,1);
% a_HYCI = zstar*std(a_HY,1,1);
% a_LCI = zstar*std(a_L,1,1);
if(plotFigure)
    figure(101);
    plot(startSparsity:m,a_HXMean,'g--','Linewidth',2);
    hold on;
    plot(startSparsity:m,a_HYMean,'m-','Linewidth',2,'MarkerSize',10);
    plot(startSparsity:m,a_LMean,'b-.','Linewidth',2,'MarkerSize',10);
    plot(startSparsity:m,a_RMean,'k-','Linewidth',2,'MarkerSize',10);
    plot(sparsityLevels4KM,a_MMean,'r-.s','Linewidth',2,'MarkerSize',10);
    if(plotBudget && runBudget)
        plot(startSparsity:m,a_BMean,'c-','Linewidth',2,'MarkerSize',10);
    end
    hold off;
    titleString = sprintf('Dataset: %s',name);
    title(titleString,'fontsize',24,'fontweight','bold');
    xlabel('Sparsity level','fontsize',20,'fontweight','bold');
    ylabel('Classification Accuracy','fontsize',20,'fontweight','bold');
    if(plotBudget && runBudget)
        legend('ProtoDash-Across','ProtoDash-Within','L2C','Random','K-Medoids','L1-Reg');
    else
        legend('ProtoDash-Across','ProtoDash-Within','L2C','Random','K-Medoids');
    end
    set(gca,'fontsize',20,'fontweight','bold');
    if(saveOutput)
        saveas(gcf,outputFileName,'jpeg');
        saveas(gcf,outputFileName);
    end
end

if(saveOutput)
    if(runBudget)
        save(strcat('Variables_',outputFileName),'a_HX','a_HY','a_L','a_R','a_M','a_B');
    else
        save(strcat('Variables_',outputFileName),'a_HX','a_HY','a_L','a_R','a_M');
    end
end