runCode = false;
name = 'MNIST';
kernelType = 'Gaussian';
sigmaVal = 5;
% Level of sparsity
m = 205;
m4KM = 205;
startSparsity = 5;
numLevelsSparsity = m-startSparsity+1;
incrementLevels4KM = 50;
sparsityLevels4KM = startSparsity:incrementLevels4KM:m4KM; 
numLevelsSparsity4KM = length(sparsityLevels4KM);
%%
switch name
    case 'random'
        %% Load random data
        X = randn(100,4000);
        Y = randn(100,4000);
        normalizeData = true;
        if (normalizeData)
            X = normc(X);
            Y = normc(Y);
        end
    case 'ORL'
        Path = '.\att_faces\';
       [Y,originalData] = get_Processed_ORL_Data(Path);
        X = Y;
    case 'MNIST'
        labelNum = 6;
        [I_test,labels_test,YT,labels_YT] = readMNIST(60000);
        numSamples = min(5000,length(labels_YT));
        sampleNum = randperm(length(labels_YT));
        Y = YT(:,sampleNum(1:numSamples));
        labelsY = labels_YT(sampleNum(1:numSamples));
        %locs = labels==5;
        %Y = Y(:,locs);
        locs = labels_test==labelNum;
        X = I_test(:,locs);
    otherwise
        %% Load NHANES 2013-2014 Disability data
        fileName = strcat(name,'.XPT');
        [Y,originalData] = get_Processed_NHANES_Data(fileName);
        X = Y;
end
%%
plotFigure = true; plotBudget = true;
saveOutput = true;
runBudget = true;
runGreedy = true;
%%
saveFigFileName='SetValueComparison_6';
if(runGreedy)
    saveFigFileName = strcat(saveFigFileName,'_WithGreedy');
end
saveFigFileName = strcat(saveFigFileName,'_',name,'_m',num2str(m),'_kVarying','_I',num2str(numSamples),'_K_',kernelType);
if(strcmp(kernelType,'Gaussian'))
    saveFigFileName = strcat(saveFigFileName,'_sigma',num2str(sigmaVal));
end
setValueFileName = strcat('Variables_',saveFigFileName);
fprintf('Output file: %s\n',setValueFileName);
if(runCode)
    %%
    fprintf('Computing the vector meanInnerProductX...\n');
    meanInnerProductX = computeMeanInnerProductX(X,Y,kernelType,sigmaVal,'faster');
    fprintf('Computing the vector meanInnerProductY...\n');
    meanInnerProductY = computeMeanInnerProductX(Y,Y,kernelType,sigmaVal,'faster');
    if(runBudget)
        fprintf('Running Budget\n');
        l1bound = 0.5;
        individualMaxVal = l1bound/m;
        [w_B,S_B,setValues_B,allw_B,numNonZero] = SVMBudgetSetSelection(X,Y,m,kernelType,individualMaxVal,sigmaVal,meanInnerProductX,'Incremental');
        fprintf('l1bound = %f\tLength = %d\tNum nonzero=%d\n',l1bound,length(S_B),numNonZero);
    end
    fprintf('Running Heuristic\n');
    [~,~,setValues_H,~,~] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX,1);
    %[~,~,setValues_H5,~,~] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX,5);
    %[~,~,setValues_H10,~,~] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX,10);
    %[~,~,setValues_H20,~,~] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX,20);
    %[~,~,setValues_H50,~,~] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX,50);
    %[~,~,setValues_H100,~,~] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX,100);
    %[~,~,setValues_H200,~,~] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX,200);
    fprintf('Running L2C adapted\n');
    [w_LA,S_LA,setValues_LA] = Learn2CriticizeSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX);
    fprintf('Running Learn2Criticize\n');
    [w_L,S_L,setValues_L] = Learn2CriticizeSetSelection(Y,Y,m,kernelType,sigmaVal,meanInnerProductY);
    fprintf('Running Random\n');
    [w_RU,setValues_RU,w_RE,setValues_RE,S_R] = RandomSetSelection(Y,Y,m,kernelType,sigmaVal,meanInnerProductY);
    fprintf('Running K-medoids\n');
    [w_MU,setValues_MU,w_ME,setValues_ME,S_M] = KmedoidsSetSelection(Y,Y,m4KM,kernelType,sigmaVal,startSparsity,incrementLevels4KM,'faster');
    if(runGreedy)
        fprintf('Running Greedy\n');
        [w_G,S_GT,setValues_G,stageWeights_G] = GreedySetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX);
    end
    if(saveOutput)
        deleteFileName = strcat(setValueFileName,'.mat');
        delete(deleteFileName);
        if(runBudget)
            if(runGreedy)
                save(setValueFileName,'setValues_H','setValues_L','setValues_LA','setValues_RU','setValues_MU','setValues_B','setValues_G');
            else
                save(setValueFileName,'setValues_H','setValues_L','setValues_LA','setValues_RU','setValues_MU','setValues_B');
            end
        else
            if(runGreedy)
               save(setValueFileName,'setValues_H','setValues_L','setValues_LA','setValues_RU','setValues_MU','setValues_G');
            else
                save(setValueFileName,'setValues_H','setValues_L','setValues_LA','setValues_RU','setValues_MU');
            end
        end
    end
else
    load(setValueFileName);
end
%%
figure(101);
if(plotFigure)
    plot(startSparsity:m,setValues_H(startSparsity:m),'g--','Linewidth',2);
    hold on;
    %plot(startSparsity:m,setValues_H5(startSparsity:m),'g-.','Linewidth',2);
    %plot(startSparsity:m,setValues_H10(startSparsity:m),'g-','Linewidth',2,'MarkerSize',10);
    %plot(startSparsity:m,setValues_H20(startSparsity:m),'g:','Linewidth',2,'MarkerSize',10);
    %plot(startSparsity:m,setValues_H50(startSparsity:m),'m--','Linewidth',2,'MarkerSize',10);
    %plot(startSparsity:m,setValues_H100(startSparsity:m),'m-.','Linewidth',2,'MarkerSize',10);
    %plot(startSparsity:m,setValues_H200(startSparsity:m),'m-','Linewidth',2,'MarkerSize',10);
    plot(startSparsity:m,setValues_L(startSparsity:m),'b-.','Linewidth',2,'MarkerSize',10);
    plot(startSparsity:m,setValues_LA(startSparsity:m),'b--','Linewidth',2,'MarkerSize',10);
    plot(startSparsity:m,setValues_RU(startSparsity:m),'k-','Linewidth',2,'MarkerSize',10);
    %plot(startSparsity:m,setValues_RE(startSparsity:m),'k-.d','Linewidth',2,'MarkerSize',10);
    plot(sparsityLevels4KM,setValues_MU(sparsityLevels4KM),'r-.s','Linewidth',2,'MarkerSize',10);
    %plot(startSparsity:m,setValues_ME(startSparsity:m),'r-+','Linewidth',2,'MarkerSize',10);
    if(plotBudget && runBudget)
        plot(startSparsity:length(setValues_B),setValues_B(startSparsity:end),'c-','Linewidth',2,'MarkerSize',10);
    end
    if(runGreedy)
        plot(startSparsity:m,setValues_G(startSparsity:m),'m--','Linewidth',2,'MarkerSize',10);
    end
    hold off;
    titleString = sprintf('MNIST');
    title(titleString,'fontsize',24,'fontweight','bold');
    xlabel('Sparsity level (m)','fontsize',30,'fontweight','bold');
    ylabel('Function value','fontsize',30,'fontweight','bold');
    if(plotBudget && runBudget)
        if(runGreedy)
            leghan1 = legend('PrDash','L2C','L2C-A','RndW','K-Med','P-Las','PrGreedy');
        else
            leghan1 = legend('PrDash','L2C','L2C-A','RndW','K-Med','P-Las');
        end
    else
        if(runGreedy)
            leghan1 = legend('PrDash','L2C','L2C-A','RndW','K-Med','PrGreedy');
        else
            leghan1 = legend('PrDash','L2C','L2C-A','RndW','K-Med');
        end
    end
    %leghan1 = legend('PrDash (k=1)','PrDash (k=5)','PrDash (k=10)','PrDash (k=20)','PrDash (k=50)','PrDash (k=100)','PrDash (k=200)','L2C','RndW','K-Med');
    set(gca,'fontsize',30,'fontweight','bold');
    set(leghan1,'fontsize',22,'fontweight','bold');
    if(saveOutput)
        saveas(gcf,setValueFileName,'jpeg');
        saveas(gcf,setValueFileName);
    end
end