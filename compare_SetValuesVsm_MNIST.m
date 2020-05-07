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
%% Read data
[I_test,labels_test,YT,labels_YT] = readMNIST(60000);
numSamples = min(5000,length(labels_YT));
sampleNum = randperm(length(labels_YT));
Y = YT(:,sampleNum(1:numSamples));
labelsY = labels_YT(sampleNum(1:numSamples));
%%
plotFigure = true; plotBudget = true;
saveOutput = true;
runBudget = true;
runGreedy = true;
%%
saveFigFileName='SetValueComparison';
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
    setValues_H = zeros(10,m);
    setValues_L = zeros(10,m);
    setValues_LA = zeros(10,m);
    setValues_R = zeros(10,m);
    setValues_M = zeros(10,m);
    if(runBudget)
        setValues_B = zeros(10,m);
        setSize_B = zeros(10,1);
    end
    if(runGreedy)
        setValues_G = zeros(10,m);
    end
    for labelNum = 0:9
        fprintf('Label number = %d\n',labelNum);
        %name = strcat('MNIST-',num2str(labelNum));
        locs = labels_test==labelNum;
        X = I_test(:,locs);
        fprintf('Computing the vector meanInnerProductX...\n');
        meanInnerProductX = computeMeanInnerProductX(X,Y,kernelType,sigmaVal,'faster');
        if(runBudget)
            fprintf('Running Budget\n');
            l1bound = 0.9;
            individualMaxVal = l1bound/m;
            [w_B,S_B,sV_B,allw_B,numNonZero] = SVMBudgetSetSelection(X,Y,m,kernelType,individualMaxVal,sigmaVal,meanInnerProductX,'Incremental');
            fprintf('l1bound = %f\tLength = %d\tNum nonzero=%d\n',l1bound,length(S_B),numNonZero);
            fprintf('Size of set values = %d\n',numel(sV_B));
            setValues_B(labelNum+1,1:length(S_B)) = sV_B(1:length(S_B))';
            setSize_B(labelNum+1) = length(S_B);
        end
        fprintf('Running Heuristic\n');
        [~,~,sV_H,~,~] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX,1);
        setValues_H(labelNum+1,:) = sV_H(:)';
        %[~,~,setValues_H5,~,~] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX,5);
        %[~,~,setValues_H10,~,~] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX,10);
        %[~,~,setValues_H20,~,~] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX,20);
        %[~,~,setValues_H50,~,~] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX,50);
        %[~,~,setValues_H100,~,~] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX,100);
        %[~,~,setValues_H200,~,~] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX,200);
        %%
        fprintf('Running L2C adapted\n');
        [~,~,sV_LA] = Learn2CriticizeSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX);
        setValues_LA(labelNum+1,:) = sV_LA(:)';
        %%
        if(labelNum == 0)
                fprintf('Computing meanInnerProductY...\n');
                meanInnerProductY = computeMeanInnerProductX(Y,Y,kernelType,sigmaVal,'faster');
                fprintf('Running Learn2Criticize\n');
                [~,~,sV_L] = Learn2CriticizeSetSelection(Y,Y,m,kernelType,sigmaVal,meanInnerProductY);
                fprintf('Running Random\n');
                [w_RU,sV_RU,w_RE,sV_RE,S_R] = RandomSetSelection(Y,Y,m,kernelType,sigmaVal,meanInnerProductY);
                fprintf('Running K-medoids\n');
                [w_MU,sV_MU,w_ME,sV_ME,S_M] = KmedoidsSetSelection(Y,Y,m4KM,kernelType,sigmaVal,startSparsity,incrementLevels4KM,'faster');
        end
        setValues_L(labelNum+1,:) = sV_L(:)';
        setValues_R(labelNum+1,:) = sV_RU(:)';
        setValues_M(labelNum+1,:) = sV_MU(:)'; 
        
        if(runGreedy)
            fprintf('Running Greedy\n');
            [w_G,S_GT,sV_G,stageWeights_G] = GreedySetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX);
            setValues_G(labelNum+1,:) = sV_G(:)';
        end
        if(saveOutput)
            deleteFileName = strcat(setValueFileName,'.mat');
            delete(deleteFileName);
            if(runBudget)
                if(runGreedy)
                    save(setValueFileName,'setValues_H','setValues_L','setValues_LA','setValues_R','setValues_M','setValues_B','setSize_B','setValues_G');
                else
                    save(setValueFileName,'setValues_H','setValues_L','setValues_LA','setValues_R','setValues_M','setValues_B','setSize_B');
                end
            else
                if(runGreedy)
                   save(setValueFileName,'setValues_H','setValues_L','setValues_LA','setValues_R','setValues_M','setValues_G');
                else
                    save(setValueFileName,'setValues_H','setValues_L','setValues_LA','setValues_R','setValues_M');
                end
            end
        end
    end
else
    load(setValueFileName);
end
%%
if(plotFigure)
    figure(101);
    minSupportParam4B = m;
    sV_H = mean(setValues_H,1);
    sV_LA = mean(setValues_LA,1);
    sV_L = mean(setValues_L,1);
    sV_R = mean(setValues_R,1);
    sV_M = mean(setValues_M,1);
    if(runBudget)
        minSupportRuns = setSize_B >=minSupportParam4B;
        sV_B = mean(setValues_B(minSupportRuns,:),1);
    end
    if(runGreedy)
        sV_G = mean(setValues_G,1);
    end
    plot(startSparsity:m,sV_H(startSparsity:m),'g-','Linewidth',2);
    hold on;
    %plot(startSparsity:m,setValues_H5(startSparsity:m),'g-.','Linewidth',2);
    %plot(startSparsity:m,setValues_H10(startSparsity:m),'g-','Linewidth',2,'MarkerSize',10);
    %plot(startSparsity:m,setValues_H20(startSparsity:m),'g:','Linewidth',2,'MarkerSize',10);
    %plot(startSparsity:m,setValues_H50(startSparsity:m),'m--','Linewidth',2,'MarkerSize',10);
    %plot(startSparsity:m,setValues_H100(startSparsity:m),'m-.','Linewidth',2,'MarkerSize',10);
    %plot(startSparsity:m,setValues_H200(startSparsity:m),'m-','Linewidth',2,'MarkerSize',10);
    plot(startSparsity:m,sV_L(startSparsity:m),'b-.','Linewidth',2,'MarkerSize',10);
    %plot(startSparsity:m,sV_LA(startSparsity:m),'b--','Linewidth',2,'MarkerSize',10);
    plot(startSparsity:m,sV_R(startSparsity:m),'k-','Linewidth',2,'MarkerSize',10);
    %plot(startSparsity:m,setValues_RE(startSparsity:m),'k-.d','Linewidth',2,'MarkerSize',10);
    plot(sparsityLevels4KM,sV_M(sparsityLevels4KM),'r-.s','Linewidth',2,'MarkerSize',10);
    %plot(startSparsity:m,setValues_ME(startSparsity:m),'r-+','Linewidth',2,'MarkerSize',10);
    if(plotBudget && runBudget)
        sparsityLevelForB = min(setSize_B(setSize_B >=minSupportParam4B));
        plot(startSparsity:sparsityLevelForB,sV_B(startSparsity:sparsityLevelForB),'c-','Linewidth',2,'MarkerSize',10);
    end
    if(runGreedy)
        plot(startSparsity:m,sV_G(startSparsity:m),'m--','Linewidth',2,'MarkerSize',10);
        hold on;
    end
    hold off;
    titleString = sprintf('b) MNIST: Objective value');
    title(titleString,'fontsize',30,'fontweight','bold');
    xlabel('Sparsity level (m)','fontsize',30,'fontweight','bold');
    ylabel('Objective value','fontsize',30,'fontweight','bold');
    if(plotBudget && runBudget)
        if(runGreedy)
            leghan1 = legend('PrDash','L2C','RndW','K-Med','P-Las','PrGrdy');
        else
            leghan1 = legend('PrDash','L2C','RndW','K-Med','P-Las');
        end
    else
        if(runGreedy)
            leghan1 = legend('PrDash','L2C','L2C-A','RndW','K-Med','PrGrdy');
        else
            leghan1 = legend('PrDash','L2C','L2C-A','RndW','K-Med');
        end
    end
    %leghan1 = legend('PrDash (k=1)','PrDash (k=5)','PrDash (k=10)','PrDash (k=20)','PrDash (k=50)','PrDash (k=100)','PrDash (k=200)','L2C','RndW','K-Med');
    set(gca,'fontsize',30,'fontweight','bold');
    set(leghan1,'fontsize',22,'fontweight','bold');
    if(saveOutput)
        saveas(gcf,saveFigFileName,'jpeg');
        saveas(gcf,saveFigFileName);
    end
end