axesFontSize = 30;
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
%proportions = [10 20 30 40 50 60 70 80 90 100];
proportions = [10 30 50 70 90 100];
numRuns = length(proportions)*10;
numSamples = 1500;
zstar = 2.228/sqrt(10);
%%
plotBudget = true;
runBudget = true;
runGreedy = true;
saveOutput = false;
%%
outputFileName = 'TestSetMatchingAccuracy';
if(runGreedy)
    outputFileName = strcat(outputFileName,'WithGreedyAndAdaptedL2C');
else
     outputFileName = strcat(outputFileName,'WithAdaptedL2C');
end
outputFileName = strcat(outputFileName,'_',name,'_m',num2str(m),'_I',num2str(numSamples),'_K_',kernelType);
if(strcmp(kernelType,'Gaussian'))
    outputFileName = strcat(outputFileName,'_sigma',num2str(sigmaVal));
end
fprintf('Output file: %s\n',outputFileName);
%%
load(strcat('Variables_',outputFileName));
%%
a_HXMean = mean(a_HX,1);
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
plot(startSparsity:m,a_HXMean,'g-','Linewidth',2);
hold on;
plot(startSparsity:m,a_LMean,'b-.','Linewidth',2,'MarkerSize',10);
%plot(startSparsity:m,a_LAMean,'b-','Linewidth',2,'MarkerSize',10);
plot(startSparsity:m,a_RMean,'k-','Linewidth',2,'MarkerSize',10);
plot(sparsityLevels4KM,a_MMean,'r-.s','Linewidth',2,'MarkerSize',10);
if(plotBudget && runBudget)
    plot(startSparsity:m,a_BMean,'c-','Linewidth',2,'MarkerSize',10);
end
if(runGreedy)
    plot(startSparsity:m,a_GMean,'m--','Linewidth',2,'MarkerSize',10);
end
hold off;
titleString = 'a) MNIST: Performance';
title(titleString,'fontsize',axesFontSize,'fontweight','bold');
xlabel('Sparsity level (m)','fontsize',axesFontSize,'fontweight','bold');
ylabel('Classification accuracy (%)','fontsize',axesFontSize,'fontweight','bold');
if(plotBudget && runBudget)
    if(runGreedy)
        leghan1 = legend('PrDash','L2C','RndW','K-Med','P-Las','PrGrdy');
    else
        legend('PrDash','L2C','L2C-A','RndW','K-Med','P-Las');
    end
else
    if(runGreedy)
        legend('PrDash','L2C','L2C-A','RndW','K-Med','PrGrdy');
    else
        legend('PrDash','L2C','L2C-A','RndW','K-Med');
    end
end
set(gca,'fontsize',axesFontSize,'fontweight','bold');
set(leghan1,'fontsize',22,'fontweight','bold');
if(saveOutput)
    saveas(gcf,outputFileName,'jpeg');
    saveas(gcf,outputFileName);
end
%% Determine accuracy for each proportions
plotErrorBar = false;
spLevelForB = numLevelsSparsity;
numProp = length(proportions);
a_PH = zeros(numProp,1);
a_PL = zeros(numProp,1);
a_PLA = zeros(numProp,1);
a_PR = zeros(numProp,1);
a_PM = zeros(numProp,1);
if(plotErrorBar)
    a_PHCI = zeros(numProp,1);
    a_PLCI = zeros(numProp,1);
    a_PLACI = zeros(numProp,1);
    a_PRCI = zeros(numProp,1);
    a_PMCI = zeros(numProp,1);
end
if(runBudget)
    a_PB = zeros(numProp,1);
    if(plotErrorBar)
        a_PBCI = zeros(numProp,1);
    end
end
if(runGreedy)
    a_PG = zeros(numProp,1);
    if(plotErrorBar)
        a_PGCI = zeros(numProp,1);
    end
end
for p = 1:numProp
    a_PH(p) = mean(a_HX(p:numProp:numRuns,end));
    a_PL(p) = mean(a_L(p:numProp:numRuns,end));
    a_PLA(p) = mean(a_LA(p:numProp:numRuns,end));
    a_PR(p) = mean(a_R(p:numProp:numRuns,end));
    a_PM(p) = mean(a_M(p:numProp:numRuns,end));
    if(plotErrorBar)
        a_PHCI(p) = zstar*std(a_HX(p:numProp:numRuns,end),1);
        a_PLCI(p) = zstar*std(a_L(p:numProp:numRuns,end),1);
        a_PLACI(p) = zstar*std(a_LA(p:numProp:numRuns,end),1);
        a_PRCI(p) = zstar*std(a_R(p:numProp:numRuns,end),1);
        a_PMCI(p) = zstar*std(a_M(p:numProp:numRuns,end),1);
    end
    if(runBudget)
        accValuesB = a_B(p:numProp:numRuns,spLevelForB);
        nonZeroLoc = accValuesB~=0;
        a_PB(p) = mean(accValuesB(nonZeroLoc));
        if(plotErrorBar)
            a_PBCI(p) = (2.228/sqrt(sum(nonZeroLoc)))*std(accValuesB(nonZeroLoc),1);
        end
    end
    if(runGreedy)
        a_PG(p) = mean(a_G(p:numProp:numRuns,end));
        if(plotErrorBar)
            a_PGCI(p) = zstar*std(a_G(p:numProp:numRuns,end),1);
        end
    end
end
figure(102);
if(plotErrorBar)
    errorbar(proportions,a_PH,a_PHCI,'g--o','Linewidth',2);
    hold on;
    errorbar(proportions,a_PL,a_PLCI,'b-.*','Linewidth',2,'MarkerSize',10);
    errorbar(proportions,a_PLA,a_PLACI,'b--d','Linewidth',2,'MarkerSize',10);
    errorbar(proportions,a_PR,a_PRCI,'k-x','Linewidth',2,'MarkerSize',10);
    errorbar(proportions,a_PM,a_PMCI,'r-.s','Linewidth',2,'MarkerSize',10);
    if(plotBudget && runBudget)
        errorbar(proportions,a_PB,a_PBCI,'c-p','Linewidth',2,'MarkerSize',10);
    end
    if(runGreedy)
       errorbar(proportions,a_PG,a_PGCI,'m--+','Linewidth',2); 
    end
    hold off;
else
    plot(proportions,a_PH,'g--o','Linewidth',2,'MarkerSize',10);
    hold on;
    plot(proportions,a_PL,'b-.*','Linewidth',2,'MarkerSize',10);
    plot(proportions,a_PLA,'b-d','Linewidth',2,'MarkerSize',10);
    plot(proportions,a_PR,'k-x','Linewidth',2,'MarkerSize',10);
    plot(proportions,a_PM,'r-.s','Linewidth',2,'MarkerSize',10);
    if(plotBudget && runBudget)
        plot(proportions,a_PB,'c-p','Linewidth',2,'MarkerSize',10);
    end
    if(runGreedy)
        plot(proportions,a_PG,'m--+','Linewidth',2,'MarkerSize',10);
    end
    hold off;
end
titleString = sprintf('b) MNIST performance');
title(titleString,'fontsize',axesFontSize,'fontweight','bold');
xlabel('% Skew','fontsize',axesFontSize,'fontweight','bold');
ylabel('Classification accuracy (%)','fontsize',axesFontSize,'fontweight','bold');
if(plotBudget && runBudget)
    if(runGreedy)
        leghan = legend('PrDash','L2C','L2C-A','RndW','K-Med','P-Las','PrGrdy');
    else
        leghan = legend('PrDash','L2C','L2C-A','RndW','K-Med','P-Las');
    end
else
    if(runGreedy)
        leghan = legend('PrDash','L2C','L2C-A','RndW','K-Med','PrGrdy');
    else
        leghan = legend('PrDash','L2C','L2C-A','RndW','K-Med');
    end
end
set(gca,'fontsize',axesFontSize,'fontweight','bold');
set(leghan,'fontsize',22,'fontweight','bold');
if(saveOutput)
    saveas(gcf,strcat('Proportions_',outputFileName),'jpeg');
    saveas(gcf,strcat('Proportions_',outputFileName));
end