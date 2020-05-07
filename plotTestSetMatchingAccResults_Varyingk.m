axesFontSize = 30;
kernelType = 'Gaussian';
sigmaVal = 5;
name = 'MNIST';
% Level of sparsity
m = 200;
kValues = [1 m];
numkValues = length(kValues);
startSparsity = 50;
numLevelsSparsity = m-startSparsity+1;
incrementLevels4KM = 50;
sparsityLevels4KM = startSparsity:incrementLevels4KM:m; 
numLevelsSparsity4KM = length(sparsityLevels4KM);
%proportions = [10 20 30 40 50 60 70 80 90 100];
proportions = [10 30 50 70 90 100];
labelPosToRun = 1:10;
numDiffLables = length(labelPosToRun);
numRuns = length(proportions)*numDiffLables;
numSamples = 5000;
zstar = 2.228/sqrt(10);
%%
plotBudget = false;
runBudget = false;
runGreedy = false;
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
load(strcat('Variables_',outputFileName));
%%
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
% plot(startSparsity:m,a_HXMean(:,3),'g-','Linewidth',2);
% plot(startSparsity:m,a_HXMean(:,4),'g:','Linewidth',2);
% plot(startSparsity:m,a_HXMean(:,5),'m--','Linewidth',2);
% plot(startSparsity:m,a_HXMean(:,6),'m-.','Linewidth',2);
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
		%legend('PrDash (k=1)','PrDash (k=5)','PrDash (k=10)','PrDash (k=20)','PrDash (k=50)','PrDash (k=100)','PrDash Streaming','L2C','L2C-A','RndW','K-Med');
        legend('PrDash','PrDash Non-Th Streaming','PrDash Th Streaming','L2C','L2C-A','RndW','K-Med');
	end
end
set(gca,'fontsize',20,'fontweight','bold');
% if(saveOutput)
% 	saveas(gcf,outputFileName,'jpeg');
% 	saveas(gcf,outputFileName);
% end
%% Determine accuracy for each proportions
spLevelForB = numLevelsSparsity;
numProp = length(proportions);
a_PH = zeros(numProp,numkValues);
if(runStreaming)
	a_PHS = zeros(numProp,1);
end
a_PL = zeros(numProp,1);
a_PLA = zeros(numProp,1);
a_PR = zeros(numProp,1);
a_PM = zeros(numProp,1);
if(runBudget)
    a_PB = zeros(numProp,1);
end
if(runGreedy)
    a_PG = zeros(numProp,1);
end
for p = 1:numProp
	for kcount = 1:numkValues
		a_PH(p,kcount) = mean(a_HX(p:numProp:numRuns,end,kcount));
	end
	if(runStreaming)
		spLevelForStreaming = min(setSize_PDS(p:numProp:numRuns))-startSparsity+1;
		a_PHS(p) = mean(a_HS(p:numProp:numRuns,spLevelForStreaming));
	end
    a_PL(p) = mean(a_L(p:numProp:numRuns,end));
	a_PL(p) = mean(a_L(p:numProp:numRuns,end));
    a_PLA(p) = mean(a_LA(p:numProp:numRuns,end));
    a_PR(p) = mean(a_R(p:numProp:numRuns,end));
    a_PM(p) = mean(a_M(p:numProp:numRuns,end));
    if(runBudget)
        accValuesB = a_B(p:numProp:numRuns,spLevelForB);
        nonZeroLoc = accValuesB~=0;
        a_PB(p) = mean(accValuesB(nonZeroLoc));
    end
    if(runGreedy)
        a_PG(p) = mean(a_G(p:numProp:numRuns,end));
    end
end
figure(102);
plot(proportions,a_PH(:,1),'g--o','Linewidth',2,'MarkerSize',10);
hold on;
plot(proportions,a_PH(:,2),'g-.+','Linewidth',2,'MarkerSize',10);
% plot(proportions,a_PH(:,3),'g-p','Linewidth',2,'MarkerSize',10);
% plot(proportions,a_PH(:,4),'g:*','Linewidth',2,'MarkerSize',10);
% plot(proportions,a_PH(:,5),'m--o','Linewidth',2,'MarkerSize',10);
% plot(proportions,a_PH(:,6),'m-.+','Linewidth',2,'MarkerSize',10);
%plot(proportions,a_PH200,'m-p','Linewidth',2,'MarkerSize',10);
if(runStreaming)
	plot(proportions,a_PHS,'m-p','Linewidth',2,'MarkerSize',10);
end
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
titleString = sprintf('b) MNIST performance');
title(titleString,'fontsize',axesFontSize,'fontweight','bold');
xlabel('% Skew','fontsize',axesFontSize,'fontweight','bold');
ylabel('Classification accuracy (%)','fontsize',axesFontSize,'fontweight','bold');
if(plotBudget && runBudget)
    if(runGreedy)
        leghan = legend('PrDash','L2C','L2C-A','RndW','K-Med','P-Las','PrGrdy');
    else
        leghan = legend('PrDash (k=1)','PrDash (k=5)','PrDash (k=10)','PrDash (k=20)','PrDash (k=50)','PrDash (k=100)','PrDash Streaming','L2C','L2C-A','RndW','K-Med','P-Las');
    end
else
    if(runGreedy)
        leghan = legend('PrDash','L2C','L2C-A','RndW','K-Med','PrGrdy');
    else
        %leghan = legend('PrDash (k=1)','PrDash (k=5)','PrDash (k=10)','PrDash (k=20)','PrDash (k=50)','PrDash (k=100)','PrDash Streaming','L2C','L2C-A','RndW','K-Med');
        leghan = legend('PrDash','PrDash Non-Th Streaming','PrDash Th Streaming','L2C','L2C-A','RndW','K-Med');
    end
end
set(gca,'fontsize',axesFontSize,'fontweight','bold');
set(leghan,'fontsize',22,'fontweight','bold');
if(saveOutput)
    saveas(gcf,strcat('Proportions_',outputFileName),'jpeg');
    saveas(gcf,strcat('Proportions_',outputFileName));
end