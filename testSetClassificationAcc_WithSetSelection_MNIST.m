runCode = false;
runForOneTh = true;
kernelType = 'Gaussian';
sigmaVal = 5;
name = 'MNIST';
% Level of sparsity
m = 750;
kValues = [1];
numkValues = length(kValues);
startSparsity = 50;
numLevelsSparsity = m-startSparsity+1;
numRuns = 1;
%% Read data
[X,labels_X,Y,labels_Y] = readMNIST(60000);
numSamples = numel(labels_Y);
%%
plotFigure = true; 
runGreedyStreaming = true;
saveOutput = true;
%%
outputFileName = 'WithSetSelection_ClassificationAccuracy_Streaming';
if(runGreedyStreaming)
    outputFileName = strcat(outputFileName,'_WithPG');
end
if(runForOneTh)
    outputFileName = strcat(outputFileName,'_1Th');
end
outputFileName = strcat(outputFileName,'_',name,'_m',num2str(m),'_k',num2str(m),'_I',num2str(numSamples),'_K_',kernelType);
if(strcmp(kernelType,'Gaussian'))
    outputFileName = strcat(outputFileName,'_sigma',num2str(sigmaVal));
end
fprintf('Output file: %s\n',outputFileName);
%%
if(runCode)
    a_PD = zeros(numRuns,numLevelsSparsity,numkValues);
    a_PDS = zeros(numRuns,numLevelsSparsity);
    a_RS = zeros(numRuns,numLevelsSparsity);
    a_L = zeros(numRuns,numLevelsSparsity);
    protoLabels_PD = zeros(numRuns,m,numkValues);
    protoLabels_PDS = zeros(numRuns,m);
    protoLabels_RS = zeros(numRuns,m);
    protoLabels_L = zeros(numRuns,m);
    setValue_PD = zeros(numRuns,numkValues);
    setValue_PDS = zeros(1,numRuns);
    setValue_RS = zeros(1,numRuns);
    setValue_L = zeros(1,numRuns);
    setSize_PDS = zeros(1,numRuns);
    S_PD = zeros(m,numkValues);
    if(runGreedyStreaming)
        a_PGS = zeros(numRuns,numLevelsSparsity);
        protoLabels_PGS = zeros(numRuns,m);
        setValue_PGS = zeros(1,numRuns);
        setSize_PGS = zeros(1,numRuns);
    end
    %%
    numX = numel(labels_X);
    fprintf('Size of test data = %d\n',numX);
    %% Computing mean inner product with different data sets
    fprintf('Computing meanInnerProductY...\n');
    meanInnerProductY = computeMeanInnerProductX(Y,Y,kernelType,sigmaVal,'faster');
    %%
    r = 1;
    %% Run ProtoDash to choose prototypes from Y that best represents Y
    classifiedLabels_PD = zeros(size(X,2),numLevelsSparsity,numkValues);
    fprintf('Running ProtoDash for across prototype selection...\n');
    for kcount = 1:numkValues
        k = kValues(kcount);
        if(k==m)
            [w_PD,S_PDT,functionSetValue,~] = ProtoDashStreaming(Y,Y,m,kernelType,sigmaVal,meanInnerProductY);
        else
            [w_PD,S_PDT,functionSetValue,~] = HeuristicSetSelection(Y,Y,m,kernelType,sigmaVal,meanInnerProductY,k);
        end
        setValue_PD(r,kcount) = functionSetValue(end);
        [~,IX] = sort(w_PD,'descend');
        %S_PD(:,kcount) = S_PDT(IX);
        S_PD(:,kcount) = S_PDT;
        protoLabels_PD(r,1:m,kcount) = labels_Y(S_PD(1:m,kcount));
        fprintf('Objective value for ProtoDash  = %f\n',setValue_PD(r,kcount));
        fprintf('Computing classification accuracy for ProtoDash across at all levels of sparsity...\n');
        classifiedLabels_PD(:,:,kcount) = NNC(X,Y(:,S_PD(1:m,kcount)),labels_Y(S_PD(1:m,kcount)),startSparsity);
    end
    %% Run ProtoDash streaming to choose prototypes from Y that best represents Y
    epsilon = 0.4;
    if(runForOneTh)
        currTh = 0.000102;
        drLowerBound = m*(1+epsilon);
        fprintf('Running ProtoStream only once for a given threshold = %f\n',currTh);
        [w_PDS, S_PDST, setValue_PDS(r)] = ProtoDashStreamingWithThreshold_Variation2(Y,Y,m,kernelType,currTh,sigmaVal,meanInnerProductY,false,drLowerBound);
    else
        [w_PDS, S_PDST, setValue_PDS(r)] = ProtoDashStreamingWithMultipleThreshold(Y,Y,m,kernelType,epsilon,sigmaVal,meanInnerProductY);
    end
    [~,IX] = sort(w_PDS,'descend');
    %S_PDS = S_PDST(IX);
    S_PDS = S_PDST;
    setSize_PDS(r) = min(m,numel(S_PDS));
    spLevel_PDS = setSize_PDS(r)-startSparsity+1;
    protoLabels_PDS(r,1:setSize_PDS(r)) = labels_Y(S_PDS(1:setSize_PDS(r)));
    fprintf('Size of the chosen set in ProtoDash streaming = %d\n',setSize_PDS(r));
    classifiedLabels_PDS = NNC(X,Y(:,S_PDS(1:setSize_PDS(r))),labels_Y(S_PDS(1:setSize_PDS(r))),startSparsity);
    %% Run random streaming to choose prototypes from Y that best represents Y
    [w_RSU,setValues_RSU,w_RSE,setValues_RSE,S_RST] = RandomStreamingSetSelection(Y,Y,m,kernelType,sigmaVal,meanInnerProductY);
    setValue_RS(r) = setValues_RSU(end);
    [~,IX] = sort(w_RSU,'descend');
    S_RS = S_RST(IX);
    protoLabels_RS(r,:) = labels_Y(S_RS(1:m));
    fprintf('Computing classification accuracy for random across at all levels of sparsity...\n');
    classifiedLabels_RS = NNC(X,Y(:,S_RS(1:m)),protoLabels_RS(r,:),startSparsity);
    %% Run L2C adapted to choose prototypes from Y that best represents X
    fprintf('Running L2C adapted for within prototype selection...\n');
    [w_L,S_L,setValues_L] = Learn2CriticizeSetSelection(Y,Y,m,kernelType,sigmaVal,meanInnerProductY);
    setValue_L(r) = setValues_L(end);
    protoLabels_L(r,:) = labels_Y(S_L(1:m));
    fprintf('Objective value for L2C  = %f\n',setValue_L(r));
    fprintf('Computing classification accuracy for L2C across at all levels of sparsity...\n');
    classifiedLabels_L = NNC(X,Y(:,S_L),labels_Y(S_L),startSparsity);
    %%
    if(runGreedyStreaming)
        if(runForOneTh)
            currTh = 0.000328;
            epsilon = 0.4;
            drLowerBound = 9*(m)*(1+epsilon);
            fprintf('Running Streak only once for a given threshold = %f\n',currTh);
            [w_PGS, S_PGST, setValue_PGS(r)] = ProtoGreedyStreamingWithThreshold_Variation2(Y,Y,m,kernelType,currTh,sigmaVal,meanInnerProductY,false,drLowerBound);
        else
            [w_PGS, S_PGST, setValue_PGS(r)] = ProtoGreedyStreamingWithMultipleThreshold(Y,Y,m,kernelType,epsilon,sigmaVal,meanInnerProductY);
        end
        [~,IX] = sort(w_PGS,'descend');
        %S_PGS = S_PGST(IX);
        S_PGS = S_PGST;
        setSize_PGS(r) = min(m,numel(S_PGS));
        spLevel_PGS = setSize_PGS(r)-startSparsity+1;
        protoLabels_PGS(r,1:setSize_PGS(r)) = labels_Y(S_PGS(1:setSize_PGS(r)));
        fprintf('Size of the chosen set in ProtoGreedy streaming = %d\n',setSize_PGS(r));
        classifiedLabels_PGS = NNC(X,Y(:,S_PGS(1:setSize_PGS(r))),labels_Y(S_PGS(1:setSize_PGS(r))),startSparsity);
    end
    %%
    for levelNum = 1:numLevelsSparsity
        for kcount = 1:numkValues
            a_PD(r,levelNum,kcount) = (sum(classifiedLabels_PD(:,levelNum,kcount)==labels_X(:))/numX)*100;
        end
        if(levelNum <= spLevel_PDS)
            a_PDS(r,levelNum) = (sum(classifiedLabels_PDS(:,levelNum)==labels_X(:))/numX)*100;
        end
        a_RS(r,levelNum) = (sum(classifiedLabels_RS(:,levelNum)==labels_X(:))/numX)*100;
        a_L(r,levelNum) = (sum(classifiedLabels_L(:,levelNum)==labels_X(:))/numX)*100;
        if(runGreedyStreaming)
            if(levelNum <= spLevel_PGS)
                a_PGS(r,levelNum) = (sum(classifiedLabels_PGS(:,levelNum)==labels_X(:))/numX)*100;
            end
        end
    end
    %%
    if(saveOutput)
        deleteFileName = strcat('Variables_',outputFileName,'.mat');
        delete(deleteFileName);
        if(runGreedyStreaming)
            save(strcat('Variables_',outputFileName),'a_PD','a_PDS','protoLabels_PD','protoLabels_PDS','setValue_PD','setValue_PDS','setSize_PDS','a_PGS','protoLabels_PGS','setValue_PGS','setSize_PGS','S_PD','S_PDS','S_PGS', 'a_RS','protoLabels_RS','setValue_RS','S_RS','a_L','protoLabels_L','setValue_L','S_L');
        else
            save(strcat('Variables_',outputFileName),'a_PD','a_PDS','protoLabels_PD','protoLabels_PDS','setValue_PD','setValue_PDS','setSize_PDS','S_PD','S_PDS','a_RS','protoLabels_RS','setValue_RS','S_RS','a_L','protoLabels_L','setValue_L','S_L');
        end
    end
else
    load(strcat('Variables_',outputFileName));
end
% outputFileName_2 = 'WithSetSelection_ClassificationAccuracy_Streaming_MNIST_m1000_k1000_I10000_K_Gaussian_sigma5';
% load(strcat('Variables_',outputFileName_2));
if(plotFigure)
    a_PDMean = zeros(numLevelsSparsity,numkValues);
    for kcount = 1:numkValues
        a_PDMean(:,kcount) = mean(a_PD(:,:,kcount),1)';
    end
    a_PDSMean = zeros(1,numLevelsSparsity);
    for i = 1:numLevelsSparsity
        nonZeroLoc = a_PDS(:,i)~=0;
        a_PDSMean(i) = mean(a_PDS(nonZeroLoc,i));
    end
    a_RSMean = mean(a_RS,1);
    a_LMean = mean(a_L,1);
    if(runGreedyStreaming)
        a_PGSMean = zeros(1,numLevelsSparsity);
        for i = 1:numLevelsSparsity
            nonZeroLoc = a_PGS(:,i)~=0;
            a_PGSMean(i) = mean(a_PGS(nonZeroLoc,i));
        end
    end
    figure(101);
    sparsityValues = startSparsity:m;
    startSparsityValuePlot = 150;
    endSparsityValuePlot = 750;
    startLoc = startSparsityValuePlot-startSparsity+1;
    endLoc = endSparsityValuePlot-startSparsity+1;
    spLevel_PDS = min(setSize_PDS)-startSparsity+1;
    plot(sparsityValues(startLoc:endLoc),a_PDMean(startLoc:end,1),'b-','Linewidth',2);
    hold on;
    %plot(sparsityValues(startLoc:endLoc),a_PDMean(startLoc:endLoc,numkValues),'r-.','Linewidth',2);
    %hold on;
    plot(sparsityValues(startLoc:endLoc),a_LMean(startLoc:endLoc),'r-','Linewidth',2);
    plot(sparsityValues(startLoc:min(spLevel_PDS,endLoc)),a_PDSMean(startLoc:min(spLevel_PDS,endLoc)),'g-','Linewidth',2);
    %hold on;
    if(runGreedyStreaming)
        spLevel_PGS = min(setSize_PGS)-startSparsity+1;
        plot(sparsityValues(startLoc:min(spLevel_PGS,endLoc)),a_PGSMean(startLoc:min(spLevel_PGS,endLoc)),'k--','Linewidth',2);
    end
    hold off;
    fontSize = 36;
    titleString = sprintf('Dataset: %s',name);
    title(titleString,'fontsize',fontSize,'fontweight','bold');
    xlabel('Sparsity level (m)','fontsize',fontSize,'fontweight','bold');
    ylabel('Accuracy (%)','fontsize',fontSize,'fontweight','bold');
    if(runGreedyStreaming)
        legend('ProtoDash','MMD-Critic','ProtoStream','Streak');
    else
        legend('ProtoDash','MMD-Critic','ProtoStream');
    end
    set(gca,'fontsize',30,'fontweight','bold');
    axis tight;
%     if(saveOutput)
%         saveas(gcf,outputFileName,'jpeg');
%         saveas(gcf,outputFileName);
%     end
    %%
    axesFontSize = 25;
    fh = figure(102);
%     titleString = sprintf('Label distribution');
%     annotation(fh,'textbox',[0.25 0.96 0.56 0.06],'String',titleString,...
%         'LineStyle','none',...
%         'FontWeight','bold',...
%         'FontSize',22,...
%         'FitBoxToText','off');
    if(runGreedyStreaming)
        numPlots = 5;
    else
        numPlots = 4;
    end
    plotPosition = 1;
    %%
    subplot(numPlots,1,plotPosition);
    histogram(categorical(labels_Y),'Normalization','probability');
    title('True label distribution: MNIST','fontsize',fontSize,'fontweight','bold');
    set(gca,'fontsize',axesFontSize,'fontweight','bold');
    axis tight;
    plotPosition = plotPosition +1;
    %%
    for kcount = 1:1
        subplot(numPlots,1,plotPosition);
        histogram(categorical(protoLabels_PD(1,1:m,kcount)),'Normalization','probability');
        if(kValues(kcount)==1)
            titleString = sprintf('ProtoDash');
        else
            titleString = sprintf('ProtoBasic');
        end
        title(titleString,'fontsize',fontSize,'fontweight','bold');
        set(gca,'fontsize',axesFontSize,'fontweight','bold');
        axis tight;
        ylim([0 0.11]);
        plotPosition = plotPosition +1;
    end
    %%
    subplot(numPlots,1,plotPosition);
    histogram(categorical(protoLabels_L(1,1:m)),'Normalization','probability');
    titleString = sprintf('MMD-Critic');
    title(titleString,'fontsize',fontSize,'fontweight','bold');
    set(gca,'fontsize',axesFontSize,'fontweight','bold');
    axis tight;
    plotPosition = plotPosition +1;
    %%
    subplot(numPlots,1,plotPosition);
    histogram(categorical(protoLabels_PDS(1,1:setSize_PDS(1))),'Normalization','probability');
    titleString = sprintf('ProtoStream');
    title(titleString,'fontsize',fontSize,'fontweight','bold');
    set(gca,'fontsize',axesFontSize,'fontweight','bold');
    axis tight;
    plotPosition = plotPosition +1;
    %%
    if(runGreedyStreaming)
        subplot(numPlots,1,plotPosition);
        histogram(categorical(protoLabels_PGS(1,1:setSize_PGS(1))),'Normalization','probability');
        titleString = sprintf('Streak');
        title(titleString,'fontsize',fontSize,'fontweight','bold');
        set(gca,'fontsize',axesFontSize,'fontweight','bold');
        axis tight;
        plotPosition = plotPosition +1;
    end
    %%
%     figure(103);
%     plotPosition = 4;
%     for protoIndex = 1:5
%         plotPosition = plotPosition + 1;
%         subplot(6,3,plotPosition)
%         imagesc(reshape(Y(:,protoLabels_PD(1,protoIndex,2)),28,28)'); axis off;
%         hold on;
%         plotPosition = plotPosition + 1;
%         subplot(6,3,plotPosition);
%         imagesc(reshape(Y(:,protoLabels_PDS(1,protoIndex)),28,28)'); axis off;
%         hold on;
%         plotPosition = plotPosition + 1;
%         subplot(6,3,plotPosition);
%         imagesc(reshape(Y(:,protoLabels_PGS(1,protoIndex)),28,28)'); axis off;
%         hold on;
%     end
%     figure2 = gcf;
% 
%     annotation(figure2,'line',[0.01 0.94],...
%         [0.998 0.998],'LineWidth',2);
%     annotation(figure2,'line',[0.01 0.94],...
%         [0.81 0.81],'LineWidth',2);
%     annotation(figure2,'line',[0.01 0.94],...
%         [0.064 0.064],'LineWidth',2);
% 
%     annotation(figure2,'line',[0.01 0.01],...
%         [0.998 0.064],'LineWidth',2);
%     annotation(figure2,'line',[0.12 0.12],...
%         [0.998 0.064],'LineWidth',2);
%     annotation(figure2,'line',[0.59 0.59],...
%         [0.998 0.064],'LineWidth',2);
%     annotation(figure2,'line',[0.76 0.76],...
%         [0.998 0.064],'LineWidth',2);
%     annotation(figure2,'line',[0.94 0.94],...
%         [0.998 0.064],'LineWidth',2);
% 
% 
%     annotation(figure2,'textbox',[0.019 0.928436018957346 0.107 0.104],...
%         'String','Target digit',...
%         'LineStyle','none',...
%         'FontWeight','bold',...
%         'FontSize',30,...
%         'FitBoxToText','off');
% 
%     annotation(figure2,'textbox',...
%         [0.171 0.831 0.047 0.123],...
%         'String',num2str(labelsPlotted(1)),...
%         'LineStyle','none',...
%         'FontWeight','bold',...
%         'FontSize',36,...
%         'FitBoxToText','off');
% 
%     annotation(figure2,'textbox',...
%         [0.335 0.831 0.047 0.123],...
%         'String',num2str(labelsPlotted(2)),...
%         'LineStyle','none',...
%         'FontWeight','bold',...
%         'FontSize',36,...
%         'FitBoxToText','off');
% 
%     annotation(figure2,'textbox',...
%         [0.503 0.831 0.047 0.123],...
%         'String',num2str(labelsPlotted(3)),...
%         'LineStyle','none',...
%         'FontWeight','bold',...
%         'FontSize',36,...
%         'FitBoxToText','off');
% 
%     xlim=get(gca,'XLim');
%     ylim=get(gca,'YLim');
%     ht = text(0.01,0.6*ylim(1)+0.4*ylim(2),'ProtoDash');
%     set(ht,'Rotation',90)
%     set(ht,'FontSize',36)
%     set(ht,'FontWeight','bold')
% 
%     annotation(figure2,'textbox',...
%         [0.635 0.849 0.107 0.104],'String',{'L2C'},...
%         'LineStyle','none',...
%         'FontWeight','bold',...
%         'FontSize',36,...
%         'FitBoxToText','off');
% 
%     annotation(figure2,'textbox',...
%         [0.789 0.845 0.107 0.106],...
%         'String','K-Med',...
%         'LineStyle','none',...
%         'FontWeight','bold',...
%         'FontSize',36,...
%         'FitBoxToText','off');
% 
% 
%     hold off;


    figure(104);
    labelsForPlot = 0:9;
    numLabelsPlot = length(labelsForPlot); 
    numProtoPerDigits = 5;
    protosSelected = Y(:,S_PDS);
    for protoLabel = 1:numLabelsPlot
        locs = protoLabels_PDS==labelsForPlot(protoLabel);
        img = protosSelected(:,locs);
        plotPosition = protoLabel+numLabelsPlot;
        for protoIndex = 2:numProtoPerDigits+1
            subplot(numProtoPerDigits+1,numLabelsPlot,plotPosition)
            imagesc(reshape(img(:,protoIndex),28,28)'); axis off; axis tight;
            plotPosition = plotPosition + numLabelsPlot;
        end
    end
end
            
