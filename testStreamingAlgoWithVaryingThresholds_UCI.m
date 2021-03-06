runCode = false;
kernelType = 'Gaussian';
sigmaVal = 5;
name = 'UCI';
allLabels = 'A':'Z';
numLabels = length(allLabels);
% Level of sparsity
m = 100;
kValues = [1 m];
fixTestsetSize = false;
epsilon = 0.4;
numkValues = length(kValues);
%proportions = [10 20 30 40 50 60 70 80 90 100];
proportions = [100];
numProp = length(proportions);
labelPosToRun = [6 10 16 20 25];
numDiffLablesToRun = length(labelPosToRun);
numRuns = length(proportions)*numDiffLablesToRun;
%% Read data
[test,labels_test,Y,labels_Y] = readUCILetters;
numSamples = numel(labels_Y);
%%
plotFigure = true; 
saveOutput = true;
runGreedyStreaming = true;
%%
if(runGreedyStreaming)
    outputFileName = 'PrototypeDistribution_Streaming_WithPG';
else
    outputFileName = 'PrototypeDistribution_Streaming';
end
outputFileName = strcat(outputFileName,'_',name,'_m',num2str(m),'_k',num2str(m),'_I',num2str(numSamples),'_K_',kernelType);
if(strcmp(kernelType,'Gaussian'))
    outputFileName = strcat(outputFileName,'_sigma',num2str(sigmaVal));
end
fprintf('Output file: %s\n',outputFileName);
%%
if(runCode)
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
    %%
    protoLabels_HX = zeros(numRuns,m,numkValues);
    protoLabels_PDS = zeros(numRuns,m);
    setSize_PDS = zeros(numRuns);
    if(runGreedyStreaming)
        protoLabels_PGS = zeros(numRuns,m);
        setSize_PGS = zeros(numRuns);
    end
    %% Run other algorithms exactly once
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
            %% Run ProtoDash to choose prototypes from Y that best represents X
            fprintf('Running ProtoDash for across prototype selection...\n');
            for kcount = 1:numkValues
                k = kValues(kcount);
                if(k==m)
                    [~,S_HX,~,~] = ProtoDashStreaming(X,Y,m,kernelType,sigmaVal,meanInnerProductX);
                else
                    [~,S_HX,~,~] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX,k);
                end
                protoLabels_HX(r,1:m,kcount) = labels_Y(S_HX(1:m));
            end
            [W_HS, S_HS, setValue_HS] = ProtoDashStreamingWithMultipleThreshold(X,Y,m,kernelType,epsilon,sigmaVal,meanInnerProductX);
            setSize_PDS(r) = min(m,numel(S_HS));
            fprintf('Size of the optimal set in ProtoDash Streaming = %d\n',setSize_PDS(r));
            protoLabels_PDS(r,1:setSize_PDS(r)) = labels_Y(S_HS(1:setSize_PDS(r)));
            if(runGreedyStreaming)
                [W_GS, S_GS, setValue_GS] = ProtoGreedyStreamingWithMultipleThreshold(X,Y,m,kernelType,epsilon,sigmaVal,meanInnerProductX);
                setSize_PGS(r) = min(m,numel(S_GS));
                fprintf('Size of the optimal set in ProtoGreedy Streaming = %d\n',setSize_PGS(r));
                protoLabels_PGS(r,1:setSize_PGS(r)) = labels_Y(S_GS(1:setSize_PGS(r)));
            end
           if(saveOutput)
                deleteFileName = strcat('Variables_',outputFileName,'.mat');
                delete(deleteFileName);
                if(runGreedyStreaming)
                    save(strcat('Variables_',outputFileName),'protoLabels_HX','protoLabels_PDS','setSize_PDS','protoLabels_PGS','setSize_PGS');
                else
                    save(strcat('Variables_',outputFileName),'protoLabels_HX','protoLabels_PDS','setSize_PDS');
                end
            end
        end
    end
else
    load(strcat('Variables_',outputFileName));
end
if(plotFigure)
    plottingLabel = 1;
    fontSize = 26;
    axesFontSize = 20;
    for p = 1:numProp
        fh = figure(p);
        titleString = sprintf('Distribution at %s %% skew for letter %c',num2str(proportions(p)),allLabels(labelPosToRun(plottingLabel+1)));
        annotation(fh,'textbox',[0.25 0.96 0.56 0.06],'String',titleString,...
        'LineStyle','none',...
        'FontWeight','bold',...
        'FontSize',26,...
        'FitBoxToText','off');
        if(runGreedyStreaming)
            numPlots = 3;
        else
            numPlots = 2;
        end
        plotPosition = 1;
        for kcount = numkValues:numkValues
            subplot(numPlots,1,plotPosition);
            histHand = histogram(categorical(protoLabels_HX(p+plottingLabel*numProp,1:m,kcount)),'Normalization','probability');
            numCategories = numel(histHand.Categories);
            xTickValues = cell(1,numCategories);
            for catCount = 1:numCategories
                xTickValues{catCount} = char(str2num(histHand.Categories{catCount}));
            end
            if(kValues(kcount)==1)
                titleString = sprintf('ProtoDash');
            else
                titleString = sprintf('ProtoBasic');
            end
            title(titleString,'fontsize',fontSize,'fontweight','bold');
            set(gca,'fontsize',axesFontSize,'fontweight','bold');
            set(gca,'XTickLabels',xTickValues);
            axis tight;
            plotPosition = plotPosition +1;
            %set(gca,'fontsize',15,'fontweight','bold');
%             xlabel('Labels','fontsize',axesFontSize,'fontweight','bold');
%             ylabel('Count','fontsize',axesFontSize,'fontweight','bold');
        end
        subplot(numPlots,1,plotPosition);
        h2 =  histogram(categorical(protoLabels_PDS(p+plottingLabel*numProp,1:setSize_PDS(p+plottingLabel*numProp))),'Normalization','probability');
        numCategories = numel(h2.Categories);
        xTickValues2 = cell(1,numCategories);
        for catCount = 1:numCategories
            xTickValues2{catCount} = char(str2num(h2.Categories{catCount}));
        end
        titleString = sprintf('ProtoStream');
        title(titleString,'fontsize',fontSize,'fontweight','bold');
        set(gca,'fontsize',axesFontSize,'fontweight','bold');
        set(gca,'XTickLabels',xTickValues2);
        axis tight;
        plotPosition = plotPosition +1;
        if(runGreedyStreaming)
            subplot(numPlots,1,plotPosition);
            histHand = histogram(categorical(protoLabels_PGS(p+plottingLabel*numProp,1:setSize_PGS(p+plottingLabel*numProp))),'Normalization','probability');
            numCategories = numel(histHand.Categories);
            xTickValues = cell(1,numCategories);
            for catCount = 1:numCategories
                xTickValues{catCount} = char(str2num(histHand.Categories{catCount}));
            end
            titleString = sprintf('Streak');
            title(titleString,'fontsize',fontSize,'fontweight','bold');
            set(gca,'fontsize',axesFontSize,'fontweight','bold');
            set(gca,'XTickLabels',xTickValues);
            axis tight;
        end
%         if(saveOutput)
%             saveFigFileName = strcat(outputFileName,'_p',num2str(proportions(p)));
%             saveas(gcf,saveFigFileName,'jpeg');
%         end
    end
end