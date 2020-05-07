runCode = true;
kernelType = 'Gaussian';
sigmaVal = 5;
name = 'MNIST';
% Level of sparsity
m = 200;
kValues = 200;
numkValues = length(kValues);
%proportions = [10 20 30 40 50 60 70 80 90 100];
proportions = [10 30 50 70 90 100];
numProp = length(proportions);
numRuns = length(proportions)*1;
%% Read data
[test,labels_test,YT,labels_YT] = readMNIST(60000);
%% Randomly choose the training data set
numSamples = min(5000,length(labels_YT));
sampleNum = randperm(length(labels_YT));
Y = YT(:,sampleNum(1:numSamples));
labels_Y = labels_YT(sampleNum(1:numSamples));
%%
plotFigure = true; 
saveOutput = false;
%%
outputFileName = 'PrototypeDistributionWithStreaming';
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
    protoLabels_PDS = zeros(numRuns,m);
    %% Run other algorithms exactly once
    r = 0;
    for labelNum = 0:0
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
            %% Run ProtoDash to choose prototypes from Y that best represents X
            fprintf('Running ProtoDash for across prototype selection...\n');
            for kcount = 1:numkValues
                k = kValues(kcount);
                [~,S_HX,~,~] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX,k);
                protoLabels_HX(r,1:m,kcount) = labels_Y(S_HX(1:m));
            end
             [~,S_HS] = ProtoDashStreamingWithThreshold(X,Y,m,kernelType,0.015,sigmaVal,meanInnerProductX);
             %[~,S_HS,~,~] = ProtoDashStreaming(X,Y,m,kernelType,sigmaVal,meanInnerProductX);
             protoLabels_PDS(r,1:m) = labels_Y(S_HS(1:m));
            if(saveOutput)
                deleteFileName = strcat('Variables_',outputFileName,'.mat');
                delete(deleteFileName);
                save(strcat('Variables_',outputFileName),'protoLabels_HX','protoLabels_PDS');
            end
        end
    end
else
    load(strcat('Variables_',outputFileName));
end
if(plotFigure)
    plottingLabel = 0;
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
            subplot(2,1,kcount);
            histogram(categorical(protoLabels_HX(p+plottingLabel*numProp,1:m,kcount)));
            titleString = sprintf('k = %s',num2str(kValues(kcount)));
            title(titleString,'fontsize',axesFontSize,'fontweight','bold');
            axis tight;
            %set(gca,'fontsize',15,'fontweight','bold');
%             xlabel('Labels','fontsize',axesFontSize,'fontweight','bold');
%             ylabel('Count','fontsize',axesFontSize,'fontweight','bold');
        end
        subplot(2,1,2);
        histogram(categorical(protoLabels_PDS(p+plottingLabel*numProp,1:m)));
        titleString = sprintf('k = %s',num2str(kValues(kcount)));
        title(titleString,'fontsize',axesFontSize,'fontweight','bold');
        axis tight;
        %set(gca,'fontsize',15,'fontweight','bold');
%       xlabel('Labels','fontsize',axesFontSize,'fontweight','bold');
%       ylabel('Count','fontsize',axesFontSize,'fontweight','bold');
        if(saveOutput)
            saveFigFileName = strcat(outputFileName,'_p',num2str(proportions(p)));
            saveas(gcf,saveFigFileName,'jpeg');
        end
    end
end