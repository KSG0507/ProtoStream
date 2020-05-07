runCode = false;
kernelType = 'Gaussian';
sigmaVal = 5;
name = 'UCI';
allLabels = 'A':'Z';
numLabels = length(allLabels);
% Level of sparsity
m = 100;
kValues = [1];
fixTestsetSize = false;
numkValues = length(kValues);
startSparsity = 50;
numLevelsSparsity = m-startSparsity+1;
incrementLevels4KM = 50;
sparsityLevels4KM = startSparsity:incrementLevels4KM:m; 
numLevelsSparsity4KM = length(sparsityLevels4KM);
proportions = [100];
labelPosToRun = [4 7 12 20 25];
numDiffLablesToRun = length(labelPosToRun);
numRuns = length(proportions)*numDiffLablesToRun;
%% Read data
[test,labels_test,Y,labels_Y] = readUCILetters;
numSamples = numel(labels_Y);
%%
plotFigure = true; plotBudget = true;
runBudget = true;
runGreedy = true;
runStreaming = false;
saveOutput = true;
%%
if(runStreaming)
    outputFileName = 'PrototypeAccuracy_Streaming_';
else
    outputFileName = 'PrototypeAccuracy';
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
r = 0;
if(runCode)
    %% Run other algorithms exactly once
    fprintf('Computing meanInnerProductY...\n');
    meanInnerProductY = computeMeanInnerProductX(Y,Y,kernelType,sigmaVal,'faster');
    fprintf('Running L2C...\n');
    [w_L,S_L,setValues_L] = Learn2CriticizeSetSelection(Y,Y,m,kernelType,sigmaVal,meanInnerProductY);
    fprintf('Running Random\n');
    [w_RU,setValues_RU,w_RE,setValues_RE,S_RT] = RandomSetSelection(Y,Y,3*m,kernelType,sigmaVal,meanInnerProductY);
    [~,IX] = sort(w_RU,'descend');S_R = S_RT(IX(1:m));
    fprintf('Running K-medoids\n');
    [w_MU,setValues_MU,w_ME,setValues_ME,S_M] = KmedoidsSetSelection(Y,Y,m,kernelType,sigmaVal,m);
    %%
    a_HX = zeros(numRuns,numkValues);
    a_H2X = zeros(numRuns,1);
    a_H3X = zeros(numRuns,1);
    if(runStreaming)
        a_HS = zeros(numRuns,1);
    end
    a_L = zeros(numRuns,1);
    a_LA = zeros(numRuns,1);
    a_R = zeros(numRuns,1);
    a_M = zeros(numRuns,1);
    if(runBudget)
        a_B = zeros(numRuns,1);
    end
    if(runGreedy)
        a_G = zeros(numRuns,1);
        a_G2 = zeros(numRuns,1);
    end
	for labelCount = 1:numDiffLablesToRun
		labelPos = labelPosToRun(labelCount);
		labelName = allLabels(labelPos);
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
			%%
			if(runBudget)
				fprintf('Running Budget\n');
                for l1bound = 0.45:0.01:0.45
                    individualMaxVal = l1bound/m;
                    [w_B,S_B,setValues_B,allw_B,numNonZero] = SVMBudgetSetSelection(X,Y,m,kernelType,individualMaxVal,sigmaVal,meanInnerProductX ,'Incremental');
                    fprintf('l1bound = %f\tLength = %d\tNum nonzero=%d\n',l1bound,length(S_B),numNonZero);
                end
			end 
			%% Run ProtoDash to choose prototypes from Y that best represents X
			fprintf('Running ProtoDash for across prototype selection...\n');
			for kcount = 1:numkValues
				k = kValues(kcount);
                if(k==m)
					[w_HT,S_HT,setValues_H,stageWeights] = ProtoDashStreaming(X,Y,m,kernelType,sigmaVal,meanInnerProductX);
                else
                    if(k==1)
                        [w_HT,S_HT,setValues_H,stageWeights] = HeuristicSetSelection(X,Y,3*m,kernelType,sigmaVal,meanInnerProductX,k);
                    else
                        [w_HT,S_HT,setValues_H,stageWeights] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX,k);
                    end
                end
                if(k==1)
                    S_H = S_HT(1:m);
                    S_HT2 = S_HT(1:2*m); [~,IX] = sort(stageWeights(1:2*m,2*m),'descend');S_H2X = S_HT2(IX(1:m));
                    S_HT3 = S_HT(1:3*m); [~,IX] = sort(stageWeights(1:3*m,3*m),'descend');S_H3X = S_HT3(IX(1:m));
                    a_HX(r,kcount) = (sum(labels_Y(S_H)==labelName)/length(S_H))*100;
                    a_H2X(r) = (sum(labels_Y(S_H2X)==labelName)/length(S_H2X))*100;
                    a_H3X(r) = (sum(labels_Y(S_H3X)==labelName)/length(S_H3X))*100;
                else
                    S_H = S_HT(1:m);
                    a_HX(r,kcount) = (sum(labels_Y(S_H)==labelName)/length(S_H))*100;
                end
			end
			%% Run ProtoDash streaming to choose prototypes from Y that best represents X
			if(runStreaming)
				epsilon = 0.4;
				[W_HS, S_HS, setValue_HS] = ProtoDashStreamingWithMultipleThreshold(X,Y,m,kernelType,epsilon,sigmaVal,meanInnerProductX);
			end
			%% Run L2C adapted to choose prototypes from Y that best represents X
			fprintf('Running L2C adapted for across prototype selection...\n');
			[w_LA,S_LA,setValues_LA] = Learn2CriticizeSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX);
			%%
			if(runGreedy)
				fprintf('Running ProtoGreedy for across prototype selection...\n');
				[w_G,S_GT,setValues_G,stageWeights_G] = GreedySetSelection(X,Y,2*m,kernelType,sigmaVal,meanInnerProductX);
				S_G = S_GT(1:m);
				[~,IX] = sort(stageWeights_G(1:2*m,2*m),'descend');S_G2 = S_GT(IX(1:m));
            end
            fprintf('Computing prototype selection accuracy...\n');
			a_L(r) = (sum(labels_Y(S_L)==labelName)/length(S_L))*100;
			a_LA(r) = (sum(labels_Y(S_LA)==labelName)/length(S_L))*100;
			a_R(r) = (sum(labels_Y(S_R)==labelName)/length(S_R))*100;
			a_M(r) = (sum(labels_Y(S_M)==labelName)/length(S_M))*100;
			if(runBudget)
				a_B(r) = (sum(labels_Y(S_B)==labelName)/length(S_B))*100;
			end
			if(runGreedy)
				a_G(r) = (sum(labels_Y(S_G)==labelName)/length(S_G))*100;
				a_G2(r) = (sum(labels_Y(S_G2)==labelName)/length(S_G2))*100;
			end
			if(runStreaming)
				a_HS(r) = (sum(labels_Y(S_HS)==labelName)/length(S_HS))*100;
			end
			%%
			if(saveOutput)
				deleteFileName = strcat('Variables_',outputFileName,'.mat');
				delete(deleteFileName);
				if(runBudget)
					if(runGreedy)
						save(strcat('Variables_',outputFileName),'a_HX','a_H2X','a_H3X','a_L','a_LA','a_R','a_M','a_B','a_G','a_G2');
					else
						save(strcat('Variables_',outputFileName),'a_HX','a_H2X','a_H3X','a_HS','a_L','a_LA','a_R','a_M','a_B');
					end
				else
					if(runGreedy)
						save(strcat('Variables_',outputFileName),'a_HX','a_H2X','a_H3X','a_HS','a_L','a_LA','a_R','a_M','a_G','a_G2');
					else
						save(strcat('Variables_',outputFileName),'a_HX','a_H2X','a_H3X','a_HS','a_L','a_LA','a_R','a_M');
					end
				end
			end
		end
	end
else
    load(strcat('Variables_',outputFileName,'.mat'));
end
disp([a_HX(:,1),a_H2X,a_H3X,a_L,a_LA,a_R,a_M,a_B, a_G2]);
if(plotFigure)
    %accValues = [mean(a_H),mean(a_H2X),mean(a_H3X),mean(a_L),mean(a_LA),mean(a_R),mean(a_M),0,mean(a_G2)];
    %accValues = [mean(a_H),mean(a_H2X),mean(a_H3X),mean(a_H5),mean(a_H10),mean(a_H20),mean(a_H50),mean(a_H100),mean(a_H200),mean(a_L),mean(a_LA),mean(a_R),mean(a_M),0];
    accValues = [mean(a_H2X),mean(a_L),mean(a_LA),mean(a_R),mean(a_M), mean(a_B), mean(a_G2)];
    disp(accValues);
    algos = {'PrDash','L2C','L2C-A','RndW','K-Med','P-Las','PrGrdy'};
    %algos = {'PrD-r1','PrD-r2','PrD-r3','L2C','L2C-A','RndW','K-Med','P-Las'};
    axesFontSize = 30;
    p = axes;
    han = bar(p,accValues,0.4);
    x_loc = get(han, 'XData');
    y_height = get(han, 'YData');
    %text(x_loc(8), y_height(8)+0.2,'N/A', 'Color', 'k','HorizontalAlignment','center',...
    %  'VerticalAlignment','bottom','fontsize',axesFontSize,'fontweight','bold');
    titleString = 'd) 100% skew';
    set(gca,'fontsize',14,'fontweight','bold');
    set(gca,'XTickLabels',algos);
    title(titleString,'fontsize',axesFontSize,'fontweight','bold');
    ylabel('Target letter %','fontsize',axesFontSize,'fontweight','bold');
    if(saveOutput)
        saveas(gcf,outputFileName,'jpeg');
        saveas(gcf,outputFileName);
    end
end
 
