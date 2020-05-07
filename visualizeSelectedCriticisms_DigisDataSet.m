runCode = true;
kernelType = 'Gaussian';
sigmaVal = 5;
name = 'MNIST';
%% 
[I_test,labels_test,YT,labels_YT] = readMNIST(60000);
%% Randomly choose the training data set
numSamples = min(1500,length(labels_YT));
sampleNum = randperm(length(labels_YT));
Y = YT(:,sampleNum(1:numSamples));
labels_Y = labels_YT(sampleNum(1:numSamples));
%%
useL2CMethod = true;
saveOutput = true;
runBudget = false;
runGreedy = false;
%% Level of sparsity
if(useL2CMethod)
    Originalm = 100;
else
    Originalm = numSamples;
end
m = Originalm;
%%
testDigit = zeros(size(Y,1),10);
img_H = zeros(size(Y,1),5,10);
img_L = zeros(size(Y,1),5,10);
img_R = zeros(size(Y,1),5,10);
img_M = zeros(size(Y,1),5,10);
if(useL2CMethod)
    saveFigFileName='CriticismsUsingL2CMethod_';
else
    saveFigFileName='Criticisms_';
end
if(runGreedy)
    saveFigFileName = strcat(saveFigFileName,'WithGreedy');
else
    %saveFigFileName = strcat(saveFigFileName,'WithAdaptedL2C');
end
saveFigFileName = strcat(saveFigFileName,'_',name,'_m',num2str(m),'Temp_I',num2str(numSamples),'_K_',kernelType);
if(strcmp(kernelType,'Gaussian'))
    saveFigFileName = strcat(saveFigFileName,'_sigma',num2str(sigmaVal));
end
criticisms = saveFigFileName;
%%
if(runCode)
    for labelNum = 0:9
        fprintf('Label number = %d\n',labelNum);
        locs = labels_test==labelNum;
        XT = I_test(:,locs);
        if (useL2CMethod)
            X = XT;
            m = 100;
        else
            numSamples = min(1500,sum(locs));
            sampleNum = randperm(sum(locs));
            X = XT(:,sampleNum(1:numSamples));
            m = size(X,2);
        end
        testDigit(:,labelNum+1) = X(:,1);
        
        fprintf('Number of prototypes selected = %d\n',m);
        fprintf('Computing the vector meanInnerProductX...\n');
        meanInnerProductX = computeMeanInnerProductX(X,X,kernelType,sigmaVal,'faster');
        %%
        if(runBudget)
            fprintf('Running Budget\n');
            l1bound = 0.7;
            individualMaxVal = l1bound/m;
            [w_B,S_B,setValues_B,allw_B,numNonZero] = SVMBudgetSetSelection(X,Y,m,kernelType,individualMaxVal,sigmaVal,meanInnerProductX);
            fprintf('l1bound = %f\tLength = %d\tNum nonzero=%d\n',l1bound,length(S_B),numNonZero);
        end
        %%
        fprintf('Running Heuristic\n');
        [w_H,S_HT,setValues_H,stageWeights] = HeuristicSetSelection(X,X,m,kernelType,sigmaVal,meanInnerProductX);
        if(useL2CMethod)
            fprintf('Running Learn2Criticize to select criticisms using ProtoDash prototypes\n');
            [S_H,~] = L2C_Criticisms(X,X,S_HT,w_H,m,kernelType,sigmaVal,meanInnerProductX);
            img_H(:,:,labelNum+1) = X(:,S_H(1:5));
        else
            [~,IX] = sort(stageWeights(1:m,m),'ascend');S_H = S_HT(IX(1:m));
            img_H(:,:,labelNum+1) = X(:,S_H(1:5));
        end
            
        %%
%         fprintf('Running L2C adapted\n');
%         [w_LA,S_LA,setValues_LA] = Learn2CriticizeSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX);
%         img_LA(:,:,labelNum+1) = Y(:,S_LA(1:5));
        %%
        %if(labelNum == 0)
            %fprintf('Computing meanInnerProductY...\n');
            %meanInnerProductY = computeMeanInnerProductX(Y,Y,kernelType,sigmaVal,'faster');
            meanInnerProductY = meanInnerProductX;
            fprintf('Running Learn2Criticize to select prototypes\n');
            [w_L,pLocs_L,~] = Learn2CriticizeSetSelection(X,X,m,kernelType,sigmaVal,meanInnerProductY);
            fprintf('Running Learn2Criticize to select criticisms\n');
            [S_L,~] = L2C_Criticisms(X,X,pLocs_L,w_L,m,kernelType,sigmaVal,meanInnerProductY);
            img_L(:,:,labelNum+1) = X(:,S_L(1:5));
            
%             fprintf('Running Random\n');
%             [w_RU,setValues_RU,w_RE,setValues_RE,S_RT] = RandomSetSelection(X,X,m,kernelType,sigmaVal,meanInnerProductY);
%             [~,IX] = sort(w_RU,'ascend');S_R = S_RT(IX(1:m));
%             img_R(:,:,labelNum+1) = X(:,S_R(1:5));
%             
%             fprintf('Running K-medoids\n');
%             [w_MU,setValues_MU,w_ME,setValues_ME,S_MT] = KmedoidsSetSelection(X,X,m,kernelType,sigmaVal,m);
%             fprintf('Number of weights calculated in K-medoids is %d\n',length(w_MU));
%             [~,IX] = sort(w_MU,'ascend');S_M = S_MT(IX(1:m));
%             img_M(:,:,labelNum+1) = X(:,S_M(1:5));
        %end
        if(runGreedy)
            fprintf('Running Greedy\n');
            [w_G,S_GT,setValues_G,stageWeights_G] = GreedySetSelection(X,Y,2*m,kernelType,sigmaVal,meanInnerProductX);
            S_G = S_GT(1:m);
            [~,IX] = sort(stageWeights_G(1:2*m,2*m),'descend');S_G2 = S_GT(IX(1:m));
        end
        if(saveOutput)
            deleteFileName = strcat(criticisms,'.mat');
            delete(deleteFileName);
            save(criticisms, 'testDigit', 'img_H','img_L','img_R','img_M');
        end
    end
else
    load (strcat(criticisms,'.mat'));
end

figure(2);
plotPosition = 0;
labelsPlotted = [0,7];
numLabelsPlot = length(labelsPlotted);
for i = 1:numLabelsPlot
    plotPosition = plotPosition + 1;
    labelNum = labelsPlotted(i);
    subplot(6,5,plotPosition);
    imagesc(reshape(testDigit(:,labelNum+1),28,28)'); axis off;
    hold on;
end
plotPosition = plotPosition + 1;

for i = 1:numLabelsPlot
    plotPosition = plotPosition + 1;
    labelNum = labelsPlotted(i);
    subplot(6,5,plotPosition);
    imagesc(reshape(testDigit(:,labelNum+1),28,28)'); axis off;
    hold on;
end

for protoIndex = 1:5
    for i = 1:numLabelsPlot
        labelNum = labelsPlotted(i);
        plotPosition = plotPosition + 1;
        subplot(6,5,plotPosition)
        imagesc(reshape(img_H(:,protoIndex,labelNum+1),28,28)'); axis off;
        hold on;
    end
    plotPosition = plotPosition + 1;
    
    for i = 1:numLabelsPlot
        labelNum = labelsPlotted(i);
        plotPosition = plotPosition + 1;
        subplot(6,5,plotPosition)
        imagesc(reshape(img_L(:,protoIndex,labelNum+1),28,28)'); axis off;
        hold on;
    end
end
figure2 = gcf;
%plot([50,50],[0,100],'k-','Linewidth',2);
annotation(figure2,'line',[0.01 0.94],...
    [0.998 0.998],'LineWidth',2);
annotation(figure2,'line',[0.01 0.94],...
    [0.81 0.81],'LineWidth',2);
annotation(figure2,'line',[0.01 0.94],...
    [0.064 0.064],'LineWidth',2);

annotation(figure2,'line',[0.01 0.01],...
    [0.998 0.064],'LineWidth',2);
annotation(figure2,'line',[0.12 0.12],...
    [0.998 0.064],'LineWidth',2);
annotation(figure2,'line',[0.46 0.46],...
    [0.998 0.064],'LineWidth',2);
annotation(figure2,'line',[0.59 0.59],...
    [0.998 0.064],'LineWidth',2);
annotation(figure2,'line',[0.76 0.76],...
    [0.998 0.064],'LineWidth',2);
annotation(figure2,'line',[0.94 0.94],...
    [0.998 0.064],'LineWidth',2);


annotation(figure2,'textbox',[0.019 0.928436018957346 0.107 0.104],...
    'String','Test digit',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FontSize',36,...
    'FitBoxToText','off');


xlim=get(gca,'XLim');
ylim=get(gca,'YLim');
ht = text(0.01,0.6*ylim(1)+0.4*ylim(2),'ProtoDash');
set(ht,'Rotation',90)
set(ht,'FontSize',36)
set(ht,'FontWeight','bold')


htL = text(0.01,0.4*ylim(1)+0.6*ylim(2),'L2C');
set(htL,'Rotation',90)
set(htL,'FontSize',36)
set(htL,'FontWeight','bold')


hold off;
