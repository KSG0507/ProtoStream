runCode = false;
kernelType = 'Gaussian';
sigmaVal = 5;
name = 'MNIST';
% Level of sparsity
m = 100;
%% 
[I_test,labels_test,YT,labels_YT] = readMNIST(60000);
%% Randomly choose the training data set
numSamples = min(1500,length(labels_YT));
sampleNum = randperm(length(labels_YT));
Y = YT(:,sampleNum(1:numSamples));
labels_Y = labels_YT(sampleNum(1:numSamples));
%%
saveOutput = true;
runBudget = false;
runGreedy = false;
%%
testDigit = zeros(size(Y,1),10);
img_H1 = zeros(size(Y,1),5,10);
img_H2 = zeros(size(Y,1),5,10);
img_H3 = zeros(size(Y,1),5,10);
img_L = zeros(size(Y,1),5,10);
img_R = zeros(size(Y,1),5,10);
img_M = zeros(size(Y,1),5,10);
saveFigFileName='SameDatasetPrototypes_';
if(runGreedy)
    saveFigFileName = strcat(saveFigFileName,'WithGreedy');
end
saveFigFileName = strcat(saveFigFileName,'_',name,'_m',num2str(m),'_I',num2str(numSamples),'_K_',kernelType);
if(strcmp(kernelType,'Gaussian'))
    saveFigFileName = strcat(saveFigFileName,'_sigma',num2str(sigmaVal));
end
topProtoFileName = saveFigFileName;
%%
if(runCode)
    for labelNum = 0:9
        fprintf('Label number = %d\n',labelNum);
        %name = strcat('MNIST-',num2str(labelNum));
        locs = labels_test==labelNum;
        X = I_test(:,locs);
        testDigit(:,labelNum+1) = X(:,1);
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
        [w_HT,S_HT,setValues_H,stageWeights] = HeuristicSetSelection(X,X,3*m,kernelType,sigmaVal,meanInnerProductX);
        S_HT1 = S_HT(1:m); [~,IX] = sort(stageWeights(1:m,m),'descend');S_H1 = S_HT1(IX(1:m));
        S_HT2 = S_HT(1:2*m); [~,IX] = sort(stageWeights(1:2*m,2*m),'descend');S_H2 = S_HT2(IX(1:m));
        S_HT3 = S_HT(1:3*m); [~,IX] = sort(stageWeights(1:3*m,3*m),'descend');S_H3 = S_HT3(IX(1:m));
        img_H1(:,:,labelNum+1) = X(:,S_H1(1:5));
        img_H2(:,:,labelNum+1) = X(:,S_H2(1:5));
        img_H3(:,:,labelNum+1) = X(:,S_H3(1:5));
            
        %S_HT4 = S_HT(1:4*m); [~,IX] = sort(stageWeights(1:4*m,4*m),'descend');S_H4 = S_HT4(IX(1:m));
        %%
        fprintf('Running Learn2Criticize\n');
        [w_L,S_L,setValues_L] = Learn2CriticizeSetSelection(X,X,m,kernelType,sigmaVal,meanInnerProductX);
        img_L(:,:,labelNum+1) =  X(:,S_L(1:5));

%         fprintf('Running Random\n');
%         [w_RU,setValues_RU,w_RE,setValues_RE,S_RT] = RandomSetSelection(Y,Y,3*m,kernelType,sigmaVal,meanInnerProductY);
%         [~,IX] = sort(w_RU,'descend');S_R = S_RT(IX(1:m));
%         img_R(:,:,labelNum+1) = X(:,S_R(1:5));

        fprintf('Running K-medoids\n');
        [w_MU,setValues_MU,w_ME,setValues_ME,S_MT] = KmedoidsSetSelection(X,X,m,kernelType,sigmaVal,m);
        fprintf('Number of weights calculated in K-medoids is %d\n',length(w_MU));
        [~,IX] = sort(w_MU,'descend');S_M = S_MT(IX(1:m));
        img_M(:,:,labelNum+1) = X(:,S_M(1:5));
        
        if(runGreedy)
            fprintf('Running Greedy\n');
            [w_G,S_GT,setValues_G,stageWeights_G] = GreedySetSelection(X,Y,2*m,kernelType,sigmaVal,meanInnerProductX);
            S_G = S_GT(1:m);
            [~,IX] = sort(stageWeights_G(1:2*m,2*m),'descend');S_G2 = S_GT(IX(1:m));
        end
        if(saveOutput)
            deleteFileName = strcat(topProtoFileName,'.mat');
            delete(deleteFileName);
            save(topProtoFileName, 'testDigit', 'img_H1','img_H2','img_H3','img_L','img_M');
        end
    end
else
    load (strcat(topProtoFileName,'.mat'));
end

figure(2);
plotPosition = 6;
labelsPlotted = [8,3];
numLabelsPlot = length(labelsPlotted);
% for i = 1:numLabelsPlot
%     plotPosition = plotPosition + 1;
%     labelNum = labelsPlotted(i);
%     subplot(6,5,plotPosition);
%     imagesc(reshape(testDigit(:,labelNum+1),28,28)'); axis off;
%     hold on;
% end
% plotPosition = plotPosition + 1;
% 
% for i = 1:numLabelsPlot
%     plotPosition = plotPosition + 1;
%     labelNum = labelsPlotted(i);
%     subplot(6,5,plotPosition);
%     imagesc(reshape(testDigit(:,labelNum+1),28,28)'); axis off;
%     hold on;
% end

for protoIndex = 1:3
    plotPosition = plotPosition + 1;
    for i = 1:numLabelsPlot
        labelNum = labelsPlotted(i);
        plotPosition = plotPosition + 1;
        subplot(4,6,plotPosition)
        imagesc(reshape(img_H1(:,protoIndex,labelNum+1),28,28)'); axis off;
        hold on;
    end
    plotPosition = plotPosition + 1;
    
    for i = 1:numLabelsPlot
        labelNum = labelsPlotted(i);
        plotPosition = plotPosition + 1;
        subplot(4,6,plotPosition)
        imagesc(reshape(img_L(:,protoIndex,labelNum+1),28,28)'); axis off;
        hold on;
    end
end
figure2 = gcf;
annotation(figure2,'line',[0.10 0.94],...
    [0.998 0.998],'LineWidth',2);
annotation(figure2,'line',[0.10 0.94],...
    [0.81 0.81],'LineWidth',2);
annotation(figure2,'line',[0.10 0.94],...
    [0.064 0.064],'LineWidth',2);

annotation(figure2,'line',[0.10 0.10],...
    [0.998 0.064],'LineWidth',2);
annotation(figure2,'line',[0.25 0.25],...
    [0.998 0.064],'LineWidth',2);
annotation(figure2,'line',[0.52 0.52],...
    [0.998 0.064],'LineWidth',2);
annotation(figure2,'line',[0.64 0.64],...
    [0.998 0.064],'LineWidth',2);
% annotation(figure2,'line',[0.79 0.79],...
%     [0.998 0.064],'LineWidth',2);
annotation(figure2,'line',[0.94 0.94],...
    [0.998 0.064],'LineWidth',2);


annotation(figure2,'textbox',...
    [0.104 0.852 0.107 0.104],'String','Prototypes for',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FontSize',36,...
    'FitBoxToText','off');

annotation(figure2,'textbox',...
    [0.293 0.849 0.029 0.086],...
    'String',num2str(labelsPlotted(1)),...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FontSize',36,...
    'FitBoxToText','off');

annotation(figure2,'textbox',...
    [0.436 0.851 0.029 0.086],...
    'String',num2str(labelsPlotted(2)),...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FontSize',36,...
    'FitBoxToText','off');

annotation(figure2,'textbox',...
    [0.700 0.854 0.029 0.086],...
    'String',num2str(labelsPlotted(1)),...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FontSize',36,...
    'FitBoxToText','off');


annotation(figure2,'textbox',...
    [0.839 0.856 0.029 0.086],...
    'String',num2str(labelsPlotted(2)),...
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
