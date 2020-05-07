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
plotFigure = false;
saveOutput = true;
runBudget = false;
runGreedy = false;
%%
testDigit = zeros(size(Y,1),10);
img_H1 = zeros(size(Y,1),5,10);
img_H2 = zeros(size(Y,1),5,10);
img_H3 = zeros(size(Y,1),5,10);
img_L = zeros(size(Y,1),5);
img_LA = zeros(size(Y,1),5,10);
img_R = zeros(size(Y,1),5);
img_M = zeros(size(Y,1),5);
saveFigFileName='PrototypeAccuracy';
if(runGreedy)
    saveFigFileName = strcat(saveFigFileName,'WithGreedyAndAdaptedL2C');
else
    saveFigFileName = strcat(saveFigFileName,'WithAdaptedL2C');
end
saveFigFileName = strcat(saveFigFileName,'_',name,'_m',num2str(m),'_I',num2str(numSamples),'_K_',kernelType);
if(strcmp(kernelType,'Gaussian'))
    saveFigFileName = strcat(saveFigFileName,'_sigma',num2str(sigmaVal));
end
pAccFileName = strcat('Variables_',saveFigFileName);
topProtoFileName = strcat('TopPrototypes_',saveFigFileName);
fprintf('Output file: %s\n',pAccFileName);
%%
if(runCode)
    a_H = zeros(10,1);
    a_H2 = zeros(10,1);
    a_H3 = zeros(10,1);
    %a_H4 = zeros(10,1);
    a_L = zeros(10,1);
    a_LA = zeros(10,1);
    a_R = zeros(10,1);
    a_M = zeros(10,1);
    if(runBudget)
        a_B = zeros(10,1);
    end
    if(runGreedy)
        a_G = zeros(10,1);
        a_G2 = zeros(10,1);
    end
    for labelNum = 0:9
        fprintf('Label number = %d\n',labelNum);
        %name = strcat('MNIST-',num2str(labelNum));
        locs = labels_test==labelNum;
        X = I_test(:,locs);
        testDigit(:,labelNum+1) = X(:,1);
        fprintf('Computing the vector meanInnerProductX...\n');
        meanInnerProductX = computeMeanInnerProductX(X,Y,kernelType,sigmaVal,'faster');
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
        [w_HT,S_HT,setValues_H,stageWeights] = HeuristicSetSelection(X,Y,3*m,kernelType,sigmaVal,meanInnerProductX);
        S_HT1 = S_HT(1:m); [~,IX] = sort(stageWeights(1:m,m),'descend');S_H1 = S_HT1(IX(1:m));
        S_HT2 = S_HT(1:2*m); [~,IX] = sort(stageWeights(1:2*m,2*m),'descend');S_H2 = S_HT2(IX(1:m));
        S_HT3 = S_HT(1:3*m); [~,IX] = sort(stageWeights(1:3*m,3*m),'descend');S_H3 = S_HT3(IX(1:m));
        img_H1(:,:,labelNum+1) = Y(:,S_H1(1:5));
        img_H2(:,:,labelNum+1) = Y(:,S_H2(1:5));
        img_H3(:,:,labelNum+1) = Y(:,S_H3(1:5));
            
        %S_HT4 = S_HT(1:4*m); [~,IX] = sort(stageWeights(1:4*m,4*m),'descend');S_H4 = S_HT4(IX(1:m));
        %%
        fprintf('Running L2C adapted\n');
        [w_LA,S_LA,setValues_LA] = Learn2CriticizeSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX);
        img_LA(:,:,labelNum+1) = Y(:,S_LA(1:5));
        %%
        if(labelNum == 0)
            fprintf('Computing meanInnerProductY...\n');
            meanInnerProductY = computeMeanInnerProductX(Y,Y,kernelType,sigmaVal,'faster');
            fprintf('Running Learn2Criticize\n');
            [w_L,S_L,setValues_L] = Learn2CriticizeSetSelection(Y,Y,m,kernelType,sigmaVal,meanInnerProductY);
            img_L = Y(:,S_L(1:5));
            
            fprintf('Running Random\n');
            [w_RU,setValues_RU,w_RE,setValues_RE,S_RT] = RandomSetSelection(Y,Y,3*m,kernelType,sigmaVal,meanInnerProductY);
            [~,IX] = sort(w_RU,'descend');S_R = S_RT(IX(1:m));
            img_R = Y(:,S_R(1:5));
            
            fprintf('Running K-medoids\n');
            [w_MU,setValues_MU,w_ME,setValues_ME,S_MT] = KmedoidsSetSelection(Y,Y,m,kernelType,sigmaVal,m);
            fprintf('Number of weights calculated in K-medoids is %d\n',length(w_MU));
            [~,IX] = sort(w_MU,'descend');S_M = S_MT(IX(1:m));
            img_M = Y(:,S_M(1:5));
        end
        if(runGreedy)
            fprintf('Running Greedy\n');
            [w_G,S_GT,setValues_G,stageWeights_G] = GreedySetSelection(X,Y,2*m,kernelType,sigmaVal,meanInnerProductX);
            S_G = S_GT(1:m);
            [~,IX] = sort(stageWeights_G(1:2*m,2*m),'descend');S_G2 = S_GT(IX(1:m));
        end
        a_H(labelNum+1) = (sum(labels_Y(S_H1)==labelNum)/length(S_H1))*100;
        a_H2(labelNum+1) = (sum(labels_Y(S_H2)==labelNum)/length(S_H2))*100;
        a_H3(labelNum+1) = (sum(labels_Y(S_H3)==labelNum)/length(S_H3))*100;
        %a_H4(labelNum+1) = (sum(labels_Y(S_H4)==labelNum)/length(S_H4))*100;
        a_L(labelNum+1) = (sum(labels_Y(S_L)==labelNum)/length(S_L))*100;
        a_LA(labelNum+1) = (sum(labels_Y(S_LA)==labelNum)/length(S_L))*100;
        a_R(labelNum+1) = (sum(labels_Y(S_R)==labelNum)/length(S_R))*100;
        a_M(labelNum+1) = (sum(labels_Y(S_M)==labelNum)/length(S_M))*100;
        if(runBudget)
            a_B(labelNum+1) = (sum(labels_Y(S_B)==labelNum)/length(S_B))*100;
        end
        if(runGreedy)
            a_G(labelNum+1) = (sum(labels_Y(S_G)==labelNum)/length(S_G))*100;
            a_G2(labelNum+1) = (sum(labels_Y(S_G2)==labelNum)/length(S_G2))*100;
        end
        if(saveOutput)
            deleteFileName = strcat(pAccFileName,'.mat');
            delete(deleteFileName);
            deleteFileName = strcat(topProtoFileName,'.mat');
            delete(deleteFileName);
            if(runBudget)
                if(runGreedy)
                    save(pAccFileName,'a_H','a_H2','a_H3','a_L','a_LA','a_R','a_M','a_B','a_G','a_G2');
                else
                    save(pAccFileName,'a_H','a_H2','a_H3','a_L','a_LA','a_R','a_M','a_B');
                end
            else
                if(runGreedy)
                    save(pAccFileName,'a_H','a_H2','a_H3','a_L','a_LA','a_R','a_M','a_G','a_G2');
                else
                    save(pAccFileName,'a_H','a_H2','a_H3','a_L','a_LA','a_R','a_M');
                end
            end
            save(topProtoFileName, 'testDigit', 'img_H1','img_H2','img_H3','img_L','img_LA','img_R','img_M');
        end
    end
else
    %load (strcat(pAccFileName,'.mat'));
    load (strcat(topProtoFileName,'.mat'));
end

figure(2);
plotPosition = 5;
labelsPlotted = [0,3,8];
numLabelsPlot = length(labelsPlotted);

for protoIndex = 1:5
    for i = 1:numLabelsPlot
        labelNum = labelsPlotted(i);
        plotPosition = plotPosition + 1;
        subplot(6,5,plotPosition)
        imagesc(reshape(img_H1(:,protoIndex,labelNum+1),28,28)'); axis off;
        hold on;
    end
    plotPosition = plotPosition + 1;
    subplot(6,5,plotPosition);
    imagesc(reshape(img_L(:,protoIndex),28,28)'); axis off;
    hold on;
    
    plotPosition = plotPosition + 1;
    subplot(6,5,plotPosition);
    imagesc(reshape(img_M(:,protoIndex),28,28)'); axis off;
    hold on;
end
figure2 = gcf;

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
annotation(figure2,'line',[0.59 0.59],...
    [0.998 0.064],'LineWidth',2);
annotation(figure2,'line',[0.76 0.76],...
    [0.998 0.064],'LineWidth',2);
annotation(figure2,'line',[0.94 0.94],...
    [0.998 0.064],'LineWidth',2);


annotation(figure2,'textbox',[0.019 0.928436018957346 0.107 0.104],...
    'String','Target digit',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FontSize',30,...
    'FitBoxToText','off');

annotation(figure2,'textbox',...
    [0.171 0.831 0.047 0.123],...
    'String',num2str(labelsPlotted(1)),...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FontSize',36,...
    'FitBoxToText','off');

annotation(figure2,'textbox',...
    [0.335 0.831 0.047 0.123],...
    'String',num2str(labelsPlotted(2)),...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FontSize',36,...
    'FitBoxToText','off');

annotation(figure2,'textbox',...
    [0.503 0.831 0.047 0.123],...
    'String',num2str(labelsPlotted(3)),...
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

annotation(figure2,'textbox',...
    [0.635 0.849 0.107 0.104],'String',{'L2C'},...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FontSize',36,...
    'FitBoxToText','off');

annotation(figure2,'textbox',...
    [0.789 0.845 0.107 0.106],...
    'String','K-Med',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FontSize',36,...
    'FitBoxToText','off');


hold off;
 
