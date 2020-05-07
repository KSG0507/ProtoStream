runCode = false;
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
            m = Originalm;
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
        else
            [~,IX] = sort(stageWeights(1:m,m),'ascend');S_H = S_HT(IX(1:m));
        end
        img_H(:,:,labelNum+1) = X(:,S_H(1:5)); 
        
        fprintf('Running Learn2Criticize to select prototypes\n');
        [w_L,S_LT,~] = Learn2CriticizeSetSelection(X,X,m,kernelType,sigmaVal,meanInnerProductX);
        fprintf('Running Learn2Criticize to select criticisms\n');
        [S_L,~] = L2C_Criticisms(X,X,S_LT,w_L,m,kernelType,sigmaVal,meanInnerProductX);
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
plotPosition = 6;
labelsPlotted = [8,3];
numLabelsPlot = length(labelsPlotted);

for protoIndex = 1:3
    plotPosition = plotPosition + 1;
    for i = 1:numLabelsPlot
        labelNum = labelsPlotted(i);
        plotPosition = plotPosition + 1;
        subplot(4,6,plotPosition)
        imagesc(reshape(img_H(:,protoIndex+1,labelNum+1),28,28)'); axis off;
        hold on;
    end
    plotPosition = plotPosition + 1;
    
    for i = 1:numLabelsPlot
        labelNum = labelsPlotted(i);
        plotPosition = plotPosition + 1;
        subplot(4,6,plotPosition)
        imagesc(reshape(img_L(:,protoIndex+1,labelNum+1),28,28)'); axis off;
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
    [0.104 0.852 0.107 0.104],'String','Criticisms for',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FontSize',30,...
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