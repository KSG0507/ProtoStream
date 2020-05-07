close all; clear all;
dataType = 'Digits';
kernelType = 'Gaussian';
sigmaVal = 5;
algorithmType ='Heuristic';
% Level of sparsity
m = 100;
switch dataType
    case 'random'
        %% Load random data
        X = randn(100,300);
        Y = randn(100,4000);
        normalizeData = true;
        if (normalizeData)
            X = normc(X);
            Y = normc(Y);
        end
    case 'ORL'
        Path = '.\att_faces\';
        [Y,originalData] = get_Processed_ORL_Data(Path);
        X = Y;
    case 'Digits'
        [I_test,labels_test,YT,labels_YT] = readMNIST(60000);
        numSamples = min(10000,length(labels_YT));
        sampleNum = randperm(length(labels_YT));
        Y = YT(:,sampleNum(1:numSamples));
        labels = labels_YT(sampleNum(1:numSamples));
        %locs = labels==5;
        %Y = Y(:,locs);
        locs = labels_test==6;
        X = I_test(:,1:5000);
    otherwise
        %% Load NHANES 2013-2014 Disability data
        fileName = strcat(dataType,'.XPT');
        [Y,originalData] = get_Processed_NHANES_Data(fileName);
        X = Y;
end
fprintf('Computing the vector meanInnerProductX...\n');
meanInnerProductX = computeMeanInnerProductX(X,Y,kernelType,sigmaVal,'faster');
fprintf('Running the selected algorithm..\n');
switch algorithmType
    case 'Greedy'
        %% Perform greedy selection
        [w,S,setValues] = GreedySetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX);
    case 'Heuristic'
        [w,S,setValues,stageWeights,stageGradients] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX);
    case 'Learn2Criticize'
        [w,S,setValues] = Learn2CriticizeSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX);
    case 'Random'
        [w_RU,setValues_RU,w_RE,setValues_RE,S] = RandomSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX);
        w = w_RU;
        setValues = setValues_RU;
    case 'Kmedoids'
        [w_MU,setValues_MU,w_ME,setValues_ME,~] = KmedoidsSetSelection(X,Y,m,kernelType,sigmaVal);
        w = w_ME;
        setValues = setValues_ME;
    case 'Budget'
        l1bound = 0.8;
        maxValue = l1bound/m;
        [w,S,setValues,allw] = SVMBudgetSetSelection(X,Y,m,kernelType,maxValue,sigmaVal,meanInnerProductX);
        algorithmType = strcat(algorithmType,'_',num2str(l1bound));
        %% Plot the raw weights
        figure(201);
        plot(allw,'ro');
        title('Magnitude of the estimated weights','fontsize',24,'fontweight','bold');
        xlabel('Weight Index','fontsize',20,'fontweight','bold');
        ylabel('Weights','fontsize',20,'fontweight','bold');
        set(gca,'fontsize',20,'fontweight','bold');
        rawWeightFileName = strcat('RawW_',dataType,'_',kernelType,'_',algorithmType,'_',num2str(m),'.jpg');
        saveas(gcf,rawWeightFileName);
end
% figure(101);
% plot(w,'b-o','Linewidth',2,'MarkerSize',10);
% title('Magnitude of the estimated weights','fontsize',24,'fontweight','bold');
% xlabel('Weight Index','fontsize',20,'fontweight','bold');
% ylabel('Weights','fontsize',20,'fontweight','bold');
% set(gca,'fontsize',20,'fontweight','bold');
% weightFileName = strcat('W_',dataType,'_',kernelType,'_',algorithmType,'_',num2str(m),'.jpg');
% saveas(gcf,weightFileName);
% %%
% figure(102);
% plot(setValues,'b-o','Linewidth',2,'MarkerSize',10);
% title('Values of the set function over iteration','fontsize',24,'fontweight','bold');
% xlabel('Iteration','fontsize',20,'fontweight','bold');
% ylabel('Set value','fontsize',20,'fontweight','bold');
% set(gca,'fontsize',20,'fontweight','bold');
% setValueFileName = strcat('SetValue_',dataType,'_',kernelType,'_',algorithmType,'_',num2str(m),'.jpg');
% saveas(gcf,setValueFileName);
% %%
% figure(103);
% plot(diff(setValues),'b-o','Linewidth',2,'MarkerSize',10);
% title('Difference from the previous set value','fontsize',24,'fontweight','bold');
% xlabel('Iteration','fontsize',20,'fontweight','bold');
% ylabel('Difference','fontsize',20,'fontweight','bold');
% set(gca,'fontsize',20,'fontweight','bold');
% diffFileName = strcat('DiffSetValue_',dataType,'_',kernelType,'_',algorithmType,'_',num2str(m),'.jpg');
% saveas(gcf,diffFileName);

figure(104);
plot(stageGradients(1,:),'b-','Linewidth',2);
hold on;
plot(stageGradients(5,:),'r-','Linewidth',2);
plot(stageGradients(20,:),'k-','Linewidth',2);
plot(stageGradients(50,:),'g-','Linewidth',2);
hold off;
title('Gradient values at different stages  ','fontsize',24,'fontweight','bold');
xlabel('Stage number','fontsize',20,'fontweight','bold');
ylabel('Gradeint values','fontsize',20,'fontweight','bold');
set(gca,'fontsize',20,'fontweight','bold');
% diffFileName = strcat('DiffSetValue_',dataType,'_',kernelType,'_',algorithmType,'_',num2str(m),'.jpg');
% saveas(gcf,diffFileName);

%%
% for i = 1:m
%     figure(104);
%     imagesc(reshape(Y(:,S(i)),28,28)'); axis image; colormap('gray');
%     pause;
% end
