runCode = true;
kernelType = 'Gaussian';
sigmaVal = 5;
name = 'MNIST';
% Level of sparsity
m = 200;
%% 
[I_test,labels_test,YT,labels_YT] = readMNIST(60000);
%% Randomly choose the training data set
numSamples = min(5000,length(labels_YT));
sampleNum = randperm(length(labels_YT));
Y = YT(:,sampleNum(1:numSamples));
labels_Y = labels_YT(sampleNum(1:numSamples));
%%
plotFigure = false;
saveOutput = true;
runBudget = true;
runGreedy = false;
%%
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
fprintf('Output file: %s\n',pAccFileName);
%%
if(runCode)
    a_H = zeros(10,1);
    %a_H5 = zeros(10,1);
    %a_H10 = zeros(10,1);
    %a_H20 = zeros(10,1);
    %a_H50 = zeros(10,1);
    %a_H100 = zeros(10,1);
    %a_H200 = zeros(10,1);
    a_H2X = zeros(10,1);
    a_H3X = zeros(10,1);
    %a_H4 = zeros(10,1);
    a_L = zeros(10,1);
    a_LA = zeros(10,1);
    a_R = zeros(10,1);
    a_M = zeros(10,1);
    if(runBudget)
        a_B = zeros(10,1);
        setSize_B = zeros(10,1);
    end
    if(runGreedy)
        a_G = zeros(10,1);
        a_G2 = zeros(10,1);
        a_G3 = zeros(10,1);
    end
    for labelNum = 0:9
        fprintf('Label number = %d\n',labelNum);
        %name = strcat('MNIST-',num2str(labelNum));
        locs = labels_test==labelNum;
        X = I_test(:,locs);
        fprintf('Computing the vector meanInnerProductX...\n');
        meanInnerProductX = computeMeanInnerProductX(X,Y,kernelType,sigmaVal,'faster');
        %%
        if(runBudget)
            fprintf('Running Budget\n');
            l1bound = 0.7;
            individualMaxVal = l1bound/m;
            [w_B,S_B,setValues_B,allw_B,numNonZero] = SVMBudgetSetSelection(X,Y,m,kernelType,individualMaxVal,sigmaVal,meanInnerProductX,'Incremental');
            fprintf('l1bound = %f\tLength = %d\tNum nonzero=%d\n',l1bound,length(S_B),numNonZero);
            setSize_B(labelNum+1) = numNonZero;
        end
        %%
        fprintf('Running Heuristic\n');
        [w_HT,S_HT,setValues_H,stageWeights] = HeuristicSetSelection(X,Y,3*m,kernelType,sigmaVal,meanInnerProductX,1);
        w_H = stageWeights(1:m,m);S_H = S_HT(1:m);
        S_HT2 = S_HT(1:2*m); [~,IX] = sort(stageWeights(1:2*m,2*m),'descend');S_H2X = S_HT2(IX(1:m));
        S_HT3 = S_HT(1:3*m); [~,IX] = sort(stageWeights(1:3*m,3*m),'descend');S_H3X = S_HT3(IX(1:m));
        %S_HT4 = S_HT(1:4*m); [~,IX] = sort(stageWeights(1:4*m,4*m),'descend');S_H4 = S_HT4(IX(1:m));
        %[~,S_H5,~,~] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX,5);
        %[~,S_H10,~,~] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX,10);
        %[~,S_H20,~,~] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX,20);
        %[~,S_H50,~,~] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX,50);
        %[~,S_H100,~,~] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX,100);
        %[~,S_H200,~,~] = HeuristicSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX,200);
        %%
        fprintf('Running L2C adapted\n');
        [w_LA,S_LA,setValues_LA] = Learn2CriticizeSetSelection(X,Y,m,kernelType,sigmaVal,meanInnerProductX);
        %%
        if(labelNum == 0)
            fprintf('Computing meanInnerProductY...\n');
            meanInnerProductY = computeMeanInnerProductX(Y,Y,kernelType,sigmaVal,'faster');
            fprintf('Running Learn2Criticize\n');
            [w_L,S_L,setValues_L] = Learn2CriticizeSetSelection(Y,Y,m,kernelType,sigmaVal,meanInnerProductY);
            fprintf('Running Random\n');
            [w_RU,setValues_RU,w_RE,setValues_RE,S_RT] = RandomSetSelection(Y,Y,3*m,kernelType,sigmaVal,meanInnerProductY);
            [~,IX] = sort(w_RU,'descend');S_R = S_RT(IX(1:m));
            fprintf('Running K-medoids\n');
            [w_MU,setValues_MU,w_ME,setValues_ME,S_M] = KmedoidsSetSelection(Y,Y,m,kernelType,sigmaVal,m);
        end
        if(runGreedy)
            fprintf('Running Greedy\n');
            [w_G,S_GT,setValues_G,stageWeights_G] = GreedySetSelection(X,Y,3*m,kernelType,sigmaVal,meanInnerProductX);
            S_G = S_GT(1:m);
            S_GT2 = S_GT(1:2*m); [~,IX] = sort(stageWeights_G(1:2*m,2*m),'descend');S_G2 = S_GT2(IX(1:m));
            S_GT3 = S_GT(1:3*m); [~,IX] = sort(stageWeights_G(1:3*m,3*m),'descend');S_G3 = S_GT3(IX(1:m));
        end
        a_H(labelNum+1) = (sum(labels_Y(S_H)==labelNum)/length(S_H))*100;
        a_H2X(labelNum+1) = (sum(labels_Y(S_H2X)==labelNum)/length(S_H2X))*100;
        a_H3X(labelNum+1) = (sum(labels_Y(S_H3X)==labelNum)/length(S_H3X))*100;
        %a_H4(labelNum+1) = (sum(labels_Y(S_H4)==labelNum)/length(S_H4))*100;
        %%
        %a_H5(labelNum+1) = (sum(labels_Y(S_H5)==labelNum)/length(S_H5))*100;
        %a_H10(labelNum+1) = (sum(labels_Y(S_H10)==labelNum)/length(S_H10))*100;
        %a_H20(labelNum+1) = (sum(labels_Y(S_H20)==labelNum)/length(S_H20))*100;
        %a_H50(labelNum+1) = (sum(labels_Y(S_H50)==labelNum)/length(S_H50))*100;
        %a_H100(labelNum+1) = (sum(labels_Y(S_H100)==labelNum)/length(S_H100))*100;
        %a_H200(labelNum+1) = (sum(labels_Y(S_H200)==labelNum)/length(S_H200))*100;
        %%
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
            a_G3(labelNum+1) = (sum(labels_Y(S_G3)==labelNum)/length(S_G3))*100;
        end
        if(saveOutput)
            deleteFileName = strcat(pAccFileName,'.mat');
            delete(deleteFileName);
            if(runBudget)
                if(runGreedy)
                    save(pAccFileName,'a_H','a_H2X','a_H3X','a_L','a_LA','a_R','a_M','a_B','setSize_B','a_G','a_G2', 'a_G3');
                else
                    save(pAccFileName,'a_H','a_H2X','a_H3X','a_L','a_LA','a_R','a_M','a_B','setSize_B');
                    %save(pAccFileName,'a_H','a_H2X','a_H3X','a_H5','a_H10','a_H20','a_H50','a_H100','a_H200','a_L','a_LA','a_R','a_M','a_B');
                end
            else
                if(runGreedy)
                    save(pAccFileName,'a_H','a_H2','a_H3','a_L','a_LA','a_R','a_M','a_G','a_G2', 'a_G3');
                else
                    save(pAccFileName,'a_H','a_H2','a_H3','a_L','a_LA','a_R','a_M');
                end
            end
        end
    end
else
    %load('Variables_PrototypeAccuracyWithGreedyAndAdaptedL2C_MNIST_m100_I1500_K_Gaussian_sigma5');
    load (strcat(pAccFileName,'.mat'));
end
disp([a_H,a_H2X,a_H3X,a_L,a_LA,a_R,a_M,a_B]);
disp(setSize_B);
%disp([a_H,a_H2X,a_H3X,a_H5,a_H10,a_H20,a_H50,a_H100,a_H200,a_L,a_LA,a_R,a_M,a_B]);
if(plotFigure)
    accValues = [mean(a_H),mean(a_L),mean(a_R),mean(a_M),0,mean(a_G)];
    %accValues = [mean(a_H),mean(a_H2X),mean(a_H3X),mean(a_B), mean(a_G), mean(a_G2), mean(a_G3)];
    %accValues = [mean(a_H),mean(a_H2X),mean(a_H3X),mean(a_H5),mean(a_H10),mean(a_H20),mean(a_H50),mean(a_H100),mean(a_H200),mean(a_L),mean(a_LA),mean(a_R),mean(a_M),0];
    %accValues = [mean(a_H),mean(a_H2X),mean(a_H3X),mean(a_H5),mean(a_H10),mean(a_H20),mean(a_H50),mean(a_H100),mean(a_H200),mean(a_LA),mean(a_M)];
    disp(accValues);
    %algos = {'PrDash','PrDash-2','PrDash-3','P-Las', 'PrGrdy','PrGrdy-2', 'PrGrdy-3'};
    algos = {'PrDash','L2C','RndW','K-Med','P-Las','PrGrdy'};
    %algos = {'PrD-r1','PrD-r2','PrD-r3','PrD-k5','PrD-k10','PrD-k20','PrD-k50','PrD-k100','PrD-k200','L2C-A','K-Med'};
    axesFontSize = 30;
    p = axes;
    han = bar(p,accValues,0.4);
    x_loc = get(han, 'XData');
    y_height = get(han, 'YData');
    text(x_loc(5), y_height(5)+0.2,'N/A', 'Color', 'k','HorizontalAlignment','center',...
      'VerticalAlignment','bottom','fontsize',22,'fontweight','bold');
    titleString = 'c) 100% skew';
    set(gca,'fontsize',20,'fontweight','bold');
    set(gca,'XTickLabels',algos);
    title(titleString,'fontsize',axesFontSize,'fontweight','bold');
    ylabel('Target digit %','fontsize',axesFontSize,'fontweight','bold');
    if(saveOutput)
        saveas(gcf,saveFigFileName,'jpeg');
        saveas(gcf,saveFigFileName);
    end
end

% figure(223);
% plot(1:m,stageWeights(1:m,m),'g-','Linewidth',2,'MarkerSize',10);
% hold on;
% plot(1:2*m,stageWeights(1:2*m,2*m),'b-','Linewidth',2,'MarkerSize',10);
% plot(1:3*m,stageWeights(1:3*m,3*m),'r-.','Linewidth',2,'MarkerSize',10);
% plot(1:4*m,stageWeights(1:4*m,4*m),'k-','Linewidth',2,'MarkerSize',10);
% hold off;
% titleString = 'Weights for different sparsity values for Digit 9';
% title(titleString,'fontsize',24,'fontweight','bold');
% xlabel('Sparsity level','fontsize',20,'fontweight','bold');
% ylabel('Weight value','fontsize',20,'fontweight','bold');
% legend('m=200','m=400','m=600','m=800');
% set(gca,'fontsize',20,'fontweight','bold');
% weightFileName = strcat('StageWeights_Digit-9_m',num2str(m),'_I',num2str(numSamples),'_K_',kernelType,'_sigma',num2str(sigmaVal));
% saveas(gcf,weightFileName,'jpeg');
% saveas(gcf,weightFileName);    
