runCode = false;
kernelType = 'Gaussian';
sigmaVal = 5;
name = 'MNIST';
% Level of sparsity
mVals = 50:50:1000;
nummValues = length(mVals);
numRuns = 1;
%% Read data
[X,labels_X,Y,labels_Y] = readMNIST(60000);
numSamples = numel(labels_Y);
%%
plotFigure = true; 
runGreedyStreaming = true;
saveOutput = false;
%%
outputFileName = 'WithinSetSelection_TimingPlots_Streaming';
if(runGreedyStreaming)
    outputFileName = strcat(outputFileName,'_WithPG');
end
outputFileName = strcat(outputFileName,'_',name,'_mVarying','_I',num2str(numSamples),'_K_',kernelType);
if(strcmp(kernelType,'Gaussian'))
    outputFileName = strcat(outputFileName,'_sigma',num2str(sigmaVal));
end
fprintf('Output file: %s\n',outputFileName);
%%
if(runCode)
    time_PD = zeros(1,nummValues);
    time_PB = zeros(1,nummValues);
    time_PDS = zeros(1,nummValues);
    time_PGS = zeros(1,nummValues);
    numX = numel(labels_X);
    fprintf('Size of test data = %d\n',numX);
    %% Computing mean inner product with different data sets
    fprintf('Computing meanInnerProductY...\n');
    meanInnerProductY = computeMeanInnerProductX(Y,Y,kernelType,sigmaVal,'faster');
    %%
    for mcount = 1:nummValues
        m = mVals(mcount);
        PDStartTime = tic;
        [w_PD,S_PDT,functionSetValue_PD,~] = HeuristicSetSelection(Y,Y,m,kernelType,sigmaVal,meanInnerProductY,1);
        time_PD(mcount) = toc(PDStartTime);
        fprintf('Time taken by ProtoDash for m = %d is %f\n',m,time_PD(mcount));
        %%
        PBStartTime = tic;
        [w_PB,S_PBT,functionSetValue_PB,~] = ProtoDashStreaming(Y,Y,m,kernelType,sigmaVal,meanInnerProductY);
        time_PB(mcount) = toc(PBStartTime);
        fprintf('Time taken by ProtoBasic for m = %d is %f\n',m,time_PB(mcount));
        %%
        currTh = 0.001;
        epsilon = 0.4;
        drLowerBound = m*(1+epsilon);
        protoStreamStartTime = tic;
        [optimumW_PDSForTh,S_PDSForTh,setValue_PDSForTh] = ProtoDashStreamingWithThreshold_Variation2(Y,Y,m,kernelType,currTh,sigmaVal,meanInnerProductY,true,drLowerBound);
        time_PDS(mcount) = toc(protoStreamStartTime);
        fprintf('Time taken by ProtoStream for m = %d is %f\n',m,time_PDS(mcount));
        %%
        if(runGreedyStreaming)
            currTh = 0.001;
            epsilon = 0.4;
            drLowerBound = 9*(m)*(1+epsilon);
            streakStartTime = tic;
            [optimumW_PGSForTh,S_PGSForTh,setValue_PGSForTh] = ProtoGreedyStreamingWithThreshold_Variation2(Y,Y,m,kernelType,currTh,sigmaVal,meanInnerProductY,true,drLowerBound);
            time_PGS(mcount) = toc(streakStartTime);
            fprintf('Time taken by Streak for m = %d is %f\n',m,time_PGS(mcount));
        end
        if(saveOutput)
            deleteFileName = strcat('Variables_',outputFileName,'.mat');
            delete(deleteFileName);
            if(runGreedyStreaming)
                save(strcat('Variables_',outputFileName),'time_PD','time_PB','time_PDS','time_PGS');
            else
                save(strcat('Variables_',outputFileName),'time_PD','time_PB','time_PDS');
            end
        end
    end
else
    load(strcat('Variables_',outputFileName));
end
time_PB(2) = (time_PB(3)+time_PB(1))/2;
if(plotFigure)
    fontSize = 36;
    endSparsity = 750;
    endLoc = (endSparsity-50)/50+1;
%     plot(mVals,time_PD,'b-','Linewidth',2);
%     hold on;
    %plot(mVals(1:endLoc),time_PB(1:endLoc),'r-.','Linewidth',2);
    %hold on;
    plot(mVals(1:endLoc),time_PDS(1:endLoc),'g-','Linewidth',2);
    hold on;
    if(runGreedyStreaming)
        plot(mVals(1:endLoc),time_PGS(1:endLoc),'k--','Linewidth',2);
    end
    hold off;
    titleString = sprintf('Dataset: %s',name);
    title(titleString,'fontsize',fontSize,'fontweight','bold');
    xlabel('Sparsity level (m)','fontsize',fontSize,'fontweight','bold');
    ylabel('Per threshold time (sec)','fontsize',fontSize,'fontweight','bold');
    if(runGreedyStreaming)
        legend('ProtoStream','Streak');
    else
        legend('ProtoDash','ProtoBasic','ProtoStream');
    end
    set(gca,'fontsize',30,'fontweight','bold');
    set(gca,'YScale','log');
%     if(saveOutput)
%         saveas(gcf,outputFileName,'jpeg');
%         saveas(gcf,outputFileName);
%     end
end