function [optimumW_GS, maxS_GS, maxsetValue_GS] = ProtoGreedyStreamingWithMultipleThreshold(X,Y,m,kernelType,epsilon,varargin)
    if(strcmpi(kernelType,'Gaussian'))
        % Std. Dev. of the kernel if it is Gaussian
        if(~isempty(varargin))
            sigmaVal = varargin{1};
        else
            sigmaVal = 1;
        end
    end
    %% Store the mean inner products with X
    numY = size(Y,2);
    numX = size(X,2);
    if(nargin > 6)
        meanInnerProductX = varargin{2};
    else
        fprintf('Computing the vector meanInnerProductX...\n');
        meanInnerProductX = zeros(numY,1);
        for i = 1:numY
            switch kernelType
                case 'Gaussian'
                    distX = pdist2(X',Y(:,i)');
                    meanInnerProductX(i) = sum(exp(-distX.^2/(2*sigmaVal^2)))/numX;
                otherwise
                    meanInnerProductX(i) = sum(Y(:,i)'*X)/numX;
            end
        end
    end
    %% Initialize the variables
    maxFuncValue = 0;
    thIntialized = false;
    numTimesRun = 0;
    maxsetValue_GS = 0;
    maxS_GS = [];
    optimumW_GS = [];
    drLowerBound = 9*(m)*(1+epsilon);
    nrUpperBound = m;
    %% Get the data from a stream
    streamingStartTime = tic;
    thPointOfMax = 0;
    ThSetValueArray = [];
    for j = 1:numY
         switch kernelType
             case 'Gaussian'
                 eleK = 1;
             otherwise
                 eleK = newZ'*newZ;
         end
        eleu = meanInnerProductX(j);
        elew = max(eleu/eleK,0);
        currFuncValue = -0.5*eleK*(elew^2) + eleu*elew;
        if(currFuncValue > maxFuncValue)
            maxFuncValue = currFuncValue;  
        end
        if(~thIntialized && (currFuncValue > 0))
            currTh = currFuncValue/drLowerBound;
            thIntialized = true;
            thPointOfMax = currTh;
        end
        if(thIntialized && (currTh < maxFuncValue/drLowerBound))
            fprintf('As the current value of the threshold is too low it is been increased\n');
            currTh = maxFuncValue/drLowerBound;
        end
        while(thIntialized && (currTh <= nrUpperBound*maxFuncValue))
            numTimesRun = numTimesRun+1;
            fprintf('Calling the streaming algorithm for the time = %d with threshold = %f\n',numTimesRun,currTh);
            [optimumW_GSForTh,S_GSForTh,setValue_GSForTh] = ProtoGreedyStreamingWithThreshold_Variation2(X,Y(:,j:end),m,kernelType,currTh,sigmaVal,meanInnerProductX(j:end),true,drLowerBound);
            ThSetValueArray(numTimesRun,1) = currTh;
            ThSetValueArray(numTimesRun,2) = setValue_GSForTh;
            if(setValue_GSForTh > maxsetValue_GS)
                maxsetValue_GS = setValue_GSForTh;
                maxS_GS = S_GSForTh;
                optimumW_GS = optimumW_GSForTh;
                thPointOfMax = currTh;
            end
            currTh = (1+epsilon)*currTh;
        end
    end
    fprintf('Total number of times the streaming algorithm was instantiated = %d\n',numTimesRun);
    fprintf('Time taken to run for multiple threshold is %f secs \n',toc(streamingStartTime));
    fprintf('Max occured at the threshold value = %f\n',thPointOfMax);
    %%
%     semilogx(ThSetValueArray(:,1), ThSetValueArray(:,2), 'r-o','Linewidth',2);
%     line([min(ThSetValueArray(:,1)), max(ThSetValueArray(:,1))],[maxsetValue_HS, maxsetValue_HS],'Color','k','Linewidth',2);
%     xlabel('Threshold','fontsize',22,'fontweight','bold');
%     ylabel('Function value','fontsize',22,'fontweight','bold');
%     set(gca,'fontsize',22,'fontweight','bold');
%     keyboard;
end