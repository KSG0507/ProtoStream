function [optimumW_HS, maxS_HS, maxsetValue_HS] = ProtoDashStreamingWithMultipleThreshold(X,Y,m,kernelType,epsilon,varargin)
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
    maxGrad = 0;
    thIntialized = false;
    numTimesRun = 0;
    maxsetValue_HS = 0;
    maxS_HS = [];
    optimumW_HS = [];
    drLowerBound = m*(1+epsilon);
    nrUpperBound = m;
    %% Get the data from a stream
    streamingStartTime = tic;
    thPointOfMax = 0;
    ThSetValueArray = [];
    for j = 1:numY
        currGrad = meanInnerProductX(j);
        if(currGrad > maxGrad)
            maxGrad = currGrad;  
        end
        if(~thIntialized && (currGrad > 0))
            currTh = (currGrad)^2/(2*drLowerBound);
            thIntialized = true;
            thPointOfMax = currTh;
        end
        if(thIntialized && (currTh < (maxGrad)^2/(2*drLowerBound)))
            fprintf('As the current value of the threshold is too low it is been increased\n');
            currTh = (maxGrad)^2/(2*drLowerBound);
        end
        while(thIntialized && (currTh <= (nrUpperBound*(maxGrad)^2/2)))
            numTimesRun = numTimesRun+1;
            fprintf('Calling the streaming algorithm for the time = %d with threshold = %f\n',numTimesRun,currTh);
            %[optimumW_HSForTh,S_HSForTh,setValue_HSForTh] = ProtoDashStreamingWithThreshold(X,Y(:,j:end),m,kernelType,currTh,sigmaVal,meanInnerProductX(j:end),true,drLowerBound);
            [optimumW_HSForTh,S_HSForTh,setValue_HSForTh] = ProtoDashStreamingWithThreshold_Variation2(X,Y(:,j:end),m,kernelType,currTh,sigmaVal,meanInnerProductX(j:end),true,drLowerBound);
            ThSetValueArray(numTimesRun,1) = currTh;
            ThSetValueArray(numTimesRun,2) = setValue_HSForTh;
            if(setValue_HSForTh > maxsetValue_HS)
                maxsetValue_HS = setValue_HSForTh;
                maxS_HS = S_HSForTh;
                optimumW_HS = optimumW_HSForTh;
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