function [currOptw,chosenSet,currSetValue,stageWeights,timeConsumed] = ProtoGreedyStreamingWithThreshold_Variation2(X,Y,m,kernelType,th,varargin) 
    if(strcmpi(kernelType,'Gaussian'))
        % Std. Dev. of the kernel if it is Gaussian
        if(~isempty(varargin))
            sigma = varargin{1};
        else
            sigma = 1;
        end
    end
    %% Store the mean inner products with X
    numY = size(Y,2);
    numX = size(X,2);
    remainingElements = 1:numY;
    %% Store the mean inner products with X
    if(nargin > 6)
        meanInnerProductX = varargin{2};
    else
        fprintf('Computing the vector meanInnerProductX...\n');
        meanInnerProductX = zeros(numY,1);
        for i = 1:numY
            switch kernelType
                case 'Gaussian'
                    distX = pdist2(X',Y(:,i)');
                    meanInnerProductX(i) = sum(exp(-distX.^2/(2*sigma^2)))/numX;
                otherwise
                    meanInnerProductX(i) = sum(Y(:,i)'*X)/numX;
            end
        end
    end
    %%
    stopWhenNotReqd = false;
    drLowerBound = 1;
    if(nargin > 8)
        stopWhenNotReqd = varargin{3};
        drLowerBound = varargin{4};
    end
    stopThisInstance = false;
    %% Intialization
    S = zeros(1,m);
    setValues = zeros(1,m);
    sizeS = 0;
    currOptw = [];
    currSetValue = 0;
    %%
    currK = [];
    curru = [];
    stageWeights = zeros(m);
    numRemaining = length(remainingElements);
    maxFuncValue = 0;
    streamingStart = tic;
    for count = 1:numRemaining
        i = remainingElements(count);
        newZ = Y(:,i);
        %% Determine the set value when the current element is added to the null set
        switch kernelType
            case 'Gaussian'
                eleK = 1;
            otherwise
                eleK = newZ'*newZ;
        end
        eleu = meanInnerProductX(i);
        elew = max(eleu/eleK,0);
        currFuncValue = -0.5*eleK*(elew^2) + eleu*elew;
        if(currFuncValue > maxFuncValue)
            maxFuncValue = currFuncValue;  
        end
        if(stopWhenNotReqd && (th < maxFuncValue/drLowerBound))
            stopThisInstance = true;
            break;
        end
        %% Compute the new kernel matrix K
        if(sizeS ==0)
            switch kernelType
                case 'Gaussian'
                    probK = 1;
                otherwise
                    probK = newZ'*newZ;
            end
        else
            innerProduct = zeros(sizeS,1);
            switch kernelType
                case 'Gaussian'
                    selfNorm = 1;
                    for j = 1:sizeS
                        recentlyAdded = Y(:,S(j));
                        distnewZ = norm(recentlyAdded-newZ);
                        innerProduct(j) = exp(-distnewZ.^2/(2*sigma^2));
                    end
                otherwise
                    selfNorm =  newZ'*newZ;
                     for j = 1:sizeS
                        recentlyAdded = Y(:,S(j));
                        innerProduct(j) = recentlyAdded'*newZ;
                    end
            end
            K1 = horzcat(currK,innerProduct(:));
            probK = vertcat(K1,[innerProduct',selfNorm]);
        end
        probu = [curru;meanInnerProductX(i)];
        %% Compute the increment in set value after adding the current element
        if(sizeS==0)
            newCurrOptw = max(probu/probK,0);
            value = 0.5*probK*(newCurrOptw^2) - probu*newCurrOptw;
        else
            gradientVal = meanInnerProductX(i)-currOptw'*innerProduct;
            if(gradientVal<=0)
                newCurrOptw = [currOptw(:);0];
                value = -currSetValue;
            else
                [newCurrOptw,value] = runOptimiser(probK,probu,currOptw,gradientVal);
            end
        end
        incrementSetValue = (-value) - currSetValue;
        %% If the increment is more than a given threshold add the current element to the set S
        if((incrementSetValue >=(th/m)) && sizeS < m)
            currK = probK;
            curru = probu;
            currSetValue = currSetValue + incrementSetValue;
            sizeS = sizeS+1;
            S(sizeS) = i;
            currOptw = newCurrOptw;
            setValues(sizeS) = currSetValue;
            stageWeights(1:sizeS,sizeS) = currOptw(:);
        end
    end
    if(stopWhenNotReqd && stopThisInstance)
        currOptw = [];
        chosenSet = [];
        currSetValue = 0;
        fprintf('Prematurely killing this instance as the threshold used is too low\n');
        return ;
    end
    chosenSet = S(1:sizeS);
    fprintf('End of stream reached. Number of times the threshold was reached = %d\n',sizeS);
    timeConsumed = toc(streamingStart);
    fprintf('Time taken for one pass over the data = %f secs\n',timeConsumed);
    fprintf('Value of the current optimal set = %f for threshold = %f\n',currSetValue,th);
end