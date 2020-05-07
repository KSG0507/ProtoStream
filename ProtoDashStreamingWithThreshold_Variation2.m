function [currOptw,chosenSet,currSetValue,stageWeights,stageGradients] = ProtoDashStreamingWithThreshold_Variation2(X,Y,m,kernelType,th,varargin) 
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
    stageGradients = zeros(numY,m);
    numRemaining = length(remainingElements);
    currMaxGrad = 0;
    streamingStart = tic;
    for count = 1:numRemaining
        i = remainingElements(count);
        %%
        if(meanInnerProductX(i) > currMaxGrad)
            currMaxGrad = meanInnerProductX(i);  
        end
        if(stopWhenNotReqd && (th < (currMaxGrad)^2/(2*drLowerBound)))
            stopThisInstance = true;
            break;
        end
        %% Compute the gradient of the incoming element evaluated at the current optimum weight
        newZ = Y(:,i);
        if(sizeS ==0)
            gradientVal = meanInnerProductX(i);
            innerProduct = [];
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
            gradientVal = meanInnerProductX(i)-stageWeights(1:sizeS,sizeS)'*innerProduct;
        end
        %% If the gradient is more than a given threshold add the current element to the set S
        if((gradientVal >=sqrt(2*th/m)) && sizeS < m)
            if (sizeS==0)
                switch kernelType
                    case 'Gaussian'
                        currK = 1;
                    otherwise
                        currK = newZ'*newZ;
                end
                curru = meanInnerProductX(i);
                newCurrOptw = max(curru/currK,0);
                newCurrSetValue = -0.5*currK*(newCurrOptw^2) + curru*newCurrOptw;
            else
                K1 = horzcat(currK,innerProduct(:));
                currK = vertcat(K1,[innerProduct',selfNorm]);
                curru = [curru;meanInnerProductX(i)];
                [newCurrOptw,value] = runOptimiser(currK,curru,currOptw,gradientVal);
                newCurrSetValue = -value;
            end
            sizeS = sizeS+1;
            S(sizeS) = i;
            currOptw = newCurrOptw;
            currSetValue = newCurrSetValue;
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
    fprintf('Time taken for one pass over the data = %f secs\n',toc(streamingStart));
    fprintf('Value of the current optimal set = %f for threshold = %f\n',currSetValue,th);
end