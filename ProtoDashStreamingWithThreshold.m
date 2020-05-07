function [chosenWeights,chosenSet,chosenSetValue,MultipleWeights,MultipleSets,MultipleSetValues,stageWeights,stageGradients] = ProtoDashStreamingWithThreshold(X,Y,m,kernelType,th,varargin) 
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
    currK = [];
    curru = [];
    stageWeights = zeros(m);
    stageGradients = zeros(numY,m);
    numRemaining = length(remainingElements);
    H = cell(1,m);
    H{sizeS+1} = MinHeap(m-sizeS); 
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
        %% For every intermediate set, update the corresponding heap so that it contains the max m-|S| 
        % gradient values
        for j = 0:min(sizeS,m-1)
            if(j==0)
                stageGradientVal = meanInnerProductX(i);
                 %element.innerProduct = [];
            else
                stageGradientVal = meanInnerProductX(i)-stageWeights(1:j,j)'*innerProduct(1:j);
                %element.innerProduct = innerProduct(1:j);
            end
            element.key = i;
            element.value = stageGradientVal;
            element.meanInnProdX = meanInnerProductX(i);
            stageGradients(i,j+1) = stageGradientVal;
            if(H{j+1}.Count() < m-j)
                H{j+1}.InsertKey(element);
            else
                if(H{j+1}.ReturnMin() < element.value)
                    H{j+1}.ReplaceMin(element);
                end
            end
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
            %% Create the heap data structure for the next set S
            if(sizeS < m)
                H{sizeS+1} = MinHeap(m-sizeS);
            end
        end
    end
    MultipleSets = zeros(m,sizeS+1);
    MultipleWeights = zeros(m,sizeS+1);
    MultipleSetValues = zeros(1,sizeS+1);
    chosenSetValue = 0;
    if(stopWhenNotReqd && stopThisInstance)
        chosenWeights = [];
        chosenSet = [];
        fprintf('Prematurely killing this instance as the threshold used is too low\n');
        return ;
    end
    fprintf('End of stream reached. Number of times the threshold was reached = %d\n',sizeS);
    fprintf('Time taken for one pass over the data = %f secs\n',toc(streamingStart));
    timeForOptimalset = tic;
    for j = 0:sizeS
        %% Information about current elements in the set
        baseSet = S(1:j);
        baseuVec = curru(1:j);
        K1 = currK(1:j,1:j);
        %%
        additionalElements = [];
        incrementuVec = [];
        currPos = 0;
        %interInnerProduct = [];
        %%
        if( j < m)
            sortedArray = H{j+1}.Sort();
            incrementSet = zeros(1,m-j);
            incrementuVec = zeros(1,m-j);
%             if(j > 0)
%                 interInnerProduct = zeros(j,m-j);
%             end
            for posIndex = numel(sortedArray):-1:1
                currPos = currPos + 1;
                incrementSet(currPos) = sortedArray{posIndex}.key;
                incrementuVec(currPos) = sortedArray{posIndex}.meanInnProdX;
%                 if(j > 0)
%                     interInnerProduct(:,currPos) = sortedArray{posIndex}.innerProduct;
%                 end
            end
            additionalElements = incrementSet(1:currPos);
        end
        desiredElements = horzcat(baseSet(:)', additionalElements);
        uVec = horzcat(baseuVec(:)', incrementuVec(1:currPos));
        numElementsInSet = j+currPos;
		addedZ = Y(:,additionalElements);
        existingZ = Y(:,baseSet(:)');
        switch kernelType
            case 'Gaussian'
                selfDist = pdist2(addedZ',addedZ');
                selfInnerProduct = exp(-selfDist.^2/(2*sigma^2));
				interDist = pdist2(existingZ',addedZ');
				interInnerProduct = exp(-interDist.^2/(2*sigma^2));
            otherwise
                selfInnerProduct =  addedZ'*addedZ;
				interInnerProduct = existingZ'*addedZ;
        end
        if(j==m)
            K = K1;
        else
            if(j > 0)
                KTop = horzcat(K1,interInnerProduct(:,1:currPos));
                KBottom = horzcat(interInnerProduct(:,1:currPos)',selfInnerProduct);
                K = vertcat(KTop,KBottom);
            else
                K = selfInnerProduct;
            end
        end
        if(max(uVec)<=0)
            newCurrOptw = zeros(m,1);
            newCurrSetValue = 0;
        else
            [newCurrOptw,value] = runOptimiser(K,uVec,[],uVec);
            newCurrSetValue = -value;
        end
        MultipleSets(1:numElementsInSet,j+1) = desiredElements(:);
        MultipleWeights(1:numElementsInSet,j+1) = newCurrOptw;
        MultipleSetValues(j+1) = newCurrSetValue;
        if(j==0 || newCurrSetValue > chosenSetValue)
            chosenSet = MultipleSets(1:numElementsInSet,j+1);
            chosenWeights = MultipleWeights(1:numElementsInSet,j+1);
            chosenSetValue = newCurrSetValue;
        end
    end
    timeTaken = toc(timeForOptimalset);
    fprintf('Time taken to choose the optimal set for this threshold is %f secs \n',timeTaken);
    fprintf('Value of the current optimal set = %f for threshold = %f\n',chosenSetValue,th);
    %plot(MultipleSetValues,'r-');
    %keyboard;
end