function [currOptw,S,setValues,stageWeights,stageGradients] = HeuristicSetSelection(X,Y,mTemp,kernelType,varargin) 
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
    allY = 1:numY;
    %% Store the mean inner products with X
    if(nargin > 5)
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
    %% Number of selection per iteration
     if(nargin > 6)
        k = varargin{3};
     else
         k = 1;
     end
     fprintf('Choosing %d elements per iteration in ProtoDash\n',k);
    %% Intialization
    m = ceil(mTemp/k)*k;
    S = zeros(1,m);
    timeTaken = zeros(1,ceil(mTemp/k));
    setValues = zeros(1,m);
    sizeS = 0;
    currSetValue = 0;
    currOptw = [];
    currK = [];
    curru = [];
    runningInnerProduct = zeros(m,numY);
    stageWeights = zeros(m);
    stageGradients = zeros(numY,m);
    remainingElements = allY;
    iterNum = 0;
    desiredElements = [];
    start = tic;
    while (sizeS < m)
        iterationTime = tic;
        iterNum = iterNum + 1;
        remainingElements = setdiff(remainingElements,desiredElements);
        numRemaining = length(remainingElements);
        gradientVal = zeros(1,numRemaining);
        if(sizeS >= k)
            innerProducts = zeros(sizeS,numRemaining);
        end
        for count = 1:numRemaining
            i = remainingElements(count);
            newZ = Y(:,i);
            if (sizeS < k)
                u = meanInnerProductX(i);
                gradientVal(count) = u;
                stageGradients(i,k*(iterNum-1)+1:k*iterNum) = gradientVal(count);
            else
                recentlyAdded = Y(:,S(sizeS-k+1:sizeS));
                for j =1:k
                    switch kernelType
                        case 'Gaussian'
                            distnewZ = norm(recentlyAdded(:,j)-newZ);
                            runningInnerProduct(sizeS-k+j,i) = exp(-distnewZ.^2/(2*sigma^2));
                        otherwise
                            runningInnerProduct(sizeS-k+j,i) = recentlyAdded'*newZ;
                    end
                end
                innerProducts(:,count) = runningInnerProduct(1:sizeS,i);
                gradientVal(count) = meanInnerProductX(i)-currOptw'*innerProducts(:,count);
                stageGradients(i,k*(iterNum-1)+1:k*iterNum) = gradientVal(count);
            end
        end
        %% Set up values for next outer iteration (iterNum)
        [sortedValues, IX] = sort(gradientVal,'descend');
        maxGradValues = sortedValues(1:k);
        pos = IX(1:k);
        %[maxGradValues,pos] = maxk(gradientVal,k);
        desiredElements = remainingElements(pos);
        sizeS = sizeS+k;
        S(sizeS-k+1:sizeS) = desiredElements;
        curru = [curru;meanInnerProductX(desiredElements)];
        addedZ = Y(:,desiredElements);
        switch kernelType
            case 'Gaussian'
                selfDist = pdist2(addedZ',addedZ');
                selfInnerProduct = exp(-selfDist.^2/(2*sigma^2));
            otherwise
                selfInnerProduct =  addedZ'*addedZ;
        end
        if(sizeS == k)
            currK = selfInnerProduct;
        else
            newinnerProduct = innerProducts(:,pos);
            K1 = horzcat(currK,newinnerProduct);
            K2 = vertcat(K1,[newinnerProduct',selfInnerProduct]);
            currK = K2;
        end
        if(max(maxGradValues)<=0)
                newCurrOptw = [currOptw(:);zeros(k,1)];
                newCurrSetValue = currSetValue;
        else
            [newCurrOptw,value] = runOptimiser(currK,curru,currOptw,maxGradValues);
            newCurrSetValue = -value;
        end
        currOptw = newCurrOptw;
        currSetValue = newCurrSetValue;
        setValues(sizeS-k+1:sizeS) = currSetValue;
        timeTaken(iterNum) = toc(iterationTime);
        stageWeights(1:sizeS,sizeS-k+1:sizeS) = repmat(currOptw(:),1,k);
        if(mod(sizeS,50)==0)
            fprintf('Finished choosing %d elements\n',sizeS);
        end
    end
    fprintf('Number of iterations = %d\n',iterNum);
    fprintf('Time taken to choose the optima set for k = %d is = %f secs\n',k,toc(start));
end