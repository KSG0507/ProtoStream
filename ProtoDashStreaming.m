function [currOptw,S,setValue,stageGradients] = ProtoDashStreaming(X,Y,m,kernelType,varargin) 
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
    %% Intialization
    stageGradients = zeros(numY,1);
    numRemaining = length(remainingElements);
    H = MinHeap(m);
    tic;
    for count = 1:numRemaining
        i = remainingElements(count);
        u = meanInnerProductX(i);
        element.key = i;
        element.value = u;
        stageGradients(i) = u;
        if(H.Count() < m)
            H.InsertKey(element);
        else
            if(H.ReturnMin() < element.value)
                H.ReplaceMin(element);
            end
        end
    end
    sortedArray = H.Sort();
    S = zeros(1,m);
    uVec = zeros(1,m);
    currPos = 1;
    for j = numel(sortedArray):-1:1
        S(currPos) = sortedArray{j}.key;
        uVec(currPos) = sortedArray{j}.value;
        currPos = currPos + 1;
    end
    addedZ = Y(:,S);
    switch kernelType
        case 'Gaussian'
            selfDist = pdist2(addedZ',addedZ');
            selfInnerProduct = exp(-selfDist.^2/(2*sigma^2));
        otherwise
            selfInnerProduct =  addedZ'*addedZ;
    end
    K = selfInnerProduct;
    if(max(uVec)<=0)
        currOptw = zeros(m,1);
        setValue = 0;
    else
        [currOptw,value] = runOptimiser(K,uVec,[],uVec);
        setValue = -value;
    end
    timeTaken = toc;
    fprintf('Time take by the vanilla streaming algorithm is %f secs \n',timeTaken);
end