function [currOptw,S,setValues,stageWeights,stageGradients] = HeuristicSetSelection(X,Y,m,kernelType,varargin) 
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
    %% Intialization
    S = zeros(1,m);
    timeTaken = zeros(1,m);
    setValues = zeros(1,m);
    sizeS = 0;
    currSetValue = 0;
    currOptw = [];
    currK = [];
    curru = [];
    runningInnerProduct = zeros(m,numY);
    stageWeights = zeros(m);
    stageGradients = zeros(numY,m);
    %%
    while (sizeS < m)
        tic;
        remainingElements = setdiff(allY,S(1:sizeS));
        newCurrSetValue = currSetValue; 
        maxGradient=0;
        for count = 1:length(remainingElements)
            %%
            i = remainingElements(count);
            newZ = Y(:,i);
            if (sizeS==0)
                switch kernelType
                    case 'Gaussian'
                        K = 1;
                    otherwise
                        K = newZ'*newZ;
                end
                u = meanInnerProductX(i);
                stageGradients(i,sizeS+1) = u;
                gradientVal = u;
                if((gradientVal > maxGradient)|| count==1)
                    %% Bookeeping
                    maxGradient = gradientVal;
                    desiredElement = i;
                    w = max(u/K,0);
                    newCurrSetValue = -0.5*K*(w^2) + u*w;
                    newCurrOptw = w;
                    currK = K;
                end
            else
                recentlyAdded = Y(:,S(sizeS));
                switch kernelType
                    case 'Gaussian'
                        distnewZ = norm(recentlyAdded-newZ);
                        runningInnerProduct(sizeS,i) = exp(-distnewZ.^2/(2*sigma^2));
                    otherwise
                        runningInnerProduct(sizeS,i) = recentlyAdded'*newZ;
                end
                innerProduct = runningInnerProduct(1:sizeS,i);
                gradientVal = meanInnerProductX(i)-currOptw'*innerProduct;
                stageGradients(i,sizeS+1) = gradientVal;
                if((gradientVal > maxGradient)|| count==1)
                    maxGradient = gradientVal;
                    desiredElement = i;
                    newinnerProduct = innerProduct(:);
                end
            end
        end
        %% Set up values for next outer iteration over m
        sizeS = sizeS+1;
        S(sizeS) = desiredElement;
        curru = [curru;meanInnerProductX(desiredElement)];
        if(sizeS > 1)
            switch kernelType
                case 'Gaussian'
                    selfNorm = 1;
                otherwise
                    addedZ = Y(:,desiredElement);
                    selfNorm =  addedZ'*addedZ;
            end
            K1 = horzcat(currK,newinnerProduct(:));
            K2 = vertcat(K1,[newinnerProduct',selfNorm]);
            currK = K2;
            if(maxGradient<=0)
                newCurrOptw = [currOptw(:);0];
                newCurrSetValue = currSetValue;
            else
                [newCurrOptw,value] = runOptimiser(currK,curru,currOptw,maxGradient);
                newCurrSetValue = -value;
            end
        end
        currOptw = newCurrOptw;
        currSetValue = newCurrSetValue;
        setValues(sizeS) = currSetValue;
        timeTaken(sizeS) = toc;
        stageWeights(1:sizeS,sizeS) = currOptw(:);
        if(mod(sizeS,50)==0)
            fprintf('Finished choosing %d elements\n',sizeS);
        end
    end
end