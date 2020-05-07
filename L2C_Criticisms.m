function [S,setValues] = L2C_Criticisms(X,Y,pLocs,weights, m,kernelType,varargin) 
    w_N = weights(:)/sum(weights); %Make the weights to sum to one
    if(strcmpi(kernelType,'Gaussian'))
        % Std. Dev. of the kernel if it is Gaussian
        if(~isempty(varargin))
            sigma = varargin{1};
        else
            sigma = 1;
        end
    end
    %%
    numY = size(Y,2);
    numX = size(X,2);
    allY = 1:numY;
    %% Store the mean inner products with X
    if(nargin > 7)
        meanInnerProductX = varargin{2};
    else
        fprintf('Comnputing meanInnerproductX inside L2C criticisms..\n');
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
     %% Store the mean inner products with prototypes
    if(nargin > 8)
        meanInnerProductP = varargin{3};
    else
        fprintf('Comnputing meanInnerproduct with prototypes inside L2C criticisms..\n');
        meanInnerProductP = zeros(numY,1);
        prototoTypes = Y(:,pLocs);
        numP = length(pLocs);
        for i = 1:numY
            switch kernelType
                case 'Gaussian'
                    distP = pdist2(prototoTypes',Y(:,i)');
                    meanInnerProductP(i) = sum(w_N.*exp(-distP.^2/(2*sigma^2)));
                otherwise
                    meanInnerProductP(i) = sum((Y(:,i)'*prototoTypes).*w_N');
            end
        end
    end
    %% Intialization
    S = zeros(1,m);
    timeTaken = zeros(1,m);
    setValues = zeros(1,m);
    sizeS = 0;
    currSetValue = 0;
    currK = [];
    runningInnerProduct = zeros(m,numY);
    %%
    while (sizeS < m)
        tic;
        remainingElements = setdiff(allY,union(S(1:sizeS),pLocs));
        newCurrSetValue = currSetValue; 
        for count = 1:length(remainingElements)
            %%
            i = remainingElements(count);
            newZ = Y(:,i);
            if (sizeS==0)
                switch kernelType
                    case 'Gaussian'
                        selfNorm = 1;
                    otherwise
                        selfNorm = newZ'*newZ;
                end
                detValue = log(selfNorm);
            else
                recentlyAdded = Y(:,S(sizeS));
                switch kernelType
                    case 'Gaussian'
                        distnewZ = norm(recentlyAdded-newZ);
                        runningInnerProduct(sizeS,i) = exp(-distnewZ.^2/(2*sigma^2));
                        selfNorm = 1;
                    otherwise
                        runningInnerProduct(sizeS,i) = recentlyAdded'*newZ;
                        selfNorm = newZ'*newZ;
                end
                innerProduct = runningInnerProduct(1:sizeS,i);
                K1 = horzcat(currK,innerProduct(:));
                K2 = vertcat(K1,[innerProduct',selfNorm]);
                potentialK = K2;
                detValue = log(det(potentialK));
            end
            incrementSetValue = currSetValue + abs(meanInnerProductX(i) - meanInnerProductP(i)) + detValue;
            if((incrementSetValue > newCurrSetValue)|| count==1)
                %% Bookeeping
                desiredElement = i;
                newCurrSetValue = incrementSetValue;
                if(sizeS > 0)
                    newPotentialK = potentialK;
                end
            end
        end
        %% Set up values for next outer iteration over m
        sizeS = sizeS+1;
        S(sizeS) = desiredElement;
        currSetValue = newCurrSetValue;
        setValues(sizeS) = currSetValue;
        switch kernelType
            case 'Gaussian'
                selfNorm = 1;
            otherwise
                addedZ = Y(:,desiredElement);
                selfNorm =  addedZ'*addedZ;
        end
        if(sizeS == 1)
            currK = selfNorm;
        else
             currK = newPotentialK;
        end
        timeTaken(sizeS) = toc;
        if(mod(sizeS,50)==0)
            fprintf('Finished choosing %d elements\n',sizeS);
        end
    end
end