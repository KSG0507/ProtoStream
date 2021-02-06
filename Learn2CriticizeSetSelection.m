function [currOptw,S,setValues] = Learn2CriticizeSetSelection(X,Y,m,kernelType,varargin) 
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
    if(nargin > 5)
        meanInnerProductX = varargin{2};
    else
        fprintf('Comnputing meanInnerproductX inside L2C..\n');
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
    S = zeros(1,m);
    timeTaken = zeros(1,m);
    setValues = zeros(1,m);
    sizeS = 0;
    currSetValue = 0;
    currOptw = [];
    currK = [];
    curru = [];
    runningInnerProduct = zeros(m,numY);
    start = tic;
    %%
    while (sizeS < m)
        tic;
        remainingElements = setdiff(allY,S(1:sizeS));
        newCurrSetValue = currSetValue; 
        w = ones(sizeS+1,1)/(sizeS+1);
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
                incrementSetValue = w(sizeS+1)*meanInnerProductX(i)-0.5*selfNorm*(w(sizeS+1)^2);
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
                wstar = w(1:sizeS);
                if(count==1)
                    baseValue = wstar'*curru(:)-0.5*wstar'*currK*wstar;
                end
                addValue = w(sizeS+1)*meanInnerProductX(i)-0.5*selfNorm*(w(sizeS+1)^2);
                addValue = addValue - w(sizeS+1)*(wstar'*innerProduct);
                incrementSetValue = baseValue+addValue;
            end
            if((incrementSetValue > newCurrSetValue)|| count==1)
                %% Bookeeping
                desiredElement = i;
                newCurrSetValue = incrementSetValue;
                if(sizeS > 0)
                    newinnerProduct = innerProduct(:);
                end
            end
        end
        %% Set up values for next outer iteration over m
        sizeS = sizeS+1;
        S(sizeS) = desiredElement;
        currSetValue = newCurrSetValue;
        setValues(sizeS) = currSetValue;
        currOptw = ones(sizeS,1)/(sizeS);
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
             K1 = horzcat(currK,newinnerProduct(:));
             K2 = vertcat(K1,[newinnerProduct',selfNorm]);
             currK = K2;
        end
        curru = [curru;meanInnerProductX(desiredElement)];
        timeTaken(sizeS) = toc;
        if(mod(sizeS,50)==0)
            fprintf('Finished choosing %d elements\n',sizeS);
        end
    end
    fprintf('Time taken to choose the optima set for L2C is = %f secs\n',toc(start));
end