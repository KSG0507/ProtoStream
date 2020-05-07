function [currOptw_U,setValues_U,currOptw_E,setValues_E,S] = RandomSetSelection(X,Y,m,kernelType,varargin) 
    if(strcmpi(kernelType,'Gaussian'))
        % Std. Dev. of the kernel if it is Gaussian
        if(~isempty(varargin))
            sigma = varargin{1};
        else
            sigma = 10;
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
    setValues_U = zeros(1,m);
    setValues_E = zeros(1,m);
    sizeS = 0;
    currSetValue = 0;
    currOptw_U = [];
    currOptw_E = [];
    currK = [];
    curru = [];
    %%
    while (sizeS < m)
        tic;
        remainingElements = setdiff(allY,S(1:sizeS));
        count = randi(length(remainingElements));
        i = remainingElements(count);
        newZ = Y(:,i);
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
            newinnerProduct = zeros(sizeS,1);
            switch kernelType
                case 'Gaussian'
                    selfNorm = 1;
                    for j = 1:sizeS
                        recentlyAdded = Y(:,S(j));
                        distnewZ = norm(recentlyAdded-newZ);
                        newinnerProduct(j) = exp(-distnewZ.^2/(2*sigma^2));
                    end
                otherwise
                    selfNorm =  newZ'*newZ;
                     for j = 1:sizeS
                        recentlyAdded = Y(:,S(j));
                        newinnerProduct(j) = recentlyAdded'*newZ;
                    end
            end
            K1 = horzcat(currK,newinnerProduct(:));
            currK = vertcat(K1,[newinnerProduct',selfNorm]);
            curru = [curru;meanInnerProductX(i)];
            gradientVal = meanInnerProductX(i)-currOptw_U'*newinnerProduct;
            if(gradientVal<=0)
                newCurrOptw = [currOptw_U(:);0];
                newCurrSetValue = currSetValue;
            else
                [newCurrOptw,value] = runOptimiser(currK,curru,currOptw_U,gradientVal);
                newCurrSetValue = -value;
            end
        end
        %% Set up values for next outer iteration over m
        sizeS = sizeS+1;
        S(sizeS) = i;
        currOptw_U = newCurrOptw;
        currSetValue = newCurrSetValue;
        setValues_U(sizeS) = currSetValue;
        %% Equal weights
        currOptw_E = ones(sizeS,1)/(sizeS);
        setValues_E(sizeS) = currOptw_E'*curru(:)-0.5*currOptw_E'*currK*currOptw_E;
        %fprintf('Finished choosing %d elements\n',sizeS);
    end
end