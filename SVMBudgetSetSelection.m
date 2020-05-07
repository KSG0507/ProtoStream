function [currOptw,S,setValues,w,numNonZero] = SVMBudgetSetSelection(X,Y,m,kernelType,individualMaxVal,varargin) 
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
    if(nargin > 6)
        %fprintf('Reusing the vector meanInnerProductX...\n');
        u = varargin{2};
    else
        fprintf('Computing the vector meanInnerProductX...\n');
        u = zeros(numY,1);
        for i = 1:numY
            switch kernelType
                case 'Gaussian'
                    distX = pdist2(X',Y(:,i)');
                    u(i) = sum(exp(-distX.^2/(2*sigma^2)))/numX;
                otherwise
                    u(i) = sum(Y(:,i)'*X)/numX;
            end
        end
    end
    switch kernelType
        case 'Gaussian'
            distY = pdist2(Y',Y');
            K = exp(-distY.^2/(2*sigma^2));
        otherwise
            K = Y'*Y;
    end
    fprintf('Condition number of K = %f\n',cond(K));
    %fprintf('Running the Optimization...\n');
    algorithmName = 'interior-point-convex';
    options = optimoptions(@quadprog,'Display','off','MaxIter',500,'TolFun',1e-8,...
                'TolX',1e-8,'Algorithm',algorithmName);
    l1bound = m*individualMaxVal;
    lb = zeros(numY,1);
    ub = individualMaxVal*ones(numY,1);
    A = ones(1,numY);
    x0 = (l1bound/numY)*ones(numY,1);
    [w,value] = quadprog(K,-u,A,l1bound,[],[],lb,[],x0,options);
    w = w(:);
    %% Statistics about the computed weights
    nonZero = w >= 1e-04;
    numNonZero = sum(nonZero);
    fprintf('Bound on L1 = %f\n',l1bound);
    fprintf('Sum of weights = %f\n',sum(w));
    fprintf('Max value of weights = %f\n',max(w));
    fprintf('Min value of weights = %f\n',min(w));
    fprintf('Number of non-zero elements = %d\n',numNonZero);
    if(numNonZero==0)
        currOptw = [];
        S = [];
        setValues = -value;
        return
    end
    %% Select only those components correponding to non zero weights
    Shat = allY(nonZero);
    Khat = K(nonZero,nonZero);
    uhat = u(nonZero);
    what = w(nonZero);
    %% Determine the order of points 
    pointsOrder = 'Sorting';
    if(nargin > 7)
        pointsOrder = varargin{3};
    end
    numElements = min(numNonZero,m);
    %numElements = numNonZero;
    if(nargin > 8)
        switch varargin{4}
            case 'chooseall'
                numElements = numNonZero;
            case 'choosem'
                numElements = min(numNonZero,m);
        end
    end
    setValues = zeros(numElements,1);
    switch pointsOrder
        case 'Incremental'
            fprintf('Determining the order based on incremental values...\n');
            allIndices = 1:numNonZero;
            IX = zeros(1,numElements);
            currOptw = zeros(numElements,1);
            S = zeros(numElements,1);
            sizeS = 0;
            currSetValue = 0;
            while (sizeS < numElements)
                remainingElements = setdiff(allIndices,IX(1:sizeS));
                currMaxValue = 0;
                for count = 1:length(remainingElements)
                    i = remainingElements(count);
                    incrementSetValue = what(i)*uhat(i)-0.5*Khat(i,i)*(what(i)^2);
                    if(sizeS > 0)
                        innerProduct = Khat(IX(1:sizeS),i);
                        incrementSetValue = incrementSetValue-what(i)*(currOptw(1:sizeS)'*innerProduct);
                    end
                    if((incrementSetValue > currMaxValue) || count==1)
                        currMaxValue = incrementSetValue;
                        desiredElement = i;
                        newCurrSetValue = currSetValue + incrementSetValue;
                    end
                end
                %% Set up values for next outer iteration over m
                sizeS = sizeS+1;
                S(sizeS) = Shat(desiredElement);
                IX(sizeS) = desiredElement;
                currSetValue = newCurrSetValue;
                setValues(sizeS) = currSetValue;
                currOptw(sizeS) = what(desiredElement);
            end
        case 'Sorting'
            %fprintf('Determining the order by sorting...\n');
            [sortedW,IXTemp] = sort(what,'descend');
            currOptw = sortedW(1:numElements);
            IX = IXTemp(1:numElements);
            S = Shat(IX);
            currSetValue = 0;
            for i = 1:numElements
                selfNorm = Khat(IX(i),IX(i));
                incrementSetValue = currOptw(i)*uhat(IX(i))-0.5*selfNorm*(currOptw(i)^2);
                if(i > 1)
                    innerProduct = Khat(IX(1:i-1),IX(i));
                    incrementSetValue = incrementSetValue - currOptw(i)*(currOptw(1:i-1)'*innerProduct);
                end
                currSetValue = currSetValue+incrementSetValue;
                setValues(i) = currSetValue;
            end
    end
end