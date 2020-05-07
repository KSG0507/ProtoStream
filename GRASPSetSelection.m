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
    %% Intialization
    timeTaken = zeros(1,m);
    setValues = zeros(1,m);
    currSetValue = 0;
    currOptw = [];
    runningInnerProduct = zeros(m,numY);
    stageWeights = zeros(m);
    stageGradients = zeros(numY,m);
	stoppingConditionMet = false;
	S = [];
	currIter = 0;
    %%
    while (~stoppingConditionMet)
        tic;
		%Compute the gradient at the current value of weights
		gradientVec = meanInnerProductX;
		if(currIter > 0)
			Kgrad = computeInnerProductMatrix(Y,Y(:,S),kernelType,varargin);
			gradientVec = meanInnerProductX - Kgrad*currOptw;
		end
		stageGradients(:,sizeS+1)=gradientVec(:);
		[sortedGradient, maxLocations] = sort(gradientVec,'descend');
		gradientSupport = maxLocations(1:2*m);
		allSupport = union(S,gradientSupport);
		K = computeInnerProductMatrix(Y(:,allSupport),Y(:,allSupport),kernelType,varargin);
		u = meanInnerProductX(allSupport);
		u = u(:);
        [b,value] = runOptimiser(K,u,currOptw,maxGradient);
		[~,maxLocations] = sort(b,'descend');
		topLoc = maxLocations(1:m);
		currOptw = b(topLoc);
		S = allSupport(topLoc);
		curru = u(topLoc);
		currK = K(topLoc,topLoc);
        currSetValue = curru'*currOptw - 0.5*currOptw'*currK*currOptw;
        setValues(currIter+1) = currSetValue;
        timeTaken(currIter+1) = toc;
        stageWeights(1:m,currIter+1) = currOptw(:);
        currIter = currIter+1;
		if(mod(currIter,50)==0)
            fprintf('Finished choosing %d elements\n',sizeS);
        end
    end
end