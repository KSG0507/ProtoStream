function [currOptw,S,setValues,stageWeights] = GreedySetSelection(X,Y,m,kernelType,varargin) 
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
    S = zeros(1,m);
    timeTaken = zeros(1,m);
    ratioNotRun = zeros(1,m);
    setValues = zeros(1,m);
    sizeS = 0;
    currSetValue = 0;
    currOptw = [];
    currK = [];
    curru = [];
    runningInnerProduct = zeros(m,numY);
    stageWeights = zeros(m);
    %%
    while (sizeS < m)
        startNextSelection = tic;
        remainingElements = setdiff(allY,S(1:sizeS));
        newCurrSetValue = currSetValue; 
        newQuickSelectVal=0;
        optimiserNotRun = 0;
        for count = 1:length(remainingElements)
            i = remainingElements(count);
            %% Testing purpose
            if(count==1)
                quickChoice = i;
            end
            %%
            newZ = Y(:,i);
            if (sizeS==0)
                switch kernelType
                    case 'Gaussian'
                        K = 1;
                    otherwise
                        K = newZ'*newZ;
                end
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
                K = vertcat(K1,[innerProduct',selfNorm]);
            end
            u = [curru;meanInnerProductX(i)];
            %%
            if(sizeS==0)
                w = max(u/K,0);
                value = 0.5*K*(w^2) - u*w;
                optimiserNotRun = optimiserNotRun + 1;
            else
                gradientVal = meanInnerProductX(i)-currOptw'*innerProduct;
                if(gradientVal<=0)
                    w = [currOptw(:);0];
                    value = -currSetValue;
                    optimiserNotRun = optimiserNotRun + 1;
                else
                    [w,value] = runOptimiser(K,u,currOptw,gradientVal);
                    if(gradientVal > newQuickSelectVal)
                        newQuickSelectVal = gradientVal;
                        quickChoice = i;
                    end
                end
            end
            incrementSetValue = -value;
            if((incrementSetValue > newCurrSetValue)|| count==1)
                %% Bookeeping
                desiredElement = i;
                newCurrSetValue = incrementSetValue;
                newCurrOptw = w;
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
        currOptw = newCurrOptw;
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
        timeTaken(sizeS) = toc(startNextSelection);
        stageWeights(1:sizeS,sizeS) = currOptw(:);
        if(mod(sizeS,50)==0)
            fprintf('Finished choosing %d elements\n',sizeS);
            disp(timeTaken(1:sizeS));
        end
        %fprintf('Number of times the optimiser is not run: %d out of %d\n',optimiserNotRun,count);
        ratioNotRun(sizeS) = (optimiserNotRun/count)*100;
        %if(sizeS > 1)
            %fprintf('Comparing quick choice:%d with right choice:%d\n',quickChoice,desiredElement);
        %end
    end
%     figure(1);
%     plot(timeTaken,'r-','Linewidth',2);
%     xlabel('Sparsity level','fontsize',20,'fontweight','bold');
%     ylabel('Time taken (in secs)','fontsize',20,'fontweight','bold');
%     set(gca,'fontsize',20,'fontweight','bold');
%     
%     figure(2);
%     plot(ratioNotRun,'b-','Linewidth',2);
%     xlabel('Sparsity level','fontsize',20,'fontweight','bold');
%     ylabel('Optimiser not run (in %)','fontsize',20,'fontweight','bold');
%     set(gca,'fontsize',20,'fontweight','bold');
end