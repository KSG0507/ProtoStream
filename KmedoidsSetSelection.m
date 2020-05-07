function [currOptw_U,setValues_U,currOptw_E,setValues_E,S] = KmedoidsSetSelection(X,Y,m,kernelType,varargin) 
    if(strcmpi(kernelType,'Gaussian'))
        % Std. Dev. of the kernel if it is Gaussian
        if(~isempty(varargin))
            sigma = varargin{1};
        else
            sigma = 1;
        end
    end
    %%
    numX = size(X,2);
    setValues_U = zeros(1,m);
    setValues_E = zeros(1,m);
    %% Run kmedoids algorithm for k = 1 through to m
    startSparsity = 1;
    if(nargin > 5)
        startSparsity = varargin{2};
    end
    incrementLevel = 1;
     if(nargin > 6)
        incrementLevel = varargin{3};
    end
    numLevelsSparsity = length(startSparsity:incrementLevel:m);
    S = zeros(m,numLevelsSparsity);
    spLevel = 0;
    for sizeS = startSparsity:incrementLevel:m
        [~,protoTypes,~,~,medoidsPos] = kmedoids(Y',sizeS);
        spLevel = spLevel+1;
        S(1:sizeS,spLevel) = medoidsPos(:);
        switch kernelType
            case 'Gaussian'
                distProto = pdist2(protoTypes,protoTypes);
                K = exp(-distProto.^2/(2*sigma^2));
            otherwise
                K = protoTypes*protoTypes';
        end
        %% Store the mean inner products with X
        methodType = 'faster';
        if(nargin > 7)
            methodType = varargin{4};
        end
        switch methodType
            case 'slower'
                u = zeros(sizeS,1);
                for i = 1:sizeS
                    switch kernelType
                        case 'Gaussian'
                            distX = pdist2(X',protoTypes(i,:));
                            u(i) = sum(exp(-distX.^2/(2*sigma^2)))/numX;
                        otherwise
                            u(i) = sum(protoTypes(i,:)*X)/numX;
                    end
                end
            case 'faster'
                switch kernelType
                    case 'Gaussian'
                        distX = pdist2(protoTypes,X');
                        KX = exp(-distX.^2/(2*sigma^2));
                    otherwise
                        KX = protoTypes*X;
                end
                u = sum(KX,2)/numX;   
        end        
        %% Unequal weights
        if(sizeS==1)
            newCurrOptw = max(u/K,0);
            currSetValue = -0.5*K*(newCurrOptw^2) + u*newCurrOptw;
        else
            [newCurrOptw,value] = runOptimiser(K,u,rand(sizeS-1,1),0);
            currSetValue = -value;
        end
        currOptw_U = newCurrOptw;
        setValues_U(sizeS) = currSetValue;
        %% Equal weights
        currOptw_E = ones(sizeS,1)/(sizeS);
        setValues_E(sizeS) = currOptw_E'*u(:)-0.5*currOptw_E'*K*currOptw_E;
        fprintf('Finished choosing %d elements\n',sizeS);
    end
end