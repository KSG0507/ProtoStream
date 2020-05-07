function meanInnerProductX = computeMeanInnerProductX(X,Y,kernelType,varargin)
    if(strcmpi(kernelType,'Gaussian'))
        % Std. Dev. of the kernel if it is Gaussian
        if(~isempty(varargin))
            sigma = varargin{1};
        else
            sigma = 1;
        end
    end
    numY = size(Y,2);
    numX = size(X,2);
    %% Store the mean inner products with X
    methodType = 'slower';
    if(nargin > 4)
        methodType = varargin{2};
    end
    switch methodType
        case 'slower'
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
        case 'faster'
            switch kernelType
                case 'Gaussian'
                    distX = pdist2(Y',X');
                    KX = exp(-distX.^2/(2*sigma^2));
                otherwise
                    KX = Y'*X;
            end
            meanInnerProductX = sum(KX,2)/numX;     
    end
end