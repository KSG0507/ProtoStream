function classifiedLabels = NNC(X,Y,labels_Y,varargin)
    startSparsity = 5;
    if(nargin > 3)
        startSparsity = varargin{1};
    end
    fprintf('Inside NNC...Start Sparsity Level = %d\n',startSparsity);
    numY = size(Y,2);
    numX = size(X,2);
    startSparsity = min(startSparsity,numY);
    distY = pdist2(X',Y');
    [nearestDist,nearestIndex] = min(distY(:,1:startSparsity),[],2);
    assignedLabels = labels_Y(nearestIndex);
    %% Labels for each level of sparisty
    classifiedLabels = zeros(numX,numY-startSparsity+1);
    count = 1;
    classifiedLabels(:,count) = assignedLabels(:);
    for m = startSparsity+1:numY
        for x = 1:numX
            if(distY(x,m) < nearestDist(x))
                nearestDist(x) = distY(x,m);
                nearestIndex(x) = m;
            end
        end
        count = count + 1;
        assignedLabels = labels_Y(nearestIndex);
        classifiedLabels(:,count) = assignedLabels(:);
    end
end