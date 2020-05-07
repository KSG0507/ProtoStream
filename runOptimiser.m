function [w,value] = runOptimiser(K,u,preOptw,initialValue,varargin)
    algorithmName = 'interior-point-convex';
    if(~isempty(varargin))
        maxWeight = varargin{1};
    else
        maxWeight = 100000;
    end
    d = length(u);
    options = optimoptions(@quadprog,'Display','off','MaxIter',500,'TolFun',1e-8,...
            'TolX',1e-8,'Algorithm',algorithmName);
    %options = optimoptions(@quadprog,'Display','off','MaxIter',500);
    %Bound constraints
    lb = zeros(d,1);
    ub = maxWeight*ones(d,1);
    %Initial condition
    k = d-length(preOptw);
    diagK = diag(K);
    x0 = [preOptw(:);initialValue(:)./diagK(d-k+1:d)];
    %x0 = [preOptw(:);initialValue/K(d,d)];
    [w,value] = quadprog(K,-u,[],[],[],[],lb,[],x0,options);
end
