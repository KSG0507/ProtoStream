function [w,value] = runLearnToCritize(K,u)
    d = length(u);
    w = ones(d,1)/d;
    value = 0.5*(w'*K*w)-w'*u(:);
end