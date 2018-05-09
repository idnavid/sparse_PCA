function [u,v,d] = pmd_rank1(X,c)
% Implementation of rank-1 PMD for dual lasso penalties as proposed in
%  Witten, D.M., Tibshirani, R. and Hastie, T., 2009. 
%  "A penalized matrix decomposition, with applications 
%   to sparse PCA and CCA," Biostatistics, 10(3), pp.515-534.
%
%  This is the PCA implementation. The CCA (more general) implementation of this 
%  algorithm is available in https://github.com/idnavid/sparse_CCA
%
% Inputs:
%       X: data matrix
%       c: sparcity coefficient (c closer to zero is stricter)
% 
% Outputs:
%       u, v: eigenvectors
%       d: eigenvalue
%
%
% Navid Shokouhi, 
% 2018

[n1,n2] = size(X);
Delta_steps = 0.001; % increments used to threshold u(or v) so L1 const. is satisfied
epsilon = 1e-6; % stopping criterion
max_iter = 15; % emergency stopping criterion

c1 = c*sqrt(n1);
c2 = c*sqrt(n2);

v = randn(n1,1);
v = v/norm(v,2);
iter = 1;
while true
    u = onesided_maximization(X,v,c1,Delta_steps);
    v_old = v;
    v = onesided_maximization(X',u,c2,Delta_steps);
    if (norm(v_old - v,2)<epsilon) || iter > max_iter
        break;
    end
    iter = iter+1;
end
d = u'*X*v;
end


function u = onesided_maximization(X,v,c,increment)
    % maximizize either u or v such that L1 penalty is satisfied. 
    % If u is satisfied, it means v is kept fix. 
    Delta= 0;
    s = soft_threshold(X*v,Delta);
    u = s/norm(s);
    while norm(u,1)>c
        Delta= Delta+increment;
        s = soft_threshold(X*v,Delta);
        if norm(s,2)~=0
            u = s/norm(s,2);
        else
            return;
        end
        if norm(u,1)<c
            increment = increment/2;
            Delta = Delta - increment;
            s = soft_threshold(X*v,Delta);
            u = s/norm(s,2);
        end
    end    
end

function s = soft_threshold(x,y)
temp = (abs(x) - y); 
temp(temp<=0) = 0;
s = sign(x).*temp;
end

