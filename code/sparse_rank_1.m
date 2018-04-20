function [u,s,v] = sparse_rank_1(X, u1, v1, s1, lambda, iter, tol)
% Sparse rank-one matrix approximation
% Inputs:
%       X:          matrix of size n x p
%       u1,v1,s1:   initial rank-1 approximation (typically svd)
%       lambda:     paramter controlling the sparsity
%       iter:       maximum number of iterations for sparse rank-1 approximation. 
%       tol:        minimum required tollerance to break iterations for sparse
%                   rank-1 approximation. ||X -u*s*v'||_2 < tol
%
% Outputs:
%       u,v,s:      sparse rank-1 approximation of X (s is scalar)

u = u1;
v_old = v1*s1; 
v = zeros(size(v1));

lambda_scaled = zeros(size(v1));
p = length(v1);

for i = 1:p
    lambda_scaled(i) = lambda/abs(v_old(i));
end

j = 0;
while j < iter
    j = j+1;
    y = X'*u;
    for n = 1:p
        v(n) = sign(y(n))*(abs(y(n))>=lambda_scaled(n))*(abs(y(n))-...
                                                         lambda_scaled(n));
    end
    u = X*v/(norm(X*v,2)+1e-7);
    delta = norm(v-v_old,2);
    if (delta<tol),break; end
    v_old = v;
end

v = v/norm(v,2);
s = u'*X*v;
    
end

