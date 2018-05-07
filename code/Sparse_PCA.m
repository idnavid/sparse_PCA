function [U1,V1,U2,V2] = Sparse_PCA(X, lambda, K, iter1, iter2, tol1, tol2)
% Perform PCA matrix decomposition with consistent sparsity pattern accross 
% all principle components (i.e., globally sparse PCA). 
% Inputs:
%       X:      data matrix of size n x p
%       K:      number of principle components to estimate
%       lambda: regularization paramter
%       iter1:  maximum number of iterations for standard sparse PCA
%       iter2:  maximum number of iteration for globally sparse PCA. 
%       tol1:   minimum required tollerance to break iterations for 
%               standard sparse PCA. 
%       tol2:   minimum required tollerance to break iterations for 
%               globally sparse PCA. 
%
% Outputs:
%       U1,V1:  PCA matrix decomposition using standard sparse PCA. 
%       U2,V2:  PCA matrix decomposition using sparse PCA with consistent
%               globally sparse principle components.
% NOTE: in both outputs U are the principle components and V are the loadings.  

[n,p] = size(X);
[U,S,V] = fsvd(X,K);


%% U1 and V1 (standard sparse PCA)
U1 = zeros(n,K);
V1 = zeros(p,K);
u1 = U(:,1);
v1 = V(:,1);
s1 = S(1,1);

X_residual = X;
for i = 1:K
    [u,s,v] = sparse_rank_1(X_residual, u1, v1, s1, lambda, iter1, tol1);
    X_residual = X_residual - u*s*v';
    U1(:,i) = u;
    V1(:,i) = s*v;
    u1 = U(:,i);
    v1 = V(:,i);
    s1 = S(i,i);
end

%% U2 and V2 (globally sparse PCA)
U2 = U(:,1:K);
V2 = V(:,1:K);

lambda1 = zeros(p,1);
for i = 1:p
    lambda1(i) = lambda/norm(V2(i,:));
end

U2_old = U2;
j = 0;
while j < iter2
    j = j+1;
    for i = 1:p        
        res_i = U2'*X(:,i); % projection residual
        if (lambda1(i)*sqrt(p)) < 2*norm(res_i,2)
            v = (1-((lambda1(i)*sqrt(p))/(2*norm(res_i,2))))*res_i;
        else
            v = zeros(K,1);
        end
        V2(i,:) = v;
    end
    [U_j,~,V_j] = svd(X*V2);
    U2 = U_j(:,1:K)*V_j';
    
    delta = norm(U2-U2_old,'fro');    % Norm change during successive iterations
    if (delta < tol2), break; end
    U2_old = U2;
end

end

