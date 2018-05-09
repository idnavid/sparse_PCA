function [Wx,Wy] = pmd_rankK(X,Y,K,c)
% Deflation procedure for sparse rank-K approximation of XY'.
% Inputs:
%       X:          NxM matrix, N: dimension of data, M: number of samples
%       Y:          NxM matrix
%       K:          number of canonical components
%       c:          sparsity coefficient (c closer to zero is stricter)
% 
% NOTE: This is a generic implementation that also applies to CCA. In the 
%       case of PCA, X = Y. 
%
% Outputs:
%       Wx: collection of principal loading vectors
%
% The implementation of this algorithm for the generic CCA case is available
% here: https://github.com/idnavid/sparse_CCA
%
% Navid Shokouhi, 
% 2018

[Nx,Mx] = size(X);
[Ny,My] = size(Y);
if Mx~=My
    error("number of samples must be the same!");
end
n_ccvecs = K;

X = normalize(X,n_ccvecs);
Y = normalize(Y,n_ccvecs);

Wx = zeros(Nx,n_ccvecs);
Wy = zeros(Ny,n_ccvecs);
Kxy = X*Y';
% enforce sparsity constraints
for i=1:n_ccvecs
    [ui,vi,di] = pmd_rank1(Kxy,c);
    
    Kxy = Kxy - di*ui*vi';
    Wx(:,i) = ui;
    Wy(:,i) = vi;
end

end

function X = normalize(X,K)
% Each row of X is an attribute. 
% The columns of X are samples
X = X - repmat(mean(X,2),1,size(X,2));
[Ux,~,Vx] = fsvd(X,K);
X = Ux*Vx';
end
