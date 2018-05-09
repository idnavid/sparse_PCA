function percent_var = explained_variance(U,X)
% Calculte percentage of variance explained using the principal subspace spanning
% U (i.e., UU').
% 
% Inputs:
%	U: matrix containting q principal loading vectors, size: p by q, p: is dimension
%       X: data matrix. 
%
% Outputs:
%       percent_var: percentage of variance explained in span(U). 
%
% Navid Shokouhi,
% 2018

U = U/norm(U); 

% NOTE: To calculate the variance of the subspace UU', we must use adjusted variance
%       when U is sparse (See Sparse PCA paper by Zou etal. 2006 and explanation below). 
percent_var = 100*adjusted_variance(U,X)/trace(X*X');
end


function v = adjusted_variance(U,X)
% NOTE: This function calculates the adjusted variance of sparse PCA as described
% in "sparse principal component analysis," Zou, Hastie, Tibshirani, 2006. 
% The problem with calculating the explained variance is that since sparse PCA
% compromizes the uncorrelatedness of PCs, it always provides an
% overestimate of the explained variance. Therefore, each PC should be
% uncorrelated from the previous ones. This is accomplished by QR
% decomposition. 
%
% Navid Shokouhi
% 2018
pcs = U'*X;
[Q,R] = qr(pcs); 
R2 = R*R';
v = trace(R2);
end
