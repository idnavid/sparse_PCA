function percent_var = explained_variance(U,X)
U = U/norm(U); 
percent_var = 100*adjusted_variance(U,X)/trace(X*X');

end

function v = adjusted_variance(U,X)
% This function calculates the adjusted variance of sparse PCA as described
% in "sparse principal component analysis," Zou, Hastie, Tibshirani, 2006. 
% The problem with calculating the explained variance is that since sparse PCA
% compromizes the uncorrelatedness of PCs, it always provides an
% overestimate of the explained variance. Therefore, each PC should be
% uncorrelated from the previous ones. This is accomplished by QR
% decomposition. 
pcs = U'*X;
[Q,R] = qr(pcs); 
R2 = R*R';
v = trace(R2);
end