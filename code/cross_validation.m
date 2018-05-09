function lamda = cross_validation(X, n_samples, n_components, Kfold_value,func)
% This function finds the optimal sparsity parameter lambda for sparse PCA using 
% some training data X. 
% You'll have to specify the sparse PCA function.
%
% Inputs :
%       X:              data set  of size [n,p], where n: samples, p: dimension
%       n_samples:      Number of samples, where n = n_samples (This is to make
%                       sure were using the right axis.
%       n_components:   number of principal components
%       Kfold_value:    Number of folds for cross validation
%       func:           Specifies which function is being used;
%                       'proposed','spca', 'sspca', 'pmd'.
%
%  Outputs:
%     	lambda:         lambda with minimum reconstruction error.
%
% Navid Shokouhi, 
% 2018

[n,p] = size(X);
if n~=n_samples
    error('rows should correspond to number of samples. Try transposing X.');
end


% Create cross validation partition on n observations
cv_partitions = cvpartition(n,'KFold',Kfold_value); 

% Different sparse PCA algorithms use different scales/ranges of values for lambda
switch func
    case 'spca'
        lambda_k = 5:5:25;
    case 'pmd'
        lambda_k = 0.01:0.1:0.6;
    case 'sspca'
        lambda_k = 10.^(-7:0);
    case 'proposed'
        lambda_k = 10.^(-6:0);
end
n_lambda = length(lambda_k);


cv_err_mat = zeros(Kfold_value,n_lambda); 
params = set_params(n_components,p);
for k = 1:n_lambda
    for i = 1:Kfold_value
        X_test = X(cv_partitions.test(i),:); 
        X_train = X(logical(1-cv_partitions.test(i)),:); 
        params.lambda = lambda_k(k);
        cv_err_mat(i,k) = sparsepca_algorithm(X_train,X_test,params,func); 
    end
end
cv_err_mat(isnan(cv_err_mat)) = 1;
[~,min_idx] = min(mean(cv_err_mat,1));
plot(mean(cv_err_mat,1))
lamda = lambda_k(min_idx);
end


function err = sparsepca_algorithm(X,X_test,params,func)
% calculate reconstruction error for each sparse PCA algorithm 
switch func
    case 'spca'
        % For spca, lambda is proportional to the number of non-zero
        % elements: 
        [V,~] = spca(X, [],params.n_components_spca,params.ridge_coeff,...
                     params.lambda);
        err = 1 - explained_variance(V,X_test');
    case 'pmd'
        [V,~] = pmd_rankK(X',X',params.n_components_pmd,params.lambda);
        err = explained_variance(V,X_test');
    case 'sspca'
        [~,V] = sspca(X, params.spG, params);
        err = 1 - explained_variance(V,X_test');
    case 'proposed'
        [~,~,~,V] = Sparse_PCA(X, params.lambda, params.K, params.iter1,...
                                   params.iter2, params.tol1, params.tol2);
        err = 1 - explained_variance(V,X_test');        
end

end

function [params] = set_params(n_components,p)
% Sets all of the parameters (except lambda) for all of the sparse PCA
% algorithms at once.
% This is a helper function to keep the main cross-validation tidy. 
%
% Inputs:
%   n_components: number of principal components
%   p:            data dimension


% struct of SPCA parameters (Zou etal. 2006)
params.n_components_spca  = n_components;
params.ridge_coeff   = .01;

% struct of PMD parameters (Witten etal. 2009)
params.n_components_pmd  = n_components;

% struct of SSPCA parameters (Jenatton etal. 2010)
params.r              = n_components;
params.it0            = inf;% Never display iteration error (no verbose)
params.normparam      = 1;
params.spG            = sparse(eye(p));

% struct of parameters for proposed spca method
params.K = n_components;
params.iter1 = 50;
params.iter2 = 50;
params.tol1 = 1e-5;
params.tol2 = 1e-5;
end
