% Simulation based on synthesis experiment presented in 
% "Globally Sparse Probabilistic PCA" by Mattei, Bouveyron, Latouche
clear
addpath(genpath('../code/'));
n = 100; % observations
p = 10; % data dimension
d = 8; % latent space dimension (i.e., true number of PCs)
d_est = 8; % number of PCs to estimate
q = 4; % sparsity
rng('default');
rng(1);
W = randn(p,d); 
V = diag([ones(1,q),zeros(1,p - q)]);
y = get_orthonormal(n,d)';
X = V*W*y + .1*randn(p,n);
threshold = 1e-6;
%% PCA
[U_pca,~,~] = svds(X*X',d_est);
subplot(311)
plot(diag(V),'r','linewidth',2),hold on
plot((((abs(U_pca)))>threshold),'x'),hold off
ylim([0,1.5])
legend('ground-truth','PCA')
title(explained_variance(U_pca,X))

%% sparse PCA - Witten etal. 2009 
[U_pmd,~] = pmd_rankK(X,X,d_est,.45);
subplot(312)
plot(diag(V),'r','linewidth',2), hold on 
plot(abs(U_pmd)>threshold,'x'),hold off
ylim([0,1.5])
legend('ground-truth','PMD (Witten etal. 09)')
title(explained_variance(U_pmd,X))

%% Proposed
opt.lambda =1;
opt.K = d_est;
opt.iter1 = 20;
opt.iter2 = 20;
opt.tol1 = 1e-2;
opt.tol2 = 1e-2;
[~,~,~,U_spca_prop] = Sparse_PCA(X', opt.lambda, opt.K, opt.iter1, ...
                               opt.iter2, opt.tol1, opt.tol2);

subplot(313)
plot(diag(V),'r','linewidth',2), hold on
plot(abs(U_spca_prop)>threshold,'x'),hold off
ylim([0,1.5])
legend('ground-truth','sparse PCA (proposed)')
title(explained_variance(U_spca_prop,X))
