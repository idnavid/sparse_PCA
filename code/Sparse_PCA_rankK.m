function [Wx,Wy,Xp,Yp] = Sparse_CCA_rank1(X,Y,options)
% Sparse rank-1 approximation of XY'.
% Inputs:
%       X:          NxM matrix, N: dimension of data, M: number of samples
%       Y:          NxM matrix
%       options:    structure of variables.
%                   iter:       number of iterations between wx and wy.
%                   dim:        number of canonical vectors.
%                   reg_wx/wy:  regularization terms over U and V (i.e., X
%                                                                    and Y)

[Nx,Mx] = size(X);
[Ny,My] = size(Y);
if Mx~=My
    error('number of samples must be the same!');
end
iter = options.iter;
n_ccvecs = options.dim;
reg_wx = options.regwx;
reg_wy = options.regwy;

X = X - repmat(mean(X,2),1,Mx);
Y = Y - repmat(mean(Y,2),1,My);

Cxx = X*X' + 10^(-6)*eye(Nx);
Cyy = Y*Y' + 10^(-6)*eye(Ny);
iCxx = mldivide(Cxx,eye(Nx));
iCyy = mldivide(Cyy,eye(Ny));
Px = X'*iCxx*X;
Py = Y'*iCyy*Y;
Kxy = Px*Py;

Wx = zeros(Nx,n_ccvecs);
Wy = zeros(Ny,n_ccvecs);

if (reg_wx==0 && reg_wy==0)
    % standard SVD (i.e., no penalty) 
    [U,D,V] = svd(Kxy);
    Wx = iCxx*X*U(:,1:n_ccvecs)*D(1:n_ccvecs,1:n_ccvecs).^(1/2);
    Wy = iCyy*Y*V(:,1:n_ccvecs)*D(1:n_ccvecs,1:n_ccvecs).^(1/2);
else
    % enforce sparsity constraints
    for jj=1:n_ccvecs 
        [U,D,V] = svd(Kxy);
        u_tilde = U(:,1);
        v_tilde = V(:,1);
        xWx = u_tilde;
        yWy = v_tilde;
        for ii=1:iter
            Wxt = OMP_1D(Kxy*yWy,X',reg_wx);
            Wyt = OMP_1D(Kxy'*xWx,Y',reg_wy);
            xWx = X'*Wxt;
            yWy = Y'*Wyt;        
        end
        
        if norm(X'*Wxt,2)==0
            ui = X'*Wxt;
        else
            ui = X'*Wxt/norm(X'*Wxt,2);
        end
        
        if norm(Y'*Wyt,2)==0
            vi = Y'*Wyt;
        else
            vi = Y'*Wyt/norm(Y'*Wyt,2);
        end
        Kxy = Kxy - ui'*Kxy*vi*ui*vi';
        Wx(:,jj) = Wxt;
        Wy(:,jj) = Wyt;
    end
end
Xp = Wx'*X;
Yp = Wy'*Y;

end

function [c] = OMP_1D(x,D,L)
%=============================================
% Sparse coding of a single signal based on a given
% dictionary and specified number of atoms to use.
% input arguments:
%       D - the dictionary (its columns MUST be normalized).
%       x - the signal to represent
%       L - the max. number of coefficients for each signal.
% output arguments:
%       c - sparse coefficient vector.
%=============================================
n = length(x);
x = x(:);
[n,K] = size(D);
indx = zeros(L,1);
residual = x;

for j = 1:L,
    proj = D'*residual;
    [maxVal,pos] = max(abs(proj));
    indx(j) = pos;
    s = pinv(D(:,indx(1:j)))*x;
    residual = x - D(:,indx(1:j))*s;
    if sum(residual.^2) < 1e-6
        break;
    end,
end;
temp = zeros(K,1);
temp(indx(1:j)) = s;
c = sparse(temp);
return;


end

