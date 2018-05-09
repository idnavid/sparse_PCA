function eta = orthogonality(U)
% Calculates the level of orthogonality of principle components 
% in a projection matrix U. 
% Inputs: 
%        U: projection bases (principal components) (p x r)
%           where   p: data dimension
%                   r: reduced dimension
%
% Outputs: 
%       eta: level of non-orthogonality ||U^TU - I_r||*sqrt(1/r). 
%            the coefficient is multiplied by sqrt(1/r) to:
%              - remove the effect of r by dividing
%
% Navid Shokouhi,
% 2018

U = U/norm(U); 
[p,r] = size(U);
if p < r
    U = U';
    [~,r] = size(U);
end

eta = sqrt(1/r)*norm(U'*U - eye(r),'fro');
