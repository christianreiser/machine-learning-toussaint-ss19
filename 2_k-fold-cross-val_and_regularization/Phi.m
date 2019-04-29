function [Phi_X] = Phi(X)
[n,d] = size(X);

Phi_X = ones(m,1+d+d*(d+1)/2);
Phi_X(m,1:d) = X;

Phi_X(m, d+1:d+3) = X(:,2).*X(



end

