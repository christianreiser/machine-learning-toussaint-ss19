function [Phi_X] = Phi(X)
[m,d] = size(X);
Phi_X = ones(m,6);
Phi_X(:,1:3) = X;

Phi_X(:, 4) = X(:,2).*X(:,2);
Phi_X(:,5) = X(:,2).*X(:,3);
Phi_X(:,6) = X(:,3).*X(:,3);

end

