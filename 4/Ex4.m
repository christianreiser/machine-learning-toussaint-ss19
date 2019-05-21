clear;
close all;
clc;

% load the date
load('data2Class.txt');

figure(1);clf;hold on;
% decompose in input X and output Y
n = size(data2Class,1);
X = data2Class(:,1:2);
y = data2Class(:,3);

X = [ones(n,1), X];


lambda = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]';
train_loss = zeros(length(lambda),1);
beta = zeros(3,1);
%compute beta

for i = 1:length(lambda)
    error = 1;
    while(error > 1e-10)
        p = 1./(ones(n,1) + exp(-X*beta));
        W = diag(p.*(1-p));
        beta_new = beta - inv(X'*W*X + 2*lambda(i)*eye(3))*(X'*(p-y)...
            + 2*lambda(i)*eye(3)*beta); 
        error = norm(beta_new - beta);
        beta = beta_new; 
        train_loss(i) = -1/n*log(-sum(y.*log(p) ...
            + (ones(n,1)-y).*log(ones(n,1) - p)));%...
            %+ lambda(i)*norm(beta)^2);
    end 
end

plot3(data2Class(:,1), data2Class(:,2), p, 'r.');

figure(2)
semilogx(lambda, train_loss)

%%
%quadratic features

clear;
% load the date
load('data2Class.txt');

figure(1);clf;hold on;
% decompose in input X and output Y
n = size(data2Class,1);
X = data2Class(:,1:2);
y = data2Class(:,3);

X = [ones(n,1) X];
X = Phi(X);




y = data2Class(:,3);


lambda = [0, 1, 10, 100, 1000, 10000, 100000, 1000000]';
train_loss = zeros(length(lambda),1);
beta = zeros(6,1);
%compute beta

for i = 1:length(lambda)
    error = 1; 
    while(error > 1e-10)
        p = 1./(ones(n,1) + exp(-X*beta));
        W = diag(p.*(1-p));
        beta_new = beta - inv(X'*W*X + 2*lambda(i)*eye(6))*(X'*(p -  y) + 2*lambda(i)*eye(6)*beta); 
        error = norm(beta_new - beta); 
        beta = beta_new; 
        
        train_loss(i) = -1/n*log(-sum(y.*log(p) + (ones(n,1)-y).*log(ones(n,1) - p)) + lambda(i)*norm(beta)^2);
    end
    
end
plot3(data2Class(:,1), data2Class(:,2), p, 'r.');

figure(2)
semilogx(lambda, train_loss)




