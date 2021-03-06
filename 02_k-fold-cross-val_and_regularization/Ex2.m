%% part a
clear;
close all;
clc;

%Initialize Losses
train_loss = zeros(1,7); %train loss
%valid_loss = zeros(1,7);% valid loss

% load the date
load('dataLinReg2D.txt');

% plot it
figure(1);clf;hold on;
plot3(dataLinReg2D(:,1),dataLinReg2D(:,2),dataLinReg2D(:,3),'r.');

% decompose in input X and output Y
n = size(dataLinReg2D,1);
X = dataLinReg2D(:,1:2);
Y = dataLinReg2D(:,3);

% prepend 1s to inputs
X = [ones(n,1),X];

% compute optimal beta
%old beta = inv(X'*X)*X'*Y;
%new beta
lambda = [0.001,0.01,0.1,1,10,100,1000]';
for i = 1:length(lambda)
    beta = inv(X'*X+ lambda(i)*eye(3))*X'*Y;
    train_loss(i) = norm(Y-X*beta)^2 + lambda(i)*norm(beta(2:length(beta)))^2;    
end

display the function
[a, b] = meshgrid(-2:.1:2,-2:.1:2);
Xgrid = [ones(length(a(:)),1),a(:),b(:)];
Ygrid = Xgrid*beta;
Ygrid = reshape(Ygrid,size(a));
%h = surface(a,b,Ygrid);
view(3);
grid on;

figure
%set(gca, 'YScale', 'log')
semilogx(lambda, train_loss)

%% part b
clear;
close all;
clc;

%Initialize Losses
train_loss = zeros(1,7); %train loss
valid_loss = zeros(1,7);% valid loss

% load the date
load('dataQuadReg2D.txt');

% plot it
figure(1);clf;hold on;
plot3(dataQuadReg2D(:,1),dataQuadReg2D(:,2),dataQuadReg2D(:,3),'r.');

% decompose in input X and output Y
n = size(dataQuadReg2D,1);
X = dataQuadReg2D(:,1:2);
Y = dataQuadReg2D(:,3);

% prepend 1s to inputs
X = [ones(n,1),X];

%for quadratic features:
%X = quadFeature(X);

% compute optimal beta
%beta = inv(X'*X)*X'*Y;
%new beta
lambda = [0.001,0.01,0.1,1,10,20,100,1000]';
lambda = [24]
for i = 1:length(lambda)
    beta = inv(X'*X+ lambda(i)*eye(3))*X'*Y;
    train_loss(i) = norm(Y-X*beta)^2 + lambda(i)*norm(beta(2:length(beta)))^2;    
end

display the function
[a,b] = meshgrid(-2:.1:2,-2:.1:2);
Xgrid = [ones(length(a(:)),1),a(:),b(:)];
Ygrid = Xgrid*beta;
Ygrid = reshape(Ygrid,size(a));
h = surface(a,b,Ygrid);
view(3);
grid on;

figure
%set(gca, 'YScale', 'log')
semilogx(lambda, train_loss)



%% part c
clear;
close all;
clc;

k = 10; %number of partions
lambda_vec = [0.001,0.01,0.1,1,5,10,15,20,22,23,24,25,26,27,30,100,1000]'; 

%Initialize Losses
L_train = zeros(1,length(lambda_vec)); %train loss
L_val = L_train;% valid loss
L_is = L_train;

% load the date
load('dataQuadReg2D_noisy.txt');

% plot it
%figure(1);clf;hold on;
%plot3(dataQuadReg2D_noisy(:,1),dataQuadReg2D_noisy(:,2),dataQuadReg2D_noisy(:,3),'r.');

% decompose in input X and output Y
n = size(dataQuadReg2D_noisy,1);
X = dataQuadReg2D_noisy(:,1:2);
Y = dataQuadReg2D_noisy(:,3);

% prepend 1s to inputs
X = [ones(n,1),X];

%for quadratic features:
X = quadFeature(X);

%create partitions
partions = floor(linspace(0,n,k))+1;
partions(1) = 0;

for i = 1:length(lambda_vec)

    for j = 2:length(partions)
    X_train = X([1:partions(j-1), (partions(j)):n],:);
    X_val = X((partions(j-1)+1):partions(j)-1, :);
    Y_train = Y([1:partions(j-1), (partions(j)):n],:);
    Y_val = Y((partions(j-1)+1):partions(j)-1, :);
    
    % compute optimal beta
    % beta = inv(X'*X)*X'*Y; %old beta
    % regularization
    reg = lambda_vec(i) * eye(size(X_train,2));
    %reg(1,1) = 0;
    beta = inv(X_train'*X_train+reg)*X_train'*Y_train;
    
    % squared error
    L_is(j-1) = norm(Y_val-X_val*beta)^2/size(X_val,1);
    end
    
    
L_val(i) = sum(L_is)/length(L_is);

% use all data for training
reg = lambda_vec(i)*eye(size(X,2));
%reg(1,1) = 0;
beta = inv(X'*X+reg)*X'*Y;

% squared error
L_train(i) = norm(Y-X*beta)^2/size(X,1);

end

[min_val, min_pos] = min(L_val);

plot(lambda_vec, L_train);
hold on
plot(lambda_vec, L_val);
plot(lambda_vec(min_pos), min_val, 'ok');
legend({'train','val', 'minimum'})
xlabel('lambda')
ylabel('Loss')


function X = quadFeature(X)
X = [X, X(:,2).^2, X(:,2).*X(:,3), X(:,3).^2];
end


