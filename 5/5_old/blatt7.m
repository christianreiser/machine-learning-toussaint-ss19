close all;
clear;
clc;

task1 = true; % exercise 1

load('data2Class_adjusted.txt');
data = data2Class_adjusted;

X = data(:,1:3);
Y = data(:,4);


h1 = 100;

% initialize weights with Gaussian with sdv √(1/h_{l-1[)
w0 = normrnd(0,1/sqrt(h1-1),size(X));

% ini biases U(−1, 1) uniform distribution
b1 = rand([1,size(X(:,1))])*2 -1;


% gradient descent
alpha = 0.05;
k = 0;
kmax = 100;

if task1
    %z1 = leakyReLU(plus(w0 .* X,b1));
    w0X = w0 .* X;
    f = max(0.01*w0X,w0X); %lReLU
    l = f-Y; %loss
end


% while (true)
%     k = k+1;
%     [grad0, grad1] = grad(X, Y, w0, w1);
%     
%     w0 = w0 -alpha.*grad0;
%     w1 = w1 -alpha.*grad1;
%     
%     
%     l = mean_loss(X, Y, w0, w1);
%     
%     
%     if k >= kmax
%        break 
%     end
% end

plot(l);


% plot it
figure
hold on;

class1 = 1:size(X,1);
class2 = class1(Y==1);
class1 = class1(Y==-1);

%plot data
plot3(X(class1,2),X(class1,3),Y(class1),'b.');
plot3(X(class2,2),X(class2,3),Y(class2),'r.');

% display the function
[a, b] = meshgrid(-2.33:.1:2.33,-2.33:.1:2.33);
Xgrid = [a(:),b(:)];

%function
Ygrid = forward([ones(size(Xgrid,1),1),Xgrid]', w0, b1);

sigmoid = @(z) 1./(exp(-z) +1);
Ygrid   = sigmoid(Ygrid);

Ygrid = reshape(Ygrid,size(a));
h = surface(a,b,Ygrid);
view(3);
h.FaceAlpha = 0.7;
grid on;



%leaky RelU
function y = leakyReLU(x)
    y = max(0.01*x,x);
end

%TODO
%foreward
function y = forward(X,w0, b1)
    %z1 = w0*x;
    %y = w1*leakyReLU(z1);
    y = leakyReLU(plus(w0*X,b1));
end

function  [dl0, dl1] = backward(d_L2, x, w0, w1)

    x0 = x;

    sigmoid = @(z) 1./(exp(-z) +1);

    z1 = w0*x0;
    x1 = sigmoid(z1);

    d_L1 = (d_L2 * w1) .* (x1 .* (1 - x1))';

    %d_L0 = (d_L1 * w0) .* (x0 .* (1 - x0))';

    dl0 = d_L1' * x0';

    dl1 = d_L2' * x1';

end

function [grad0, grad1] = grad(X, Y, w0, w1)

    grad0 = zeros(size(w0));
    grad1 = zeros(size(w1));

    for i = 1:size(X,1)

        x = X(i,:)';
        y = Y(i);
        f = forward(x,w0, w1);

        if (1 - y * f  > 0)
            d_L2 = -y;
        else
            d_L2 = 0;
        end


        [dl0, dl1] = backward(d_L2, x, w0, w1);

        grad0 = grad0 + dl0;
        grad1 = grad1 + dl1;
    end

end


% function l = mean_loss(X, Y, w0, w1)
%     l = 0;
%     for i = 1:size(X,1)
%         x = X(i,:)';
%         y = Y(i);
%         f = forward(x,w0, w1);
%         l = max(0, 1-(f-y));
%     end
%     
%     l = l/size(X,1);
% end

function l = mean_loss(X, Y, w0, w1)
    l = 0;
    f = forward(X,w0, w1);
    l = max(0, 1-(f-Y));
end