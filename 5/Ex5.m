close all;
clear;
clc;

load('data2Class.txt');
data = data2Class;

X = data(:,1:2);
Y = data(:,3);


h1 = 100;

% initialize weights with Gaussian with sdv √(1/h_{l-1[)
w0 = normrnd(0,1/sqrt(h1-1),[h1,size(X,2)]);

% ini biases U(−1, 1) uniform distribution
w1 = rand([1,h1])*2 -1;


% gradient descent
alpha = 0.05;
k = 0;
kmax = 50;

l = mean_loss(X, Y, w0, w1);

while (true)
    k = k+1;
    [grad0, grad1] = grad(X, Y, w0, w1);
    
    w0 = w0 -alpha.*grad0;
    w1 = w1 -alpha.*grad1;
    
    
    l = [l, mean_loss(X, Y, w0, w1)];
    
    
    if k >= kmax
       break 
    end
end


plot(0:k, l);


% plot it
figure
hold on;

class1 = 1:size(X,1);
class2 = class1(Y==1);
class1 = class1(Y==0);

plot3(X(class1,1),X(class1,2),Y(class1),'b.');
plot3(X(class2,1),X(class2,2),Y(class2),'r.');

% display the function
[a, b] = meshgrid(-3:.1:3,-3:.1:3);
Xgrid = [b(:)]; %[a(:),b(:)];

Ygrid = forward([ones(size(Xgrid,1),1),Xgrid]', w0, w1);

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


function y = forward(x,w0, w1)
    z1 = w0*x;
    y = w1*leakyReLU(z1);
    %z1 = w0*x+w1;
    %y = sigmoid(z1);

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


function l = mean_loss(X, Y, w0, w1)
    l = 0;
    for i = 1:size(X,1)
        x = X(i,:)';
        y = Y(i);
        f = forward(x,w0, w1);
        
        l = l + max(0, 1-(f-y)); %hinge loss
    end
    
    l = l/size(X,1);
end
