clear;
load('data2Class_adjusted.txt');
data = data2Class_adjusted;

X = data(:,1:3);
Y = data(:,4);


h1 = 100;

%TODO gaussian
w0 = rand([h1, size(X,2)])*2 -1;
b1 = rand([1,h1])*2 -1;


% gradient descent
alpha = 0.05;
k = 0;
kmax = 100;

for i = 1:size(X,1)
    X(:,i)

l = mean_loss(X, Y, w0, b1);

while (true)
    k = k+1;
    [grad0, grad1] = grad(X, Y, w0, b1);
    
    w0 = w0 -alpha.*grad0;
    b1 = b1 -alpha.*grad1;
    
    
    l = [l, mean_loss(X, Y, w0, b1)];
    
    
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
class1 = class1(Y==-1);

plot3(X(class1,2),X(class1,3),Y(class1),'b.');
plot3(X(class2,2),X(class2,3),Y(class2),'r.');

% display the function
[a, b] = meshgrid(-2:.1:2,-2:.1:2);
Xgrid = [a(:),b(:)];

Ygrid = forward([ones(size(Xgrid,1),1),Xgrid]', w0, b1);

sigmoid = @(z) 1./(exp(-z) +1);
Ygrid   = sigmoid(Ygrid);

Ygrid = reshape(Ygrid,size(a));
h = surface(a,b,Ygrid);
view(3);
h.FaceAlpha = 0.7;
grid on;





function y = forward(x,w0, w1)
    sigmoid = @(z) 1./(exp(-z) +1);

    z1 = w0*x;
    y = w1*sigmoid(z1);

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
        
        l = l + max(0, 1-f*y);
    end
    
    l = l/size(X,1);
end
