function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;J2=0;
hx=sigmoid(X*theta);
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i=1:m
  J=J+(-y(i)*log(hx(i))-(1-y(i))*log(1-hx(i)));
  i++;
endfor
J=J/m;%ji suan wei chengfa
for j=2:size(theta)
  J2=J2+theta(j)*theta(j);
  j++;
endfor
J2=lambda*J2/(2*m);%chengfa xiang

J=J+J2;% cost function

for i=1:m
    grad(1)=grad(1)+((hx(i)-y(i))*X(i,1));
    i++;
  endfor
  grad(1)=grad(1)/m; %theta0 piandao
  
  for j=2:size(theta)
  for i=1:m
    grad(j)=grad(j)+((hx(i)-y(i))*X(i,j));
    i++;
  endfor
  grad(j)=grad(j)/m+lambda*theta(j)/m;
  j++;
  
endfor
  






% =============================================================

end
