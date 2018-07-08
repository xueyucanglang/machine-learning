function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
%J=(X*theta-y)'*(X*theta-y)/(2*m);%标准化（线性代数法）
hx=zeros(m,1);%梯度下降法
for i=1:m
 hx(i)=X(i,:)*theta;
  J=J+(hx(i)-y(i))^2;
  i++;
end
J=J/(2*m);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.





% =========================================================================

end
