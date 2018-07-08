function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));
a=size(z);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
for i=1:a(1)
  for j=1:a(2)
    g(i,j)=1/(1+e^(-z(i,j)));
    j++;
  endfor
  i++;
endfor





% =============================================================

end
