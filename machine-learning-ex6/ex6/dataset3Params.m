function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C1 = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
sigma1 = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
error1=zeros(8,8);
pred1=zeros(length(y));
for i=1:8
  for j=1:8
    model= svmTrain(X, y, C1(i), @(x1, x2) gaussianKernel(x1, x2, sigma1(j))); %%try every combination of C&sigma in train set
    pred1 = svmPredict(model, Xval);%%use the model and cross validation set to predict yval
    error1(i,j)=mean(double(pred1~=yval));%% calculate the error between prediction and real yval
    j++;
  endfor
  i++;
endfor
[indc,indr]=find(error1==min(min(error1)));%%find the index of minus error
C=C1(indc);sigma=sigma1(indr);%% use the index to find optimal C&sigma






% =========================================================================

end
