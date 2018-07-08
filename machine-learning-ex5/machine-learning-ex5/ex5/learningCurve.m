function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);
r=size(Xval,1);


% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);
merror_train = zeros(m, 1);
merror_val   = zeros(m, 1);
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------
##Inx=randperm(m);
##for i:m
##  X1(i,:)=X(inx(i),:);
##  y1(i)=y(inx(i));%%顺序打乱
##  i++;
##endfor
##T=floor(m*0.6);
##Xtrain=X1(1:T,:);Ytrian=y(1:T,:);%% 60%train set 
##
##V=floor(m*0.8); 
##Xcross_val=X1((T+1):V,:);
##Ycross_val=y1((T+1):V);%% 20%cross validation set
##
##Xtest=X1((V+1):m,:);
##Ytest=y1((V+1):m)%%20%test set
initial_theta=rand(size(X,2),1);

for i=1:m
  %[Jtrain,Gtrain]=linearRegCostFunction(X(1:i,:), y(1:i), initial_theta,lambda);
  % Create "short hand" for the cost function to be minimized
costFunction = @(t) linearRegCostFunction(X(1:i,:), y(1:i), t, lambda);
##% Now, costFunction is a function that takes in only one argument
options = optimset('MaxIter', 200, 'GradObj', 'on');
##
##% Minimize using fmincg
theta = fmincg(costFunction, initial_theta, options);
  %[theta] = trainLinearReg(X(1:i,:), y(1:i), lambda);
  %%使用i个训练集算出theta
  [Jtrain,Gtrain]=linearRegCostFunction(X(1:i,:), y(1:i), theta,0);
  error_train(i)=Jtrain;%%theta 代回训练集算训练集误差error_train
  [Jval,Gval]=linearRegCostFunction(Xval, yval, theta,0);
   %%theta代入验证集算验证集误差error_val
  error_val(i)=Jval;
  [merror_train(i),merror_val(i)]=meanLearingCurve(X(1:i,:), y(1:i),Xval, yval, lambda);
  i++;
  
endfor











% -------------------------------------------------------------

% =========================================================================

end
