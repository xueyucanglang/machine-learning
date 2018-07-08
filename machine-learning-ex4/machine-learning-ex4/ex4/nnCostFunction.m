function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X=[ones(m,1) X];
yyk=1:num_labels;%%yyk=1 2 3...K;
yk=repmat(yyk',1,m);%%K hang m lie,yk
z2=zeros(hidden_layer_size,m);za2=zeros(hidden_layer_size+1,m);
delta_2=zeros(hidden_layer_size,m);
z3=zeros(num_labels,m);za3=zeros(num_labels,m);delta_3=zeros(num_labels,m);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
dTheta_1=zeros(size(Theta1));
dTheta_2=zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients

for i=1:m
  yk(:,i)=(yk(:,i)==y(i));%%形成y的逻辑阵列，num_labels行，m列，每列对应一个逻辑阵列,即第i个逻辑化后的输出y
  x=X(i,:)';%%从X矩阵中取出第i个x
  a1=x;
  Y=yk(:,i);%%从y矩阵中取出第i个逻辑化后的y
  
  zz2=Theta1*x;
  a2=sigmoid(zz2);
  gz2=sigmoidGradient(zz2);
  a2=[1 ;a2];
  gz2=[1;gz2];
  hx=sigmoid(Theta2*a2);
  
  z2(:,i)=zz2;
  za2(:,i)=a2;%%放入隐含层激励矩阵中，hidden_layer_size + 1行，m列
  z3(:,i)=Theta2*a2;
  a3=hx;
  za3(:,i)=a3;%%放入输出层矩阵，每列对应一个训练样本的输出，num_labels行，m列
  del3=hx-Y;
  del2=Theta2'*del3.*gz2;
  del2=del2(2:end);%%去掉del20
  
  delta_3(:,i)=del3;%%放入输出误差矩阵，每列对应一个训练样本的输出，num_labels行，m列
  delta_2(:,i)=del2; %%放入隐含层误差矩阵中
  
  dTheta_1=dTheta_1+del2*a1';
  dTheta_2=dTheta_2+del3*a2';
  J=J+-1*(Y'*log(hx)+ (1-Y)'*log(1-hx));
  i++;
  
endfor
J=J/m;
Theta1_grad =dTheta_1/m;
Theta2_grad=dTheta_2/m;
TT1=0;TT2=0;
for i=1:hidden_layer_size
  TT1=TT1+Theta1(i,2:end)*Theta1(i,2:end)';
  i++;
endfor
for i=1:num_labels
  TT2=TT2+Theta2(i,2:end)*Theta2(i,2:end)';
  i++;
endfor
J=J+(TT1+TT2)*lambda/(2*m);

temp1=Theta1;
temp1(:,1)=zeros( hidden_layer_size,1);
Theta1_grad =Theta1_grad +temp1*lambda/m;%%正则化(regularize dTheta1)

temp2=Theta2;
temp2(:,1)=zeros(num_labels,1);
Theta2_grad =dTheta_2/m+temp2*lambda/m;%%正则化(regularize dTheta2),

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
