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
z2=zeros(hidden_layer_size,m);za2=zeros(hidden_layer_size+1,m);
%%加偏置单元，add bias unit
delta_2=zeros(hidden_layer_size,m);
z3=zeros(num_labels,m);za3=zeros(num_labels,m);delta_3=zeros(num_labels,m);
yk=repmat(yyk',1,m);%%K hang m lie,yk
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
dTheta_1=zeros(hidden_layer_size, input_layer_size+1);
dTheta_2=zeros(num_labels, hidden_layer_size+1);

for i=1:m
  yk(:,i)=(yk(:,i)==y(i));%%形成y的逻辑阵列，num_labels行，m列，每列对应一个逻辑阵列,即第i个逻辑化后的输出y
  %z2(:,i)=Theta1*X(i,:)';%%
  a2=sigmoid(Theta1*X(i,:)');
  a2=[1;a2];
  za2(:,i)=a2;%%形成隐含层激励矩阵，hidden_layer_size + 1行，m列
  z3(:,i)=Theta2*a2;
  hx=sigmoid(Theta2*za2(:,i));
  za3(:,i)=hx;%%形成输出层矩阵，每列对应一个训练样本的输出，num_labels行，m列
  delta_3(:,i)=hx-yk(:,i);%%输出误差矩阵，每列对应一个训练样本的输出，num_labels行，m列
  delta_2(:,i)=Theta2'*delta_3(:,i).*za2(:,i).*(1-za2(:,i));%%隐含层误差矩阵
  %delta_2(:,i)=delta_2(2:end,i);
  dTheta_2=dTheta_2+delta_3(:,i)*za2(2:end,i)';
  dTheta_1=dTheta_1+delta_2(2:end,i)*X(i,2:end);
  
  
  J=J+-1*(yk(:,i)'*log(hx))+ (1-yk(:,i))'*log(1-hx));
  i++;
  
endfor

X1=X(:,2:end);


J=J/m;
T1=0,T2=0;
for i=1:hidden_layer_size
  T1=T1+Theta1(i,2:end)*Theta1(i,2:end)';
  i++;
endfor
for i=1:num_labels
  T2=T2+Theta2(i,2:end)*Theta2(i,2:end)';
  i++;
endfor
J=J+(T1+T2)*lambda/(2*m);



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

for t=1:m
  
endfor


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



grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
