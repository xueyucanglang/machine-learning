## Copyright (C) 2018 rolis
## 
## This program is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see
## <https://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {} {@var{retval} =} meanLearingCurve (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: rolis <rolis@DESKTOP-0TLMO56>
## Created: 2018-06-24
function [merror_train, merror_val] = ...
    meanLearningCurve(X, y, Xval, yval, lambda)
    m = size(X, 1);r=size(Xval,1);
    M=floor(m/3);R=floor(r/3);
    merror_train = 0;
    merror_val   = 0;
    initial_theta=rand(size(X,2),1);
    for i=1:50%%重复取50次
    IDM=randperm(m);IDR=randperm(r);
    X1=X(IDM(1:M),:);y1=y(IDM(1:M),:)
    X1val=Xval(IDR(1:R),:);y1val=yval(IDR(1:R),:);
    costFunction = @(t) linearRegCostFunction(X1, y1, t, lambda);
    options = optimset('MaxIter', 200, 'GradObj', 'on');
   ##% Minimize using fmincg
    theta = fmincg(costFunction, initial_theta, options);
    [Jtrain,Gtrain]=linearRegCostFunction(X1, y1, theta,0);
  merror_train=merror_train+Jtrain;%%theta 代回训练集算训练集误差error_train
  [Jval,Gval]=linearRegCostFunction(X1val, y1val, theta,0);
   %%theta代入验证集算验证集误差error_val
  merror_val=merror_val+Jval;
  i++;
    endfor
    merror_train=merror_train/50;
    merror_val=merror_val/50;
      

endfunction
