function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
pos=find(y==1),neg=find(y==0);
plot(X(pos,1),X(pos,2),'k+','LineWidth',2,'MarkerSize',7);
plot(X(neg,1),X(neg,2),'ko','MarkerFacecolor','y','MarkerSize',7);
########aa=[];bb=[];
########for i=1:100
########  if y(i)==1
########    aa(i,:)=X(i,:);
########  else
########    bb(i,:)=X(i,:);
########  endif
########  i++;
########endfor
########plot(aa(:,1),aa(:,2),'k+','LineWidth',2,'MarkerSize',7)%
########plot(bb(:,1),bb(:,2),'ko','MarkerFacecolor','y','MarkerSize',7);







% =========================================================================



hold off;

end
