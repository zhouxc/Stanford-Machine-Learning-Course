function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%========================Compute costFunc=======================
H = sigmoid(X*theta);
T = y.*log(H) + (1 - y).*log(1 - H);
J = -1/m*sum(T) + lambda/(2*m)*sum(theta(2:end).^2);

%========================Compute Gradient======================

for i = 1 : m,
	grad = grad + (H(i) - y(i)) * X(i,:)';
end

ta = [0;theta(2:end)];

grad = 1/m*grad + lambda/m*ta;



% =============================================================

end
