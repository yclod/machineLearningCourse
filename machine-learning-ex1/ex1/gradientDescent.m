function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    %fprintf('#%f [B] theta = %f\n', iter, theta);
    %tmp1 = theta(1) - alpha / m * sum((X * theta - y) .* X(:,1)); % tmp1 = theta(1) - alpha / m * (X(:, 1)' * (X * theta - y))
    %tmp2 = theta(2) - alpha / m * sum((X * theta - y) .* X(:,2)); % tmp2 = theta(2) - alpha / m * (X(:, 2)' * (X * theta - y))
    %theta = [tmp1; tmp2];
    theta = theta - alpha / m * (X' * (X * theta - y));
    %fprintf('#%f [A] theta0 = %f, theta1 = %f\n', iter, theta(1), theta(2));
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    fprintf('#%f J = %f\n', iter, J_history(iter));
end

end
