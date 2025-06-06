%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear regression exercise.
%
%  You will need to complete the following functions in this
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');
[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha).
%
%               Your task is to first make sure that your functions -
%               computeCost and gradientDescent already work with
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%

% Number of iterations for gradient descent
num_iters = 400;

% Init Theta
theta = zeros(3, 1);

% testing of the cost function
fprintf('\nTesting the cost function ...\n')
% compute and display initial cost
J = computeCostMulti(X, y, theta);
fprintf('With theta = [0 ; 0 ; 0]\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 65591548106.46\n');

% further testing of the cost function
J = computeCostMulti(X, y, [3.3e5 ; 1e5 ; 3.6e3]);
fprintf('\nWith theta = [330000 ; 100000 ; 36000]\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 2144490040.42\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nRunning gradient descent for different values of alpha...\n');

% Setting test values for alpha
alpha = [.3 .1 .03 .01 .003 .001];

for i = 1:length(alpha)
  fprintf('For alpha = %f\n', alpha(i));
  theta = zeros(3, 1);
  J_history = zeros(num_iters, 1);
  [theta, J_history] = gradientDescentMulti(X, y, theta, alpha(i), num_iters);

  % Display gradient descent's result
  fprintf('Theta computed from gradient descent: \n');
  fprintf(' %f \n', theta);
  fprintf('\n');

  % Plot the convergence graph
  if (i == 1)
    figure;
    plot(1:numel(J_history), J_history, 'LineWidth', 1);
    title('Learning Rate (alpha) tests')
    xlabel('Number of iterations');
    ylabel('Cost J');
  else
    hold on;
    plot(1:numel(J_history), J_history, 'LineWidth', 1);
  endif
end
legend(strsplit(num2str(alpha)))
hold off;

fprintf('\nRunning gradient descent for alpha = 0.3...\n');
alpha = 0.3;
theta = zeros(3, 1);
J_history = zeros(num_iters, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
price = 0; % You should change this
new_house = ([1650 3] - mu) ./ sigma; % normalizing data
price = [1 new_house]*theta;

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('\nSolving with normal equations...\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form
%               solution for linear regression using the normal
%               equations. You should complete the code in
%               normalEqn.m
%
%               After doing so, you should complete this code
%               to predict the price of a 1650 sq-ft, 3 br house.
%

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
price = 0; % You should change this
price = [1 1650 3] * theta;

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

