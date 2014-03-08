
clear all; close all; clc

trainingData = csvread('cs-training.csv' , 1 , 1);
training_X = trainingData(:, 2:11); training_y = trainingData(:, 1);



sixCol = find(training_X(:, 5) ~= training_X(:, 5));
NsixCol = find(training_X(: , 5) == training_X(:, 5));

training_X(sixCol , 5) = 1.0 * sum(training_X(NsixCol , 5)) / size(NsixCol , 1);


eleCol = find(training_X(: , 10) ~= training_X(: , 10));
NeleCol = find(training_X(: , 10) == training_X(: , 10));


training_X(eleCol , 10) = 1.0 * sum(training_X(NeleCol , 10)) / size(NeleCol , 1);

[training_X , mu , sigma] = featureNormalize(training_X);

[m, n] = size(training_X);

training_X = [ones(m, 1) training_X];
initial_theta = zeros(n + 1, 1);

lambda = 1;

options = optimset('GradObj', 'on', 'MaxIter', 500);

[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, training_X, training_y, lambda)), initial_theta, options);
	

p = predict(theta, training_X);

fprintf('Train Accuracy: %f\n', mean(double(p == training_y)) * 100);


%"--------------------Test for TestingData-------------------------------"

testData = csvread('cs-test.csv' , 1 , 2);
test_X = testData;
[tm , tn] = size(test_X);

sixCol = find(test_X(:, 5) ~= test_X(: , 5));
NsixCol = find(test_X(: , 5) == test_X(: , 5));

test_X(sixCol , 5) = 1.0 * sum(test_X(NsixCol , 5)) / size(NsixCol , 1);


eleCol = find(test_X(: , 10) ~= test_X(: , 10));
NeleCol = find(test_X(: , 10) == test_X(: , 10));


test_X(eleCol , 10) = 1.0 * sum(test_X(NeleCol , 10)) / size(NeleCol , 1);

[test_X , mu , sigma] = featureNormalize(test_X);

test_X = [ones(tm , 1) test_X];



prediction = double(sigmoid(test_X * theta));

index = [1:tm]';

dlmwrite('result.csv' , [index , prediction]);









