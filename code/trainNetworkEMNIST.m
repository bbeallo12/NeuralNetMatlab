clear; clc;
% function to train network with MNIST data set
% requires 8+ GB of ram depending on how many example you train at once
tic
rng(1345487458);
format long g

load('EMNIST\emnist-letters.mat')

trainImagesLetters = double(dataset.train.images)/255.0;
trainImagesLetters = permute(trainImagesLetters,[2,1]);
trainImagesLetters = reshape(trainImagesLetters,[28,28,1,size(trainImagesLetters,2)]);
trainImagesLetters(trainImagesLetters <  128/255)=0;
trainImagesLetters(trainImagesLetters >= 128/255)=1;
trainLabelsLetters = categorical(num2cell(char(dataset.train.labels+64)));

testImagesLetters = double(dataset.test.images)/255.0;
testImagesLetters = permute(testImagesLetters,[2,1]);
testImagesLetters = reshape(testImagesLetters,[28,28,1,size(testImagesLetters,2)]);
testImagesLetters(testImagesLetters <  128/255)=0;
testImagesLetters(testImagesLetters >= 128/255)=1;
testLabelsLetters = categorical(num2cell(char(dataset.test.labels+64)));

% load('EMNIST\emnist-digits.mat')
% 
% trainImagesDigits = double(dataset.train.images)/255.0;
% trainImagesDigits = permute(trainImagesDigits,[2,1]);
% trainImagesDigits = reshape(trainImagesDigits,[28,28,1,size(trainImagesDigits,2)]);
% trainImagesDigits(trainImagesDigits <  128/255)=0;
% trainImagesDigits(trainImagesDigits >= 128/255)=1;
% trainLabelsDigits = categorical(dataset.train.labels);
% 
% testImagesDigits = double(dataset.test.images)/255.0;
% testImagesDigits = permute(testImagesDigits,[2,1]);
% testImagesDigits = reshape(testImagesDigits,[28,28,1,size(testImagesDigits,2)]);
% testImagesDigits(testImagesDigits <  128/255)=0;
% testImagesDigits(testImagesDigits >= 128/255)=1;
% testLabelsDigits = categorical(dataset.test.labels);

% trainImages = cat(4,trainImagesDigits,trainImagesLetters);
% testImages = cat(4,testImagesDigits,testImagesLetters);
% 
% trainLabels = [trainLabelsDigits;trainLabelsLetters];
% testLabels = [testLabelsDigits;testLabelsLetters];

trainImages = trainImagesLetters;
testImages = testImagesLetters;

trainLabels = trainLabelsLetters;
testLabels = testLabelsLetters;

%build structure
Layers = {imageInputLayer([28,28,1]),...
          convolution2dLayer([5 5], 32, [1 1], [2 2]),...
          reLuLayer(),...      
          maxPoolingLayer([2,2],[2,2],[0 0]),...           
          convolution2dLayer([5 5], 64, [1 1], [2 2]),...
          reLuLayer(),...
          maxPoolingLayer([2,2],[2,2],[0 0]),... 
          fullyConnectedLayer(1024),...
          reLuLayer(),...
          dropoutLayer(0.5),...
          fullyConnectedLayer(26),...
          softmaxLayer(),...
          classificationOutputLayer()
          };
%initialize network
NN = neuralNet(Layers);

LR = 0.001;
M1 = 0.9;
M2 = 0.999;
maxEpochs = 20;
miniBatchSize = 200;

%NN.setCats(categorical({'0','1','2','3','4','5','6','7','8','9'}));
NN.setCats(categorical({'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'}));
%NN.setCats(categorical({'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'}));

NN.trainNetwork(trainImages,trainLabels,testImages,testLabels,LR,M1,M2,maxEpochs,miniBatchSize)

figure(2)
for i = 1:32
subplot(8,4,i)
imshow(NN.Layers{4}.Values(:,:,i,1));
end
figure(3)
for i = 1:64
subplot(8,8,i)
imshow(NN.Layers{7}.Values(:,:,i,1));
end