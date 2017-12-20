%clear; clc;
%D:\GitHub\NeuralNetMatlab
% function to train network with MNIST data set
% requires 8+ GB of ram depending on how many example you train at once
%tic
% rng(1345487458);
format

load('EMNIST\emnist-mnist.mat')

trainImagesDigits = double(dataset.train.images)/255.0;
trainImagesDigits = permute(trainImagesDigits,[2,1]);
trainImagesDigits = reshape(trainImagesDigits,[28,28,1,size(trainImagesDigits,2)]);
trainImagesDigits(trainImagesDigits <  128/255)=0;
trainImagesDigits(trainImagesDigits >= 128/255)=1;
trainLabelsDigits = categorical(dataset.train.labels);

testImagesDigits = double(dataset.test.images)/255.0;
testImagesDigits = permute(testImagesDigits,[2,1]);
testImagesDigits = reshape(testImagesDigits,[28,28,1,size(testImagesDigits,2)]);
testImagesDigits(testImagesDigits <  128/255)=0;
testImagesDigits(testImagesDigits >= 128/255)=1;
testLabelsDigits = categorical(dataset.test.labels);

trainImages = trainImagesDigits;
testImages = testImagesDigits;

trainLabels = trainLabelsDigits;
testLabels = testLabelsDigits;

%parallelLayer({fullyConnectedLayer(512),fullyConnectedLayer(512),fullyConnectedLayer(512)}),...
%concatLayer(3,1),...

LR = 0.001;
M1 = 0.9;
M2 = 0.999;
miniBatchSize = 2;
Gens = 500;
P1 = 0.05;
P2 = 0.05;
P3 = 0.05;
P4 = 0.05;
Pop = 50;
maxEpochs = 100*Pop*miniBatchSize/size(testImages,2);

%build structure
% Layers = {imageInputLayer([28,28,1]),...
%           convolution2dLayer([5 5], 32, [1 1], [2 2]),...
%           reLuLayer(),...      
%           maxPoolingLayer([2,2],[2,2],[0 0]),...           
%           convolution2dLayer([5 5], 64, [1 1], [2 2]),...
%           reLuLayer(),...
%           maxPoolingLayer([2,2],[2,2],[0 0]),... 
%           evolveLayer(1024, 20, 0.05, 0.05, 0.05, 0.05, 100),...
%           reLuLayer(),...
%           dropoutLayer(0.5),...
%           fullyConnectedLayer(10),...
%           softmaxLayer(),...
%           classificationOutputLayer(20)
%           };
Layers1 = {imageInputLayer([28,28,1]),...
          evolveLayer(1024, Pop, P1, P2, P3, P4, Gens, [ 100, 100], [ 5, 5], 1),...
          fullyConnectedLayer(10),...
          softmaxLayer(),...
          classificationOutputLayer(Pop)
          };

%initialize network
NN1 = neuralNet(Layers1, 1);
NN1.setCats(categorical({'0','1','2','3','4','5','6','7','8','9'}));
NN1.trainNetwork(trainImages,trainLabels,testImages,testLabels,LR,M1,M2,maxEpochs,miniBatchSize)

maxEpochs = 100*miniBatchSize/size(testImages,2);

Layers2 = {imageInputLayer([28,28,1]),...
          Layers1.getBest(1),...
          fullyConnectedLayer(10),...
          softmaxLayer(),...
          classificationOutputLayer(Pop)
          };

%initialize network
NN2 = neuralNet(Layers2, 1);
NN2.setCats(categorical({'0','1','2','3','4','5','6','7','8','9'}));
NN2.trainNetwork(trainImages,trainLabels,testImages,testLabels,LR,M1,M2,maxEpochs,miniBatchSize)

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