% Neural Network class to contain and control layer classes
classdef neuralNet < handle
    properties
        Layers;
        Count;
        Cats;
        plotNum;
        N = 1;
    end
    methods
        % get Layers as input and initialize layers
        function nn = neuralNet(Layers, plotNum)
            nn.plotNum = plotNum;
            nn.Layers = Layers;
            nn.Count = numel(Layers);
            nn.init();
        end
        function init(nn)
            for L = 1:nn.Count
                if(L == 1)
                    nn.Layers{L}.init([],nn.Layers{L+1});
                elseif(L == nn.Count)
                    nn.Layers{L}.init(nn.Layers{L-1},[]);
                else
                    nn.Layers{L}.init(nn.Layers{L-1},nn.Layers{L+1});
                end
            end
            nn.N = 1;
        end
        % set the number of categories
        function setCats(nn, Cats)
            nn.Cats = Cats;
        end
        % forward pass input data
        % return output data as categories
        % NOTE: currently set up for classification problems
        function [Output, M] = feedForward(nn,Input,drop)
            nn.Layers{1}.setInputs(Input);
            for L = 1:nn.Count-1
                if(isa(nn.Layers{L},'dropoutLayer'))
                   if(drop == 1)
                       nn.Layers{L}.reset();
                   else
                       nn.Layers{L}.one();
                   end
                end
                nn.Layers{L}.forwardPass();
            end
            Output = nn.Layers{nn.Count}.Output();
            [M, I] = max(Output);
            Output = nn.Cats(I);
        end
        % calculate output error and return error
        function E = calcError(nn, Targets)
            E = nn.Layers{nn.Count}.calcError(Targets);
        end
        % function to train network and display progress
        function Matrix = confusionMatrix(nn,testData,testTargs)
            testTargs = testTargs';
            Outputs = zeros(size(testTargs));
            m = size(testTargs,2)/200-1;
            for i = 0:m
                Outputs(:,1+200*i:200+200*i) = nn.feedForward(testData(:,:,:,1+200*i:200+200*i),0);
                disp(num2str(i/m));
            end
            
            Matrix = num2cell(zeros(numel(nn.Cats)+1));
            for i = 2:numel(nn.Cats)+1
                Matrix{i,1} = nn.Cats(i-1);
            end
            for j = 2:numel(nn.Cats)+1
                Matrix{1,j} = nn.Cats(j-1);
            end
            
            for i = 1:numel(Outputs)
                [~,I] = max(nn.Cats==testTargs(i));
                J = Outputs(i);
                Matrix{I+1,J+1} = Matrix{I+1,J+1} + 1;
            end

        end
        function trainNetwork(nn,trainData,trainTargs,testData,testTargs,LR,M1,M2,maxEpochs,miniBatchSize)
            K = 0.999;
            if(isempty(nn.Cats))
                nn.Cats = cell2mat(categories([trainTargs;testTargs]));
            end
            if(isa(trainTargs,'categorical'))
                exc = zeros(numel(nn.Cats),numel(trainTargs));
                for i = 1:numel(nn.Cats)
                    exc(i,:) = (trainTargs==nn.Cats(i));
                end
                trainTargs = K*exc+(1-K)/numel(nn.Cats);
            end
            if(isa(testTargs,'categorical'))
                exc = zeros(numel(nn.Cats),numel(testTargs));
                for i = 1:numel(nn.Cats)
                    exc(i,:) = (testTargs==nn.Cats(i));
                end
                testTargs = K*exc+(1-K)/numel(nn.Cats);
            end
            
            testError = zeros(100,1);
            trainError = zeros(100,1);
            
            testAccAvgG = zeros(100,1);
            trainAccAvgG = zeros(100,1);
            
            testAccG = zeros(100,1);
            trainAccG = zeros(100,1);
            
            batchTest = testData;
            batchTestTarg = testTargs;
            
            testAccAvg = 0;
            testNum = 0;
            trainAccAvg = 0;
            trainNum = 0;
            lastTestAccAvg = 0;
            lastTrainAccAvg = 0;
    
            for i = 1:maxEpochs
                batchTrain = trainData;
                batchTrainTarg = trainTargs;
                
                while(((nn.Layers{1}.DimCount == 0) && (size(batchTrain,2)>0))||((nn.Layers{1}.DimCount == 2) && (size(batchTrain,4)>0)))
                    if(nn.Layers{1}.DimCount == 0)
                        if(size(batchTest,2) < miniBatchSize)
                            if(size(batchTest,2) > 0)
                                miniBatchTest = batchTest;
                                miniBatchTestTarg = batchTestTarg;
                                batchTest = testData;
                                batchTestTarg = testTargs;
                            else
                                batchTest = testData;
                                batchTestTarg = testTargs;
                                [miniBatchTest,ids] = datasample(batchTest,miniBatchSize,2,'Replace',false);
                                miniBatchTestTarg = batchTestTarg(:,ids);
                                batchTest(:,ids)=[];
                                batchTestTarg(:,ids)=[];
                            end
                        else
                            [miniBatchTest,ids] = datasample(batchTest,miniBatchSize,2,'Replace',false);
                            miniBatchTestTarg = batchTestTarg(:,ids);
                            batchTest(:,ids)=[];
                            batchTestTarg(:,ids)=[];
                        end
                        
                        if(size(batchTrain,2) < miniBatchSize)
                            miniBatchTrain = batchTrain;
                            miniBatchTrainTarg = batchTrainTarg;
                            batchTrain = [];
                            batchTrainTarg = [];
                        else
                            [miniBatchTrain,ids] = datasample(batchTrain,miniBatchSize,nn.Layers{1}.DimCount+2,'Replace',false);
                            miniBatchTrainTarg = batchTrainTarg(:,ids);
                            batchTrain(:,ids)=[];
                            batchTrainTarg(:,ids)=[];
                        end
                    elseif(nn.Layers{1}.DimCount == 2)
                        if(size(batchTest,4) < miniBatchSize)
                            if(size(batchTest,4) > 0)
                                miniBatchTest = batchTest;
                                miniBatchTestTarg = batchTestTarg;
                                batchTest = testData;
                                batchTestTarg = testTargs;
                            else
                                batchTest = testData;
                                batchTestTarg = testTargs;
                                [miniBatchTest,ids] = datasample(batchTest,miniBatchSize,4,'Replace',false);
                                miniBatchTestTarg = batchTestTarg(:,ids);
                                batchTest(:,:,:,ids)=[];
                                batchTestTarg(:,ids)=[];
                            end
                        else
                            [miniBatchTest,ids] = datasample(batchTest,miniBatchSize,4,'Replace',false);
                            miniBatchTestTarg = batchTestTarg(:,ids);
                            batchTest(:,:,:,ids)=[];
                            batchTestTarg(:,ids)=[];
                        end
                        
                        if(size(batchTrain,4) < miniBatchSize)
                            miniBatchTrain = batchTrain;
                            miniBatchTrainTarg = batchTrainTarg;
                            batchTrain = [];
                            batchTrainTarg = [];
                        else
                            [miniBatchTrain,ids] = datasample(batchTrain,miniBatchSize,4,'Replace',false);
                            miniBatchTrainTarg = batchTrainTarg(:,ids);
                            batchTrain(:,:,:,ids)=[];
                            batchTrainTarg(:,ids)=[];
                        end
                    end
                    
                    % Get Accuracy of testing data
                    testOut = nn.feedForward(miniBatchTest,0);
                    testError(1:end-1) = testError(2:end);
                    testError(end) = nn.calcError(miniBatchTestTarg);
                    [~, I] = max(miniBatchTestTarg);
                    miniBatchTestTarg2 = nn.Cats(I);
                    testAcc = sum(testOut == miniBatchTestTarg2);
                    testAccAvg = testAccAvg + testAcc;
                    testNum = testNum + numel(testOut);
                    
                    testAccG(1:end-1) = testAccG(2:end);
                    testAccG(end) = testAcc/numel(testOut);
                    
                    testAccAvgG(1:end-1) = testAccAvgG(2:end);
                    testAccAvgG(end) = testAccAvg/testNum;
                    
                    % Get Accuracy of training data
                    trainOut = nn.feedForward(miniBatchTrain, 1);
                    trainError(1:end-1) = trainError(2:end);
                    trainError(end) = nn.calcError(miniBatchTrainTarg);
                    [~, I] = max(miniBatchTrainTarg);
                    miniBatchTrainTarg2 = nn.Cats(I);
                    trainAcc = sum(trainOut == miniBatchTrainTarg2);
                    trainAccAvg = trainAccAvg + trainAcc;
                    trainNum = trainNum + numel(trainOut);
                    
                    trainAccG(1:end-1) = trainAccG(2:end);
                    trainAccG(end) = trainAcc/numel(trainOut);
                    
                    trainAccAvgG(1:end-1) = trainAccAvgG(2:end);
                    trainAccAvgG(end) = trainAccAvg/trainNum;
                    
                    % Train with Training data
                    % Test data is only used for testing
                    for L = nn.Count:-1:2
                        nn.Layers{L}.backPass(LR,M1,M2,nn.N);
                    end
                    
                    nn.N = nn.N + 1;
                    
                    figure(nn.plotNum*2-1)
                    subplot(2,1,1)
                    plot(testError,'r')
                    hold on
                    plot(trainError,'b')
                    hold off
                    
                    if(max(10^ceil(log(max(testError))/log(10)),10^ceil(log(max(trainError))/log(10)))<=0)
                        ylim([0,1])
                    else
                        ylim([0,max(10^ceil(log(max(testError))/log(10)),10^ceil(log(max(trainError))/log(10)))])
                    end
                    
                    title({['Epoch : ',num2str(i),'     |     ','Iteration : ',num2str(nn.N)];
                           ['Epoch Error','     |     ','Remaining Data Set'];
                           ['testError : ',num2str(testError(end),15),'     |     ','batchTest Size: ',num2str(size(batchTestTarg,2))];
                           ['trainError: ',num2str(trainError(end),15),'     |     ','batchTrain Size: ',num2str(size(batchTrainTarg,2))]});
                    
                    subplot(2,1,2)
                    plot(testAccG,'r-.')
                    hold on
                    plot(trainAccG,'b-.')
                    plot(testAccAvgG,'r')
                    plot(trainAccAvgG,'b')                    
                    hold off
                    
                    if(min(1-10^ceil(log(max(1-testAccG))/log(10)),1-10^ceil(log(max(1-trainAccG))/log(10)))>=1)
                        ylim([0,1])
                    else
                        ylim([min(1-10^ceil(log(max(1-testAccG))/log(10)),1-10^ceil(log(max(1-trainAccG))/log(10))),1])
                    end
                    
                    title({['Current Accuracy','     |     ','Accumulated Accuracy','     |     ','Last Accuracy of Entire Data Set'];
                           ['testAcc: ',num2str(testAcc/numel(testOut),15),'     |     ','testAccAvg: ',num2str(testAccAvg/testNum,15),'     |     ','  lastTestAccAvg: ',num2str(lastTestAccAvg,15)];
                           ['trainAcc: ',num2str(trainAcc/numel(trainOut),15),'     |     ','trainAccAvg: ',num2str(trainAccAvg/trainNum,15),'     |     ','  lastTrainAccAvg: ',num2str(lastTrainAccAvg,15)]});

                    drawnow limitrate
                    
                    if(size(testData,numel(size(testData)))==testNum)
                        lastTestAccAvg = testAccAvg/testNum;
                        testAccAvg = 0;
                        testNum = 0;
                    end
                    if(size(trainData,numel(size(trainData)))==trainNum)
                        lastTrainAccAvg = trainAccAvg/trainNum;
                        trainAccAvg = 0;
                        trainNum = 0;
                    end
                    
                    %toc;
                    %tic;
                end
            end
            
        end
    end
end