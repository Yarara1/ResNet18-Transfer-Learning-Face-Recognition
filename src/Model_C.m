%% Model C: Fine-tune ALL convolution layers of ResNet-18

clc; clear; close all;

%% 1. Paths to your data
trainFolder = 'C:\Users\Yara\Downloads\Helperdata\face_dataset\facescrub_train';
testFolder  = 'C:\Users\Yara\Downloads\Helperdata\face_dataset\facescrub_test';

imdsTrain = imageDatastore(trainFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

imdsTest = imageDatastore(testFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

numClasses = numel(categories(imdsTrain.Labels));
disp("Number of subjects (classes): " + numClasses);   % should be 100

%% 2. Load pretrained ResNet-18
net = resnet18;   

inputSize = net.Layers(1).InputSize(1:2);

%% 3. Data preprocessing & augmentation

% Data augmentation for training set
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-10 10], ...
    'RandXTranslation',[-5 5], ...
    'RandYTranslation',[-5 5], ...
    'RandXReflection',true);

augTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation',imageAugmenter, ...
    'ColorPreprocessing','gray2rgb');

% No augmentation for test set
augTest  = augmentedImageDatastore(inputSize, imdsTest, ...
    'ColorPreprocessing','gray2rgb');

%% 4. Replace final FC + classification layer (100 outputs)

lgraph = layerGraph(net);
[learnableLayer, classLayer] = findLayersToReplace(lgraph);

newFC = fullyConnectedLayer(numClasses, ...
    'Name','fc100_faces', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);

newClass = classificationLayer('Name','face_classoutput');

lgraph = replaceLayer(lgraph, learnableLayer.Name, newFC);
lgraph = replaceLayer(lgraph, classLayer.Name, newClass);

%% 5. UNFREEZE ALL convolution layers (Model C)

layers = lgraph.Layers;
connections = lgraph.Connections;

for i = 1:numel(layers)
    layerName = layers(i).Name;

    % New FC and classification layer already have LR=10, keep it.
    isNewLayer = strcmp(layerName,'fc100_faces') || strcmp(layerName,'face_classoutput');

    if ~isNewLayer
        % ALLOW learning for ALL conv layers
        if isprop(layers(i),'WeightLearnRateFactor')
            layers(i).WeightLearnRateFactor = 1;
        end
        if isprop(layers(i),'BiasLearnRateFactor')
            layers(i).BiasLearnRateFactor = 1;
        end
    end
end

lgraph = createLgraphUsingConnections(layers, connections);

%% 6. Training options (manual 1-epoch loop for graphs)

miniBatchSize = 64;
numEpochs = 15;

baseOptions = trainingOptions('adam', ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', 1, ...                 % train 1 epoch at a time
    'InitialLearnRate', 1e-4, ...
    'Shuffle','every-epoch', ...
    'Verbose', false, ...
    'Plots','none', ...
    'ExecutionEnvironment','gpu');

% History containers
epochList        = zeros(numEpochs,1);
trainLossHistory = zeros(numEpochs,1);
testLossHistory  = zeros(numEpochs,1);
trainAccHistory  = zeros(numEpochs,1);
testAccHistory   = zeros(numEpochs,1);

%% 7. Train Model C with manual epoch loop

for epoch = 1:numEpochs
    fprintf('=== Training epoch %d/%d ===\n', epoch, numEpochs);

    if epoch == 1
        modelC = trainNetwork(augTrain, lgraph, baseOptions);
    else
        tmpLgraph = layerGraph(modelC);
        modelC = trainNetwork(augTrain, tmpLgraph, baseOptions);
    end

    % ---- Compute metrics ----
    % Training accuracy
    YTrainTrue = imdsTrain.Labels;
    YTrainPred = classify(modelC, augTrain);
    trainAcc = mean(YTrainPred == YTrainTrue) * 100;

    % Training loss
    scoresTrain = predict(modelC, augTrain);
    TTrain = onehotencode(YTrainTrue,2);
    trainLoss = crossentropy(TTrain', scoresTrain');

    % Test accuracy
    YTestTrue = imdsTest.Labels;
    YTestPred = classify(modelC, augTest);
    testAcc = mean(YTestPred == YTestTrue) * 100;

    % Test loss
    scoresTest = predict(modelC, augTest);
    TTest = onehotencode(YTestTrue,2);
    testLoss = crossentropy(TTest', scoresTest');

    % Store
    epochList(epoch)        = epoch;
    trainLossHistory(epoch) = trainLoss;
    testLossHistory(epoch)  = testLoss;
    trainAccHistory(epoch)  = trainAcc;
    testAccHistory(epoch)   = testAcc;

    fprintf("Epoch %d: TrainAcc=%.2f%%  TestAcc=%.2f%%  TrainLoss=%.4f  TestLoss=%.4f\n", ...
        epoch, trainAcc, testAcc, trainLoss, testLoss);
end

%% 8. Final accuracies (for report)

finalTrainAcc = trainAccHistory(end);
finalTestAcc  = testAccHistory(end);

fprintf("\nModel C FINAL Training Accuracy: %.2f %%\n", finalTrainAcc);
fprintf("Model C FINAL Testing Accuracy : %.2f %%\n", finalTestAcc);

%% 9. Plots for report

% (ii) Training vs Test Loss
figure;
plot(epochList, trainLossHistory, '-o', 'LineWidth', 2); hold on;
plot(epochList, testLossHistory, '-o', 'LineWidth', 2);
xlabel('Epoch'); ylabel('Cross-Entropy Loss');
title('Training and Test Loss vs. Epoch (Model C)');
legend('Training Loss','Test Loss','Location','best');
grid on;

% (iii) Training vs Test Accuracy
figure;
plot(epochList, trainAccHistory, '-o', 'LineWidth', 2); hold on;
plot(epochList, testAccHistory, '-o', 'LineWidth', 2);
xlabel('Epoch'); ylabel('Accuracy (%)');
title('Training and Test Accuracy vs. Epoch (Model C)');
legend('Training Accuracy','Test Accuracy','Location','best');
grid on;

%% ------------------------------------------------------------------------
% Helper function: rebuild layerGraph
function lgraph = createLgraphUsingConnections(layers, connections)
lgraph = layerGraph();
for i = 1:numel(layers)
    lgraph = addLayers(lgraph, layers(i));
end
for c = 1:size(connections,1)
    lgraph = connectLayers(lgraph, ...
        connections.Source{c}, connections.Destination{c});
end
end

% Find final learnable + classification layer
function [learnableLayer, classLayer] = findLayersToReplace(lgraph)
layers = lgraph.Layers;
isClassificationLayer = arrayfun(@(l) isa(l,'nnet.cnn.layer.ClassificationOutputLayer'), layers);
classLayer = layers(isClassificationLayer);
isLearnable = arrayfun(@(l) isa(l,'nnet.cnn.layer.FullyConnectedLayer') || ...
                            isa(l,'nnet.cnn.layer.Convolution2DLayer'), layers);
learnableLayer = layers(find(isLearnable,1,'last'));
end
