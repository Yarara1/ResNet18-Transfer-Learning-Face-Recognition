%% Model A: Fine-tune Conv5_x of ResNet-18 and log per-epoch metrics

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
disp("Number of subjects (classes): " + numClasses);   

%% 2. Load pretrained ResNet-18
net = resnet18;  

inputSize = net.Layers(1).InputSize(1:2);

% Resize + color handling 
augTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'ColorPreprocessing','gray2rgb');
augTest  = augmentedImageDatastore(inputSize, imdsTest, ...
    'ColorPreprocessing','gray2rgb');

%% 3. Replace final FC + classification layer (100 outputs)

lgraph = layerGraph(net);
[learnableLayer, classLayer] = findLayersToReplace(lgraph);

newFC = fullyConnectedLayer(numClasses, ...
    'Name','fc100_faces', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);

newClass = classificationLayer('Name','face_classoutput');

lgraph = replaceLayer(lgraph, learnableLayer.Name, newFC);
lgraph = replaceLayer(lgraph, classLayer.Name, newClass);

%% 4. Freeze all layers EXCEPT conv5_x and the new FC layer

layers = lgraph.Layers;
connections = lgraph.Connections;

% Conv5_x layers contain names with 'res5'
conv5_names = ["res5", "res5a", "res5b"];

for i = 1:numel(layers)
    layerName = layers(i).Name;

    % Check if layer is in conv5_x (any of the conv5-related names)
    isConv5 = any(contains(layerName, conv5_names, 'IgnoreCase', true));

    % Do not freeze FC or classification layer
    isNewLayer = strcmp(layerName,'fc100_faces') || strcmp(layerName,'face_classoutput');

    if ~(isConv5 || isNewLayer)
        % Freeze everything that is NOT conv5_x or the new FC
        if isprop(layers(i),'WeightLearnRateFactor')
            layers(i).WeightLearnRateFactor = 0;
        end
        if isprop(layers(i),'BiasLearnRateFactor')
            layers(i).BiasLearnRateFactor = 0;
        end
    else
        % Allow learning (fine-tune)
        if isprop(layers(i),'WeightLearnRateFactor')
            layers(i).WeightLearnRateFactor = 1;
        end
        if isprop(layers(i),'BiasLearnRateFactor')
            layers(i).BiasLearnRateFactor = 1;
        end
    end
end

lgraph = createLgraphUsingConnections(layers, connections);

%% 5. Training options (1 epoch at a time, so we can log per-epoch metrics)

miniBatchSize = 64;
numEpochs = 15;

baseOptions = trainingOptions('adam', ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', 1, ...                 % train 1 epoch per call
    'InitialLearnRate', 1e-4, ...       % smaller LR for fine-tuning conv layers
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

%% 6. Train Model A with manual epoch loop

for epoch = 1:numEpochs
    fprintf('=== Training epoch %d/%d ===\n', epoch, numEpochs);

    if epoch == 1
        % First epoch from initial lgraph
        modelA = trainNetwork(augTrain, lgraph, baseOptions);
    else
        % Continue training from previous model's weights
        tmpLgraph = layerGraph(modelA);
        modelA = trainNetwork(augTrain, tmpLgraph, baseOptions);
    end

    % Compute metrics for this epoch 
    % Training accuracy
    YTrainTrue = imdsTrain.Labels;
    YTrainPred = classify(modelA, augTrain);
    trainAcc = mean(YTrainPred == YTrainTrue) * 100;

    % Training loss (cross-entropy)
    scoresTrain = predict(modelA, augTrain);   % N x K probabilities
    TTrain = onehotencode(YTrainTrue,2);       % N x K one-hot targets
    trainLoss = crossentropy(TTrain', scoresTrain');   % K x N input to crossentropy

    % Test accuracy
    YTestTrue = imdsTest.Labels;
    YTestPred = classify(modelA, augTest);
    testAcc = mean(YTestPred == YTestTrue) * 100;

    % Test loss (cross-entropy)
    scoresTest = predict(modelA, augTest);     % N x K
    TTest = onehotencode(YTestTrue,2);         % N x K
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

%% 7. Final accuracies (last epoch)

finalTrainAcc = trainAccHistory(end);
finalTestAcc  = testAccHistory(end);

fprintf("\nModel A FINAL Training Accuracy: %.2f %%\n", finalTrainAcc);
fprintf("Model A FINAL Testing Accuracy : %.2f %%\n", finalTestAcc);

%% 8. Plots for report

% Training vs Test Loss
figure;
plot(epochList, trainLossHistory, '-o', 'LineWidth', 2); hold on;
plot(epochList, testLossHistory, '-o', 'LineWidth', 2);
xlabel('Epoch'); ylabel('Cross-Entropy Loss');
title('Training and Test Loss vs. Epoch (Model A)');
legend('Training Loss','Test Loss','Location','best');
grid on;

% Training vs Test Accuracy
figure;
plot(epochList, trainAccHistory, '-o', 'LineWidth', 2); hold on;
plot(epochList, testAccHistory, '-o', 'LineWidth', 2);
xlabel('Epoch'); ylabel('Accuracy (%)');
title('Training and Test Accuracy vs. Epoch (Model A)');
legend('Training Accuracy','Test Accuracy','Location','best');
grid on;

%% ------------------------------------------------------------------------
% Helper function: rebuild layerGraph from modified layers & connections
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

% Helper function: find last learnable layer + classification layer
function [learnableLayer, classLayer] = findLayersToReplace(lgraph)
% FINDLAYERSTOREPLACE Helper to find the last learnable layer and
% the classification layer in a layer graph.

layers = lgraph.Layers;

% 1) Find the classification layer
isClassificationLayer = arrayfun(@(l) ...
    isa(l, 'nnet.cnn.layer.ClassificationOutputLayer'), layers);
classLayer = layers(isClassificationLayer);

% 2) Find the last learnable layer (FC or Conv2D)
isLearnable = arrayfun(@(l) ...
    isa(l, 'nnet.cnn.layer.FullyConnectedLayer') || ...
    isa(l, 'nnet.cnn.layer.Convolution2DLayer'), layers);

learnableLayer = layers(find(isLearnable, 1, 'last'));
end
