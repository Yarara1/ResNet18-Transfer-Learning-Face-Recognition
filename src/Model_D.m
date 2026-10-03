%% Model D: Freeze all conv blocks, add 2 FC layers (2-hidden-layer MLP)

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
net = resnet18;   % requires Deep Learning Toolbox Model for ResNet-18

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

% No augmentation for test set (just resize & color handling)
augTest  = augmentedImageDatastore(inputSize, imdsTest, ...
    'ColorPreprocessing','gray2rgb');

%% 4. Build 2-hidden-layer MLP head (remove old head completely)

lgraph = layerGraph(net);

% Find the original last learnable layer (fc1000)
[oldLearnableLayer, oldClassLayer] = findLayersToReplace(lgraph);

% Find the layer feeding into fc1000 (true last feature layer)
conn = lgraph.Connections;
idx = find(strcmp(conn.Destination, oldLearnableLayer.Name));
lastFeatureLayer = conn.Source{idx};   % <-- the correct pool5 name automatically

% Remove original classifier head
layersToRemove = unique({oldLearnableLayer.Name, oldClassLayer.Name, 'prob'});
existing = {lgraph.Layers.Name};
lgraph = removeLayers(lgraph, intersect(layersToRemove, existing));

% NEW MLP HEAD
fc1Size = 512;
fc2Size = 256;

newHead = [
    fullyConnectedLayer(fc1Size, 'Name','fc1_faces', ...
        'WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    reluLayer('Name','relu1_faces')
    dropoutLayer(0.5,'Name','drop1_faces')
    fullyConnectedLayer(fc2Size, 'Name','fc2_faces', ...
        'WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    reluLayer('Name','relu2_faces')
    dropoutLayer(0.5,'Name','drop2_faces')
    fullyConnectedLayer(numClasses, 'Name','fc3_faces', ...
        'WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    classificationLayer('Name','face_classoutput')
];

lgraph = addLayers(lgraph, newHead);

% Connect correct last conv features → new head
lgraph = connectLayers(lgraph, lastFeatureLayer, 'fc1_faces');


%% 5. Freeze ALL conv blocks, train only new FC layers + softmax

layers = lgraph.Layers;
connections = lgraph.Connections;

for i = 1:numel(layers)
    layerName = layers(i).Name;

    % New trainable layers in the MLP head
    isNewLayer = any(strcmp(layerName, ...
        {'fc1_faces','relu1_faces','drop1_faces', ...
         'fc2_faces','relu2_faces','drop2_faces', ...
         'fc3_faces','face_classoutput'}));

    if ~isNewLayer
        % Freeze all backbone layers
        if isprop(layers(i),'WeightLearnRateFactor')
            layers(i).WeightLearnRateFactor = 0;
        end
        if isprop(layers(i),'BiasLearnRateFactor')
            layers(i).BiasLearnRateFactor = 0;
        end
        if isprop(layers(i),'ScaleLearnRateFactor')
            layers(i).ScaleLearnRateFactor = 0;
        end
        if isprop(layers(i),'OffsetLearnRateFactor')
            layers(i).OffsetLearnRateFactor = 0;
        end
    else
        % Allow learning in the new MLP head
        if isprop(layers(i),'WeightLearnRateFactor')
            layers(i).WeightLearnRateFactor = 1;
        end
        if isprop(layers(i),'BiasLearnRateFactor')
            layers(i).BiasLearnRateFactor = 1;
        end
    end
end

lgraph = createLgraphUsingConnections(layers, connections);

%% 6. Training options (manual 1-epoch loop for plotting)

miniBatchSize = 64;
numEpochs = 15;

baseOptions = trainingOptions('adam', ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', 1, ...                 % train 1 epoch per call
    'InitialLearnRate', 1e-4, ...
    'Shuffle','every-epoch', ...
    'Verbose', false, ...
    'Plots','none', ...
    'ExecutionEnvironment','gpu');      % use GPU if available

% History containers
epochList        = zeros(numEpochs,1);
trainLossHistory = zeros(numEpochs,1);
testLossHistory  = zeros(numEpochs,1);
trainAccHistory  = zeros(numEpochs,1);
testAccHistory   = zeros(numEpochs,1);

%% 7. Train Model D with manual epoch loop

for epoch = 1:numEpochs
    fprintf('=== Training epoch %d/%d ===\n', epoch, numEpochs);

    if epoch == 1
        % First epoch from initial lgraph
        modelD = trainNetwork(augTrain, lgraph, baseOptions);
    else
        % Continue training from previous weights
        tmpLgraph = layerGraph(modelD);
        modelD = trainNetwork(augTrain, tmpLgraph, baseOptions);
    end

    % ---- Compute metrics for this epoch ----

    % Training accuracy
    YTrainTrue = imdsTrain.Labels;
    YTrainPred = classify(modelD, augTrain);
    trainAcc = mean(YTrainPred == YTrainTrue) * 100;

    % Training loss (cross-entropy)
    scoresTrain = predict(modelD, augTrain);   % N x K probabilities
    TTrain = onehotencode(YTrainTrue,2);       % N x K one-hot targets
    trainLoss = crossentropy(TTrain', scoresTrain');   % K x N

    % Test accuracy
    YTestTrue = imdsTest.Labels;
    YTestPred = classify(modelD, augTest);
    testAcc = mean(YTestPred == YTestTrue) * 100;

    % Test loss (cross-entropy)
    scoresTest = predict(modelD, augTest);     % N x K
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

%% 8. Final accuracies (last epoch)

finalTrainAcc = trainAccHistory(end);
finalTestAcc  = testAccHistory(end);

fprintf("\nModel D FINAL Training Accuracy: %.2f %%\n", finalTrainAcc);
fprintf("Model D FINAL Testing Accuracy : %.2f %%\n", finalTestAcc);

%% 9. Plots for report

% (ii) Training vs Test Loss
figure;
plot(epochList, trainLossHistory, '-o', 'LineWidth', 2); hold on;
plot(epochList, testLossHistory, '-o', 'LineWidth', 2);
xlabel('Epoch'); ylabel('Cross-Entropy Loss');
title('Training and Test Loss vs. Epoch (Model D)');
legend('Training Loss','Test Loss','Location','best');
grid on;

% (iii) Training vs Test Accuracy
figure;
plot(epochList, trainAccHistory, '-o', 'LineWidth', 2); hold on;
plot(epochList, testAccHistory, '-o', 'LineWidth', 2);
xlabel('Epoch'); ylabel('Accuracy (%)');
title('Training and Test Accuracy vs. Epoch (Model D)');
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
layers = lgraph.Layers;

% 1) Classification layer
isClassificationLayer = arrayfun(@(l) ...
    isa(l, 'nnet.cnn.layer.ClassificationOutputLayer'), layers);
classLayer = layers(isClassificationLayer);

% 2) Last learnable layer (FC or Conv2D)
isLearnable = arrayfun(@(l) ...
    isa(l, 'nnet.cnn.layer.FullyConnectedLayer') || ...
    isa(l, 'nnet.cnn.layer.Convolution2DLayer'), layers);
learnableLayer = layers(find(isLearnable, 1, 'last'));
end
