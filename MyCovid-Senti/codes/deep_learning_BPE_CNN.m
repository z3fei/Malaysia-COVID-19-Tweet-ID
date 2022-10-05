for i=1:10
clear;
digitDatasetPath = fullfile('D:\bpe');%change as necessary
numTrainFiles = 0.8;%change as necessary 
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

imageSize = [94 250 1]; %final v1

augimdsTrain = augmentedImageDatastore(imageSize,imdsTrain,'ColorPreprocessing','rgb2gray');
augimdsValidation = augmentedImageDatastore(imageSize,imdsValidation,'ColorPreprocessing','rgb2gray');
totalfiles=150;%change as necessary [not in use]



layers = [

    imageInputLayer([94 250 1]) %final
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,256,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,512,'Padding','same')
    batchNormalizationLayer
    reluLayer
 
    dropoutLayer(0.5)
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',15, ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'MiniBatchSize',64, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(augimdsTrain,layers,options);

%analyzeNetwork(net)

YPred = classify(net,augimdsValidation);
YValidation = imdsValidation.Labels;

figure
confusionchart(YValidation,YPred)

accuracy = sum(YPred == YValidation)/numel(YValidation)

stats = confusionmatStats(YValidation,YPred)
macro(1)=mean(stats.accuracy); %micro
macro(2)=mean(stats.sensitivity);
macro(3)=mean(stats.specificity);
macro(4)=mean(stats.precision);
macro(5)=mean(stats.recall);
macro(6)=mean(stats.Fscore); %macro
macro(7)=mean(stats.TP);
macro(8)=mean(stats.TN);
macro(9)=mean(stats.FP);
macro(10)=mean(stats.FN);
tot_samples=sum( stats.confusionMat , 'all' );
tot_negative=sum( stats.confusionMat(1,:), 'all' );
tot_neutral=sum( stats.confusionMat(2,:), 'all' );
tot_positive=sum( stats.confusionMat(3,:), 'all' );

weight_negative=tot_negative / tot_samples;
weight_neutral=tot_neutral / tot_samples;
weight_positive=tot_positive / tot_samples;
f1_weighted=(stats.Fscore(1)*weight_negative)+(stats.Fscore(2)*weight_neutral)+(stats.Fscore(3)*weight_positive);

filename=strcat('m',num2str(i),'.mat')
save(filename);

end
