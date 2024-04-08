S = load("olivetti.mat");
[X, label] = deal(S.X, S.label); % ta faces oi einai grammes tou X

trainImg = [];
testImg = [];
trainLabel = [];
testLabel = [];

for i = 1:10:400
    for j = 0:6 
        trainImg = [trainImg; X(i+j, :)]; 
        trainLabel = [trainLabel; label(i+j, 1)];
    end

    for k = 1:3
        testImg = [testImg; X(i+j+k, :)];
        testLabel = [testLabel; label(i+j+k, 1)];
    end
end

[coeff,score,~,~,explained] = pca(trainImg, 'Economy', false); 

numOfComponents = find(cumsum(explained) > 95, 1);
eigenFaces = coeff(:,1:numOfComponents);

meanFace = mean(trainImg);
centeredTest = testImg' - meanFace';
projectionTest = centeredTest' * eigenFaces;
projectionTrain = score(:,1:numOfComponents);

recognized_label = [];

for i = 1:size(projectionTest,1)
    euclide_dist = [];
    for j = 1:size(projectionTrain,1)
        dist = (norm(projectionTest(i,:) - projectionTrain(j,:))).^2;
        euclide_dist = [euclide_dist dist];
    end

    [euclide_dist_min, recognized_index] = min(euclide_dist);
    recognized_label = [recognized_label trainLabel(recognized_index)];
end

confusion_matrix = confusionmat(testLabel, recognized_label);
figure;confusionchart(confusion_matrix);


