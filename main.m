%% Driver File for Problem 1: Part 2: Face Detection
% You will implement an Adaboost Classifier to classify between face images
% and non-face images.
% Author Name : Vedant Jain vjain14
%PLEASE BE INSIDE THE PROBLEM_1 FOLDER WITH DATA PATH AND PART_2 INCLUDED
%INTO PATH


%% Your Driver Script Starts Here
% You can use as many auxilliary scripts as you want
% As long as we can run this script to get all the plots and classification
% accuracies we require from Part 2

dname = pwd;

imageMatrix = [];

%ifw_1000 folder is:
imageFolder = [dname '/data/lfw_1000'];
imageFolders = dir(imageFolder);

for i = 3:size(imageFolders,1)
    currentfile = imageFolders(i).name;
    
    %imagefiles = dir([dname '/' currentfolder]);
    imagefiledir = [imageFolder '/' currentfile];
        %currentfilename = imagefiles(j).name;
        I = double(imread(imagefiledir));
        imageMatrix = [imageMatrix normalize(reshape(I,4096,1))];

end

% test = reshape(imageMatrix(:,1),64,64);
% imshow(test, [])

%% first 10 eigen faces
meanImage = mean(imageMatrix, 2);
mImageMatrix = imageMatrix - meanImage;
correlationMatrix = corrcoef(mImageMatrix.');
[V S] = eig(correlationMatrix);
%V eigenvectors, %S is eigenvalues
lamdba = diag(S);

%look at eigenvectors need last one
%title('EigenValue')
%xlabel('Number eigenValue')

Efs = V(:,end-9:end);
%eigFace = reshape(Ef,64,64);

%% load in training data

%questions for ta: Do i resize the training data


%load in training faces:
dname = pwd;

faceMatrix = [];

%ifw_1000 folder is:
imageFolder = [dname '/data/boosting_data/train/face'];
imageFolders = dir(imageFolder);

for i = 3:size(imageFolders,1)
    currentfile = imageFolders(i).name;
    
    %imagefiles = dir([dname '/' currentfolder]);
    imagefiledir = [imageFolder '/' currentfile];
        %currentfilename = imagefiles(j).name;
        I = imresize(double(imread(imagefiledir)),[64,64]);
        faceMatrix = [faceMatrix normalize(reshape(I,[],1))];

end


% represent each face by the weights of its faces
faceReps = [];
for i = 1:size(faceMatrix,2)
    x = faceMatrix(:,i)-meanImage;
    Z = pinv(Efs)*x;
    faceReps = [faceReps Z];

end
% test = reshape(faceMatrix(:,1),64,64);
% imshow(test, [])


%% represent all nonfaces with by weights
%load in training nofaces:
dname = pwd;

noFaceMatrix = [];

%ifw_1000 folder is:
imageFolder = [dname '/data/boosting_data/train/non-face'];
imageFolders = dir(imageFolder);

noFaceMatrix = ones(4096, size(imageFolders,1)-2);

for i = 3:size(imageFolders,1)
    currentfile = imageFolders(i).name;
    
    %imagefiles = dir([dname '/' currentfolder]);
    imagefiledir = [imageFolder '/' currentfile];
        %currentfilename = imagefiles(j).name;
        I = imresize(double(imread(imagefiledir)),[64,64]);
        noFaceMatrix(:,i-2) = normalize(reshape(I,[],1));

end


% represent each face by the weights of its faces
noFaceReps = [];
for i = 1:size(faceMatrix,2)
    x = noFaceMatrix(:,i)-meanImage;
    Z = pinv(Efs)*x;
    noFaceReps = [noFaceReps Z];

end
% test = reshape(noFaceMatrix(:,1),64,64);
% imshow(test, []);




%% new adaboost classifier using all the eigenfaces

numT = 200;
% b = 1;
% trainFaceYes = [faceReps(b,:)' ones(length(faceReps(b,:)'),1)];
% trainNoFace = [noFaceReps(b,:)' -1*ones(length(noFaceReps(b,:)'),1)];

trainFaceData = faceReps;
trainNoFaceData = noFaceReps;

togetherAll = [faceReps noFaceReps];

%wether face or no face:
realDecision = [ones(size(faceReps,2),1)' -1*ones(size(noFaceReps,2),1)'];

%together = [trainFaceYes; trainNoFace];
%together(:,3) = 1/length(together(:,1));

weights = (1/length(realDecision))*ones(length(realDecision),1);
nW = [];
nW = [nW weights];
%signs = together(:,2);
%data = together(:,1);
hW = [];
aV = [];
dim = [];
model = struct;
for i = 1:numT
    
  currentDim = pickDimension(togetherAll,realDecision, nW(:,i));
  data = togetherAll(currentDim,:);
  dim(i) = currentDim;
  [hW(i) aV(i) wE]  = adaboostIter(data,realDecision,nW(:,i));
    nW = [nW cell2mat(wE)'];
    model(i).dims = currentDim;
    model(i).threshold = hW(i);
    model(i).alpha = aV(i);
    model(i).weight = wE;
    
end



%% test:

%load test data
dname = pwd;

testFaceMatrix = [];

%ifw_1000 folder is:
imageFolder = [dname '/data/boosting_data/test/face'];
imageFolders = dir(imageFolder);

for i = 3:size(imageFolders,1)
    currentfile = imageFolders(i).name;
    
    %imagefiles = dir([dname '/' currentfolder]);
    imagefiledir = [imageFolder '/' currentfile];
        %currentfilename = imagefiles(j).name;
        I = imresize(double(imread(imagefiledir)),[64,64]);
        testFaceMatrix = [testFaceMatrix normalize(reshape(I,[],1))];

end


% represent each face by the weights of its faces
testFaceReps = [];
for i = 1:size(testFaceMatrix,2)
    x = testFaceMatrix(:,i) - meanImage;
    Z = pinv(Efs)*x;
    testFaceReps = [testFaceReps Z];

end
% test = reshape(testFaceMatrix(:,1),64,64);
% imshow(test, [])

%% load test non-faces
%load test data
dname = pwd;

nonTestFaceMatrix = [];
%fileList = dir('*.bmp');
%ifw_1000 folder is:
imageFolder = [dname '/data/boosting_data/test/non-face'];
imageFolders = dir(imageFolder);
fileList = dir(fullfile(imageFolder, '*.pgm'));
nonTestFaceMatrix = ones(4096, size(imageFolders,1)-2);

for i = 1:size(fileList,1)
    currentfile = fileList(i).name;
    
    %imagefiles = dir([dname '/' currentfolder]);
    imagefiledir = [imageFolder '/' currentfile];
        %currentfilename = imagefiles(j).name;
        I = imresize(double(imread(imagefiledir)),[64,64]);
        nonTestFaceMatrix(:,i) =  normalize(reshape(I,[],1));

end


% represent each face by the weights of its faces
nonTestFaceReps = [];
for i = 1:size(nonTestFaceMatrix,2)
    x = nonTestFaceMatrix(:,i)- meanImage;
    Z = pinv(Efs)*x;
    nonTestFaceReps = [nonTestFaceReps Z];

end
% test = reshape(nonTestFaceMatrix(:,2),64,64);
% imshow(test, [])

%% NEW TEST SHOULD WORK

% testFaceYes = [testFaceReps(2,:)' ones(length(testFaceReps(2,:)'),1)];
% testNoFace = [nonTestFaceReps(2,:)' -1*ones(length(nonTestFaceReps(2,:)'),1)];

testTogether = [testFaceReps nonTestFaceReps];
realTestDecision = [ones(length(testFaceReps(2,:)'),1)' -1*ones(length(nonTestFaceReps(2,:)'),1)'];

decision_ht = zeros(length(realTestDecision),1)';

for j = 1:numT
    d_m = model(j).dims;
    a_o = model(j).alpha;
    thres = model(j).threshold;
    dPic = testTogether(d_m,:);
    greater = dPic > thres;
    dMatrix(greater) = 1;
    dMatrix(~greater) = -1;
    dMatrix = double(a_o*dMatrix);
    decision_ht = decision_ht + dMatrix;
    
end

decisions = sign(decision_ht);
classLabels = categorical;
classLabels(1) = {'Face'};
classLabels(2) = {'No Face'};

C = confusionmat(realTestDecision,decisions,'Order',[1 -1]);
cm = confusionchart(C,classLabels');
cm.Title = 'Hit Rate';
cm.RowSummary = 'row-normalized';

%%
%cm.ColumnSummary = 'column-normalized';


% %% put test face and nonface in one array
% 
% testFaceYes = [testFaceReps(2,:)' ones(length(testFaceReps(2,:)'),1)];
% testNoFace = [nonTestFaceReps(2,:)' -1*ones(length(nonTestFaceReps(2,:)'),1)];
% 
% testTogether = [testFaceYes; testNoFace];
% 
% classif = [];
% classifiedSum = [];
% 
%     for j= 1:length(thresholdsForH)
%         firstVec = testTogether(:,1);
%         greaters = firstVec > thresholdsForH(j);
%         newClassif(greaters) = 1* errorsForWeights(j);
%         newClassif(~greaters) = -1* errorsForWeights(j);
%         classifiedSum = [classifiedSum newClassif'];
%     end
%     
%     
%     %% put test face and nonface in one array for method way
% 
% testFaceYes = [testFaceReps(2,:)' ones(length(testFaceReps(2,:)'),1)];
% testNoFace = [nonTestFaceReps(2,:)' -1*ones(length(nonTestFaceReps(2,:)'),1)];
% 
% testTogether = [testFaceYes; testNoFace];
% 
% classif = [];
% classifiedSum = [];
% 
%     for j= 1:length(aV)
%         firstVec = testTogether(:,1);
%         greaters = firstVec > hW(j);
%         newClassif(greaters) = 1* aV(j);
%         newClassif(~greaters) = -1* aV(j);
%         classifiedSum = [classifiedSum newClassif'];
%     end
%     %%
%     realValue = testTogether(:,2);
%     %Dimension = [Dimension sum(classifiedSum,2)];
%     decisions = sum(classifiedSum,2) >0;
%     tH(decisions) = 1;
%     tH(~decisions) = -1;
%     
%     compareEm = ((tH') == realValue);
%     Hits = sum(compareEm)/length(realValue)
%     
%     %% just heads
%     headsCorrect = sum(compareEm(1:472))/length(compareEm(1:472))
%     noFaceCorrect = sum(compareEm(472:end))/length(compareEm(472:end))
% %% Dimension
% %Dimension = [];
% 
% 
%     
%        
       
        
        
        
        
    




