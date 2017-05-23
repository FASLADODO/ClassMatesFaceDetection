function [P] = RecogniseFace( I, featureType, model )
format shortg %sets to no scientific notation as the outputs are prone to this
A = imread(I); %Read image file
%use viola-jones face detector, and change the merge threshold to 10
FaceDetector = vision.CascadeObjectDetector('MergeThreshold', 10);
% create bounding boxes
bbox = step(FaceDetector,A);

n = size(bbox, 1);
if n == 0
    P = []; %argument to output nothing if zero
else
    for i=1:n
% Extract the ith face- taken from lectures 
        a = bbox(i, 1);
        b = bbox(i, 2);
        c = a+bbox(i, 3);
        d = b+bbox(i, 4);
        Face = A(b:d, a:c, :);
        
        %call the models and then select whether HOG or SURF is called
        if isequal(model, 'SVM') 
            if isequal(featureType, 'HOG');
                resize_image = imresize(Face, [45, 45]); %resize like in training to [45 45]
                grayImage = rgb2gray(resize_image); %remember HOG requires gray images
                HOG = extractHOGFeatures(grayImage);
                SVM_model = loadCompactModel('HOG_SVM');
                predictions(i) = predict(SVM_model,HOG);
            elseif isequal(featureType, 'SURF');
                grayImage = rgb2gray(Face);
                bagoffeatures = load('trainingbag.mat');
                bagoffeatures = bagoffeatures.trainingbag;
                SURF = encode(bagoffeatures, grayImage);
                SVM_model = loadCompactModel('SURF_SVM');
                predictions(i) = predict(SVM_model,SURF);
            end
        end
         if isequal(model, 'DT')
            if isequal(featureType, 'HOG');
                resize_image = imresize(Face, [45, 45]);
                grayImage = rgb2gray(resize_image);
                HOG = extractHOGFeatures(grayImage);
                DT_model = load('HOG_DT.mat');
                DT_model = DT_model.HOG_compact_DT;
                predictions(i) = predict(DT_model,HOG);
            elseif isequal(featureType, 'SURF');
                grayImage = rgb2gray(Face);
                bagoffeatures = load('trainingbag.mat');
                bagoffeatures = bagoffeatures.trainingbag;
                SURF = encode(bagoffeatures, grayImage);
                DT_model = load('SURF_DT.mat');
                DT_model = DT_model.SURF_compact_DT;
                
                predictions(i) = predict(DT_model,SURF);
            end
         end
         
         if isequal(model, 'RF')
            if isequal(featureType, 'HOG');
                resize_image = imresize(Face, [45, 45]);
                grayImage = rgb2gray(resize_image);
                HOG = extractHOGFeatures(grayImage);
                RF_model = load('HOG_RF.mat');
                RF_model = RF_model.HOG_RF_COMPACT;
                predictions(i) = predict(RF_model,HOG);
            elseif isequal(featureType, 'SURF');
                grayImage = rgb2gray(Face);
                bagoffeatures = load('trainingbag.mat');
                bagoffeatures = bagoffeatures.trainingbag;
                SURF = encode(bagoffeatures, grayImage);
                RF_model = load('SURF_RF.mat');
                RF_model = RF_model.SURF_RF_COMPACT;
                predictions(i) = predict(RF_model,SURF);
            end
         end
         
        x(i) = a + bbox(i, 3)*0.5;  %calculate x
        y(i) = b + bbox(i, 4)*0.5;  %calculate y
    end
end

predictions = str2double(predictions); %convert string to double
P = horzcat(predictions',x',y'); %horizontally concatenate predictions and x and y coordinates

% visualise the output bounding boxes on the image, zooming in will show
% which label is projected to each bounding box
visual_faces = insertObjectAnnotation(A,'rectangle',bbox, predictions);
imshow(visual_faces);

end