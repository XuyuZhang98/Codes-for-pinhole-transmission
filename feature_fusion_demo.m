%% ==============================================================================
%  Feature Fusion Algorithm
%
%  Author: Xuyu Zhang
% ==============================================================================

clear; close all; clc;
warning('off', 'all');

%% -----------------------------------------------------------------------------
%  Step 1: Select the Raw Speckle Pattern
% -----------------------------------------------------------------------------
uiwait(msgbox('Step 1: Please select the raw speckle pattern image.', 'Operation Tip', 'help'));

[fName, pName] = uigetfile({'*.bmp;*.jpg;*.png;*.tif', 'Image Files'}, 'Select Speckle Image');

if isequal(fName, 0)
    fprintf('Operation canceled by user.\n');
    return;
end

specklePath = fullfile(pName, fName);

try
    SpeckleImg = imread(specklePath);
    % Convert to grayscale if the image is RGB
    if size(SpeckleImg, 3) == 3
        SpeckleImg = rgb2gray(SpeckleImg);
    end
catch
    errordlg('Failed to read the image file.', 'Error');
    return;
end

%% -----------------------------------------------------------------------------
%  Step 2: Obtain the Reference Template
% -----------------------------------------------------------------------------
choice = questdlg('Select the method to obtain the template:', ...
    'Template Selection Strategy', ...
    'A. Import from File', 'B. Manual Selection from Speckle', 'A. Import from File');

if isempty(choice)
    return; % User closed the dialog
end

rawTemplate = []; % Variable to store the raw template

if strcmp(choice, 'A. Import from File')
    % --- Strategy A: Import an existing image (e.g., autocorrelation result) ---
    [tName, tPath] = uigetfile({'*.bmp;*.jpg;*.png', 'Image Files'}, 'Select Template Image');
    if isequal(tName, 0), return; end
    
    imgTemp = imread(fullfile(tPath, tName));
    if size(imgTemp, 3) == 3
        imgTemp = rgb2gray(imgTemp);
    end
    rawTemplate = imgTemp;
    
elseif strcmp(choice, 'B. Manual Selection from Speckle')
    % --- Strategy B: Manual selection (ROI) ---
    hFig = figure('Name', 'Select ROI (Press Enter to Confirm)', 'NumberTitle', 'off');
    imshow(SpeckleImg, 'InitialMagnification', 'fit');
    title('Instruction: Draw a rectangle -> Adjust size -> Press "Enter" key to confirm');
    
    % Activate the rectangle drawing tool
    hRect = drawrectangle('Label', 'Template', 'Color', 'r');
    
    % Wait for the user to press 'Enter'
    while true
        try
            w = waitforbuttonpress;
            key = get(hFig, 'CurrentKey');
            if w == 1 && strcmp(key, 'return')
                break;
            end
        catch
            if ~isvalid(hFig), return; end
        end
    end
    
    % Crop the selected region
    if isvalid(hRect)
        pos = round(hRect.Position);
        rawTemplate = imcrop(SpeckleImg, pos);
    end
    
    if isvalid(hFig), close(hFig); end
end

if isempty(rawTemplate)
    errordlg('No valid template obtained.', 'Error');
    return;
end

%% -----------------------------------------------------------------------------
%  Step 3: Template Preprocessing (Smart Cropping & Background Removal)
% -----------------------------------------------------------------------------
fprintf('Preprocessing the template...\n');

% 1. Binary Segmentation using Otsu's method
level = graythresh(rawTemplate);
% Relax the threshold (0.8x) to preserve weak edges
bw = imbinarize(rawTemplate, level * 0.8);

% 2. Morphological Operations: Fill holes and close gaps
bw = imfill(bw, 'holes');
se = strel('disk', 3);
bw = imclose(bw, se);

% 3. Find the largest connected component (Assuming it is the object)
stats = regionprops(bw, 'BoundingBox', 'Area');

if isempty(stats)
    % Fallback: Use the original crop if segmentation fails
    TemplateImg = rawTemplate;
else
    [~, idx] = max([stats.Area]);
    bbox = stats(idx).BoundingBox; % [x, y, w, h]
    
    % 4. Calculate bounding box with padding
    % Avoid tight cropping; expand by 15% to keep natural boundaries
    paddingRatio = 0.15;
    padX = bbox(3) * paddingRatio;
    padY = bbox(4) * paddingRatio;
    
    [rawH, rawW] = size(rawTemplate);
    
    x1 = max(1, floor(bbox(1) - padX));
    y1 = max(1, floor(bbox(2) - padY));
    x2 = min(rawW, ceil(bbox(1) + bbox(3) + padX));
    y2 = min(rawH, ceil(bbox(2) + bbox(4) + padY));
    
    cropRect = [x1, y1, x2-x1, y2-y1];
    TemplateImg = imcrop(rawTemplate, cropRect);
end

% Display preprocessing result
figure('Name', 'Template Preprocessing Check', 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none');
subplot(1,2,1); imshow(rawTemplate); title('Original Selection');
subplot(1,2,2); imshow(TemplateImg); title('Processed Template');
drawnow;

%% -----------------------------------------------------------------------------
%  Step 4: Parameter Configuration
% -----------------------------------------------------------------------------
prompt = {'SSIM Threshold (0-1, Recommended 0.5):', ...
          'Min Feature Matches (Recommended 5):', ...
          'Scanning Step Size (Pixels, Recommended 2-5):'};
dlgtitle = 'Algorithm Parameters';
dims = [1 60];
definput = {'0.5', '5', '4'};
answer = inputdlg(prompt, dlgtitle, dims, definput);

if isempty(answer), return; end

ssimThresh = str2double(answer{1});
featThresh = str2double(answer{2});
stepSize = str2double(answer{3});

%% -----------------------------------------------------------------------------
%  Step 5: Global Scanning & Matching
% -----------------------------------------------------------------------------
fprintf('Starting global scanning...\n');

[H, W] = size(SpeckleImg);
[th, tw] = size(TemplateImg);

% Extract SURF features from the template
pointsTemplate = detectSURFFeatures(TemplateImg);
[featuresTemplate, pointsTemplate] = extractFeatures(TemplateImg, pointsTemplate);

candidates = {};      % Store matched image patches
candidateLocs = {};   % Store feature point locations for registration

hWait = waitbar(0, 'Scanning and matching...');

count = 0;
totalSteps = floor((H-th)/stepSize) * floor((W-tw)/stepSize);

% Nested loop for pixel-wise scanning
for r = 1:stepSize:(H - th + 1)
    for c = 1:stepSize:(W - tw + 1)
        count = count + 1;
        if mod(count, 200) == 0
            waitbar(count/totalSteps, hWait, sprintf('Scanning... Found %d candidates', length(candidates)));
        end
        
        % Crop local region
        localRegion = SpeckleImg(r:r+th-1, c:c+tw-1);
        
        % --- Criterion 1: SSIM Verification ---
        currentSSIM = ssim(localRegion, TemplateImg);
        
        if currentSSIM >= ssimThresh
            % --- Criterion 2: Feature Point Matching ---
            % Compute features only if SSIM passes (Optimization)
            pointsLocal = detectSURFFeatures(localRegion);
            [featuresLocal, pointsLocal] = extractFeatures(localRegion, pointsLocal);
            
            if isempty(pointsLocal)
                continue;
            end
            
            % Match features
            indexPairs = matchFeatures(featuresTemplate, featuresLocal, 'Unique', true);
            matchedCount = size(indexPairs, 1);
            
            if matchedCount >= featThresh
                % Store candidate if both criteria are met
                candidates{end+1} = localRegion;
                
                % Store matched points for geometric registration
                locStruct.matchedPointsTemp = pointsTemplate(indexPairs(:, 1));
                locStruct.matchedPointsLocal = pointsLocal(indexPairs(:, 2));
                candidateLocs{end+1} = locStruct;
            end
        end
    end
end
close(hWait);

if isempty(candidates)
    errordlg('No matching regions found. Try lowering the SSIM threshold.', 'No Results');
    return;
end

%% -----------------------------------------------------------------------------
%  Step 6: Feature Fusion (Geometric Registration & Averaging)
% -----------------------------------------------------------------------------
fprintf('Found %d candidates. Starting fusion...\n', length(candidates));
hWait = waitbar(0, 'Performing geometric registration and fusion...');

sumImg = double(TemplateImg);   % Initialize accumulator
countImg = ones(size(TemplateImg)); % Initialize counter

for k = 1:length(candidates)
    waitbar(k/length(candidates), hWait);
    
    candImg = candidates{k};
    ptsT = candidateLocs{k}.matchedPointsTemp;
    ptsL = candidateLocs{k}.matchedPointsLocal;
    
    try
        % Estimate geometric transform (Similarity: Translation + Rotation + Scale)
        % Minimum 3 points required for similarity transform
        if ptsT.Count >= 3
            tform = estimateGeometricTransform(ptsL, ptsT, 'similarity');
            
            % Warp the candidate image to align with the template
            warpedImg = imwarp(candImg, tform, 'OutputView', imref2d(size(TemplateImg)));
            
            % Accumulate
            sumImg = sumImg + double(warpedImg);
            countImg = countImg + 1;
        else
            % Fallback: Direct accumulation if points are insufficient
            sumImg = sumImg + double(candImg);
            countImg = countImg + 1;
        end
    catch
        % Fallback for registration errors
        sumImg = sumImg + double(candImg);
        countImg = countImg + 1;
    end
end
close(hWait);

% Compute the average image
FinalResult = uint8(sumImg ./ countImg);

% Enhance contrast
EnhancedResult = imadjust(FinalResult);

%% -----------------------------------------------------------------------------
%  Step 7: Result Visualization & Saving
% -----------------------------------------------------------------------------
figure('Name', 'Reconstruction Results', 'NumberTitle', 'off');

subplot(2, 2, 1); 
imshow(SpeckleImg); 
title('Raw Speckle Pattern');

subplot(2, 2, 2); 
imshow(TemplateImg); 
title('Reference Template');

subplot(2, 2, 3); 
imshow(FinalResult); 
title(sprintf('Fused Result (N=%d)', length(candidates)));

subplot(2, 2, 4); 
imshow(EnhancedResult); 
title('Contrast Enhanced');

% Prompt to save
saveChoice = questdlg('Do you want to save the results?', 'Save Result', 'Yes', 'No', 'Yes');
if strcmp(saveChoice, 'Yes')
    [sName, sPath] = uiputfile('Reconstructed_Result.png', 'Save Image As');
    if ~isequal(sName, 0)
        fullSavePath = fullfile(sPath, sName);
        imwrite(EnhancedResult, fullSavePath);
        msgbox(['Image saved to: ', fullSavePath], 'Success');
    end
end

fprintf('Process completed successfully.\n');