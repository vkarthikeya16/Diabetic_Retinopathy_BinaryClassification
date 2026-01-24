%% ========================================================================
%% COMPLETE DIABETIC RETINOPATHY CLASSIFICATION PIPELINE - OPTIMIZED
%% All phases integrated + Performance Improvements + ALL FEATURES
%% Version: 3.5 (No Feature Selection - Maximum Performance)
%% ========================================================================
%%
%% CRITICAL CHANGE IN v3.5:
%%   🔥 USES ALL 20 FEATURES (no feature selection)
%%   → Feature selection was limiting performance to 77%
%%   → Using all features preserves feature interactions
%%   → Expected to reach 80-86% accuracy
%%
%% OTHER IMPROVEMENTS:
%%   ✅ SVM hyperparameter tuning (kernel + C parameter grid search)
%%   ✅ k-NN hyperparameter tuning (optimal k selection)
%%   ✅ Decision Tree with pruning (prevents overfitting)
%%   ✅ Best model selection with F1-score tiebreaker
%%   ✅ APTOS external validation (800 samples)
%%
%% EXPECTED PERFORMANCE:
%%   • Internal Test Accuracy: 80-86%
%%   • External APTOS Accuracy: 50-55%
%%   • Best Model: SVM or Logistic Regression
%%
%% ========================================================================

clear all; close all; clc;

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║ DIABETIC RETINOPATHY CLASSIFICATION PIPELINE             ║\n');
fprintf('║ MAXIMUM PERFORMANCE - Using ALL Features (No Selection)  ║\n');
fprintf('║ Version: 3.5 (Targeting 80-86%% Accuracy)                 ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

rng(42);  % Set random seed for reproducibility
totalTime = tic;

%% ========================================================================
%% CONFIGURATION
%% ========================================================================

fprintf('📁 Project root: %s\n\n', pwd);

%% PATHS
cfg = struct();
cfg.paths = struct();
cfg.paths.root = pwd;
cfg.paths.data = fullfile(cfg.paths.root, 'data');
cfg.paths.results = fullfile(cfg.paths.root, 'results');
cfg.paths.forR = fullfile(cfg.paths.root, 'R_exports');
cfg.paths.figures = fullfile(cfg.paths.root, 'figures');

%% CREATE DIRECTORIES
dirs_to_create = {cfg.paths.data, cfg.paths.results, cfg.paths.forR, cfg.paths.figures};
for i = 1:length(dirs_to_create)
    if ~exist(dirs_to_create{i}, 'dir')
        mkdir(dirs_to_create{i});
    end
end

%% CLOUD CONFIGURATION
cfg.cloud = struct();
cfg.cloud.enabled = true;
cfg.cloud.platform = 'github';
cfg.cloud.username = 'vkarthikeya16';
cfg.cloud.repo = 'data';
cfg.cloud.branch = 'main';
cfg.cloud.base_url = 'https://raw.githubusercontent.com/vkarthikeya16/data/main/';
cfg.cloud.images_base = 'images/';
cfg.cloud.healthy_folder = 'Healthy';
cfg.cloud.dr_folder = 'DR';
cfg.cloud.manifest_file = 'file_manifest.txt';
cfg.cloud.healthy_url = [cfg.cloud.base_url, cfg.cloud.images_base, cfg.cloud.healthy_folder, '/'];
cfg.cloud.dr_url = [cfg.cloud.base_url, cfg.cloud.images_base, cfg.cloud.dr_folder, '/'];
cfg.cloud.manifest_url = [cfg.cloud.base_url, cfg.cloud.manifest_file];
cfg.cloud.cache_dir = fullfile(cfg.paths.root, 'data_cache');
cfg.cloud.force_fresh = false;
cfg.cloud.timeout = 60;
cfg.cloud.max_retries = 3;
cfg.cloud.retry_delay = 5;
cfg.cloud.show_progress = true;

%% SMOTE CONFIGURATION
cfg.smote = struct();
cfg.smote.numNeighbors = 5;
cfg.smote.enabled = true;

%% MODEL TRAINING CONFIGURATION
cfg.models = struct();
cfg.models.dt.maxDepth = 20;
cfg.models.dt.minLeafSize = 5;
cfg.models.svm.kernelFunction = 'rbf';
cfg.models.svm.C = 1;
cfg.models.knn.kValues = [3 5 7 9 11 15];
cfg.models.nb.distributionNames = 'normal';

%% FIGURE CONFIGURATION
cfg.figures = struct();
cfg.figures.dpi = 300;
cfg.figures.format = 'png';
cfg.figures.width = 10;
cfg.figures.height = 8;

fprintf('✓ Configuration loaded\n\n');

%% ========================================================================
%% PHASE 1: DATA LOADING
%% ========================================================================

fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║ PHASE 1: DATA LOADING                                     ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

phase1Time = tic;

if cfg.cloud.enabled
    fprintf('☁️  Cloud loading ENABLED\n');
    fprintf('   Repository: %s/%s\n\n', cfg.cloud.username, cfg.cloud.repo);
    
    %% Create cache directory
    if ~exist(cfg.cloud.cache_dir, 'dir')
        mkdir(cfg.cloud.cache_dir);
    end
    
    %% Check cache first
    cached_files = [dir(fullfile(cfg.cloud.cache_dir, '*.png')); ...
                    dir(fullfile(cfg.cloud.cache_dir, '*.jpg'))];
    
    if ~isempty(cached_files) && ~cfg.cloud.force_fresh
        fprintf('💾 Cache found: %d images\n', length(cached_files));
        fprintf('   Using cached data (fast!)\n\n');
        
        images = cell(length(cached_files), 1);
        labels = cell(length(cached_files), 1);
        
        for i = 1:length(cached_files)
            img_path = fullfile(cfg.cloud.cache_dir, cached_files(i).name);
            images{i} = imread(img_path);
            
            if startsWith(cached_files(i).name, 'Healthy_')
                labels{i} = 'Healthy';
            else
                labels{i} = 'DR';
            end
        end
        
    else
        %% Download from GitHub
        fprintf('[1/3] Reading manifest...\n');
        
        manifest_data = webread(cfg.cloud.manifest_url);
        manifest_lines = strsplit(manifest_data, '\n');
        
        healthy_count = 0;
        dr_count = 0;
        
        for i = 1:length(manifest_lines)
            line = strtrim(manifest_lines{i});
            if isempty(line) || startsWith(line, '#'), continue; end
            
            parts = strsplit(line, ',');
            if length(parts) >= 2
                class = strtrim(parts{2});
                if strcmpi(class, 'Healthy')
                    healthy_count = healthy_count + 1;
                elseif strcmpi(class, 'DR')
                    dr_count = dr_count + 1;
                end
            end
        end
        
        fprintf('   Found: %d Healthy, %d DR\n\n', healthy_count, dr_count);
        
        %% Download Healthy images
        fprintf('[2/3] Downloading Healthy images...\n');
        
        healthy_images = {};
        healthy_count_downloaded = 0;
        healthy_failed = 0;
        
        for i = 1:length(manifest_lines)
            line = strtrim(manifest_lines{i});
            if isempty(line) || startsWith(line, '#'), continue; end
            
            parts = strsplit(line, ',');
            if length(parts) < 2, continue; end
            
            filename = strtrim(parts{1});
            class = strtrim(parts{2});
            
            if strcmpi(class, 'Healthy')
                img_url = [cfg.cloud.healthy_url, filename];
                cache_path = fullfile(cfg.cloud.cache_dir, ['Healthy_', filename]);
                
                if exist(cache_path, 'file')
                    success = true;
                else
                    success = download_robust(img_url, cache_path, 3, 30);
                end
                
                if success
                    try
                        img = imread(cache_path);
                        healthy_images{end+1} = img;
                        healthy_count_downloaded = healthy_count_downloaded + 1;
                        
                        if mod(healthy_count_downloaded, 20) == 0
                            fprintf('   Progress: %d/%d\n', healthy_count_downloaded, healthy_count);
                        end
                    catch
                        healthy_failed = healthy_failed + 1;
                    end
                else
                    healthy_failed = healthy_failed + 1;
                end
            end
        end
        
        fprintf('   ✓ Downloaded %d/%d', healthy_count_downloaded, healthy_count);
        if healthy_failed > 0
            fprintf(' (%d failed)', healthy_failed);
        end
        fprintf('\n\n');
        
        %% Download DR images
        fprintf('[3/3] Downloading DR images...\n');
        
        dr_images = {};
        dr_count_downloaded = 0;
        dr_failed = 0;
        
        for i = 1:length(manifest_lines)
            line = strtrim(manifest_lines{i});
            if isempty(line) || startsWith(line, '#'), continue; end
            
            parts = strsplit(line, ',');
            if length(parts) < 2, continue; end
            
            filename = strtrim(parts{1});
            class = strtrim(parts{2});
            
            if strcmpi(class, 'DR')
                img_url = [cfg.cloud.dr_url, filename];
                cache_path = fullfile(cfg.cloud.cache_dir, ['DR_', filename]);
                
                if exist(cache_path, 'file')
                    success = true;
                else
                    success = download_robust(img_url, cache_path, 3, 30);
                end
                
                if success
                    try
                        img = imread(cache_path);
                        dr_images{end+1} = img;
                        dr_count_downloaded = dr_count_downloaded + 1;
                        
                        if mod(dr_count_downloaded, 50) == 0
                            fprintf('   Progress: %d/%d\n', dr_count_downloaded, dr_count);
                        end
                    catch
                        dr_failed = dr_failed + 1;
                    end
                else
                    dr_failed = dr_failed + 1;
                end
            end
        end
        
        fprintf('   ✓ Downloaded %d/%d', dr_count_downloaded, dr_count);
        if dr_failed > 0
            fprintf(' (%d failed)', dr_failed);
        end
        fprintf('\n\n');
        
        images = [healthy_images(:); dr_images(:)];
        labels = [repmat({'Healthy'}, length(healthy_images), 1); repmat({'DR'}, length(dr_images), 1)];
    end
end

%% SUMMARY
fprintf('📊 Dataset loaded:\n');
fprintf('   • Total:   %d images\n', length(images));
fprintf('   • Healthy: %d (%.1f%%)\n', sum(strcmp(labels, 'Healthy')), 100*sum(strcmp(labels, 'Healthy'))/length(labels));
fprintf('   • DR:      %d (%.1f%%)\n\n', sum(strcmp(labels, 'DR')), 100*sum(strcmp(labels, 'DR'))/length(labels));

fprintf('✓ Phase 1 complete (%.1fs)\n\n', toc(phase1Time));

%% ========================================================================
%% PHASE 2: FEATURE EXTRACTION
%% ========================================================================

fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║ PHASE 2: FEATURE EXTRACTION                               ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

phase2Time = tic;

numImages = length(images);
num_features = 20;
features = zeros(numImages, num_features);

fprintf('   Extracting 20 clinical features from %d images...\n\n', numImages);

for i = 1:numImages
    try
        img = images{i};
        features(i, :) = extract_real_features(img);
    catch ME
        warning('Feature extraction failed for image %d: %s', i, ME.message);
        features(i, :) = nan(1, num_features);
    end
    
    if mod(i, 100) == 0
        fprintf('   Progress: %d/%d (%.1f%%)\n', i, numImages, 100*i/numImages);
    end
end

fprintf('   Progress: %d/%d (100.0%%)\n\n', numImages, numImages);

%% VALIDATION
nan_count = sum(any(isnan(features), 2));
inf_count = sum(any(isinf(features), 2));

if nan_count > 0 || inf_count > 0
    warning('Found %d samples with NaN, %d with Inf - removing them', nan_count, inf_count);
    valid_idx = ~any(isnan(features) | isinf(features), 2);
    features = features(valid_idx, :);
    labels = labels(valid_idx);
    images = images(valid_idx);
    fprintf('   ⚠️  Removed %d invalid samples\n', sum(~valid_idx));
end

%% FEATURE NAMES
featureNames = {
    'Mean_Intensity', 'Std_Intensity', 'Skewness', 'Kurtosis', ...
    'Red_Channel_Mean', 'Green_Channel_Mean', 'Blue_Channel_Mean', ...
    'GLCM_Contrast', 'GLCM_Correlation', 'GLCM_Energy', 'GLCM_Homogeneity', ...
    'Entropy', 'RG_Ratio', 'Optic_Disc_Intensity', ...
    'Lesion_Area', 'Lesion_Perimeter', 'Lesion_Eccentricity', ...
    'Blood_Vessel_Density', 'Vessel_Count', 'Mean_Vessel_Length'
};

labels = categorical(labels);
fprintf('✓ Phase 2 complete (%.1fs)\n\n', toc(phase2Time));

%% ========================================================================
%% PHASE 3: DATA PREPARATION & FEATURE SELECTION
%% ========================================================================

fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║ PHASE 3: DATA PREPARATION & FEATURE SELECTION            ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

phase3Time = tic;

%% Remove zero-variance features
fprintf('   [1/6] Detecting and removing zero-variance features...\n');

featureVariances = var(features);
varianceThreshold = 1e-10;
validFeaturesIdx = find(featureVariances > varianceThreshold);
invalidFeaturesIdx = find(featureVariances <= varianceThreshold);

if ~isempty(invalidFeaturesIdx)
    fprintf('      ⚠️  Found %d features with zero variance - removing\n', length(invalidFeaturesIdx));
    features = features(:, validFeaturesIdx);
    featureNames = featureNames(validFeaturesIdx);
end

fprintf('      ✓ Working with %d valid features\n\n', length(validFeaturesIdx));

Y = double(labels == 'DR');

%% Split data (60/20/20)
fprintf('   [2/6] Splitting data (60/20/20)...\n');

cv1 = cvpartition(Y, 'HoldOut', 0.20, 'Stratify', true);
X_trainval = features(cv1.training, :);
Y_trainval = Y(cv1.training);
X_test = features(cv1.test, :);
Y_test = Y(cv1.test);

cv2 = cvpartition(Y_trainval, 'HoldOut', 0.25, 'Stratify', true);
X_train = X_trainval(cv2.training, :);
Y_train = Y_trainval(cv2.training);
X_val = X_trainval(cv2.test, :);
Y_val = Y_trainval(cv2.test);

fprintf('      Train: %d, Val: %d, Test: %d\n\n', size(X_train,1), size(X_val,1), size(X_test,1));

Y_train_orig = Y_train;

%% Normalize (z-score)
fprintf('   [3/6] Normalizing features...\n');

mu = mean(X_train);
sigma = std(X_train);
sigma(sigma == 0) = 1;

X_train_norm = (X_train - mu) ./ sigma;
X_val_norm = (X_val - mu) ./ sigma;
X_test_norm = (X_test - mu) ./ sigma;

fprintf('      ✓ Applied z-score normalization\n\n');

%% ⭐ SAVE DATA BEFORE SMOTE (for Phase 5 visualization)
X_train_before_smote = X_train_norm;
Y_train_before_smote = Y_train;

%% SMOTE
fprintf('   [4/6] Applying SMOTE...\n');

numHealthy = sum(Y_train == 0);
numDR = sum(Y_train == 1);

fprintf('      Pre-SMOTE:  Healthy=%d, DR=%d (ratio 1:%.2f)\n', numHealthy, numDR, numDR/numHealthy);

if numHealthy < numDR
    minorityClass = 0;
    numMinority = numHealthy;
    numMajority = numDR;
else
    minorityClass = 1;
    numMinority = numDR;
    numMajority = numHealthy;
end

numToGenerate = numMajority - numMinority;
k = min(cfg.smote.numNeighbors, numMinority - 1);

if k >= 1
    minorityIdx = (Y_train == minorityClass);
    X_minority = X_train_norm(minorityIdx, :);
    X_synthetic = zeros(numToGenerate, size(X_train_norm, 2));
    
    for i = 1:numToGenerate
        idx = randi(numMinority);
        sample = X_minority(idx, :);
        
        distances = sum((X_minority - sample).^2, 2);
        [~, sortIdx] = sort(distances);
        neighborIdx = sortIdx(2:k+1);
        
        neighbor = X_minority(neighborIdx(randi(k)), :);
        lambda = rand();
        X_synthetic(i, :) = sample + lambda * (neighbor - sample);
    end
    
    X_train_norm = [X_train_norm; X_synthetic];
    Y_train = [Y_train; repmat(minorityClass, numToGenerate, 1)];
    
    fprintf('      Post-SMOTE: Healthy=%d, DR=%d (ratio 1:%.2f)\n\n', ...
        sum(Y_train==0), sum(Y_train==1), sum(Y_train==1)/sum(Y_train==0));
end

%% Feature selection - USE ALL FEATURES (MAXIMUM PERFORMANCE)
fprintf('   [5/6] Feature selection: USING ALL FEATURES...\n');

% Calculate F-statistics for reporting purposes only
fStats = zeros(size(X_train_norm, 2), 1);
pValues = zeros(size(X_train_norm, 2), 1);

for i = 1:size(X_train_norm, 2)
    class0 = X_train_norm(Y_train == 0, i);
    class1 = X_train_norm(Y_train == 1, i);
    
    mean_overall = mean(X_train_norm(:, i));
    mean0 = mean(class0);
    mean1 = mean(class1);
    n0 = length(class0);
    n1 = length(class1);
    
    ss_between = n0*(mean0-mean_overall)^2 + n1*(mean1-mean_overall)^2;
    ss_within = sum((class0-mean0).^2) + sum((class1-mean1).^2);
    
    df_between = 1;
    df_within = n0 + n1 - 2;
    ms_between = ss_between / df_between;
    ms_within = ss_within / df_within;
    fStats(i) = ms_between / (ms_within + eps);
    
    [~, p] = ttest2(class0, class1);
    pValues(i) = p;
end

[~, sortIdx] = sort(fStats, 'descend');

fprintf('      Top 5 most discriminative features:\n');
for i = 1:min(5, length(sortIdx))
    idx = sortIdx(i);
    fprintf('         %d. %s (F=%.2f, p=%.2e)\n', i, featureNames{idx}, fStats(idx), pValues(idx));
end
fprintf('\n');

% CRITICAL CHANGE: Use ALL features - no selection!
fprintf('      Strategy: Using ALL %d features (no selection)\n', size(X_train_norm, 2));
fprintf('      Rationale: Preserve feature interactions for maximum performance\n\n');

% Keep all features
selectedFeatureIdx = 1:size(X_train_norm, 2);
optimalNumFeatures = length(selectedFeatureIdx);

% No changes to data - already using all features
% featureNames already contains all names

fprintf('   [6/6] Final validation...\n');
fprintf('      ✓ Train: %d samples, %d features (ALL)\n', size(X_train_norm,1), size(X_train_norm,2));
fprintf('      ✓ Val:   %d samples, %d features (ALL)\n', size(X_val_norm,1), size(X_val_norm,2));
fprintf('      ✓ Test:  %d samples, %d features (ALL)\n\n', size(X_test_norm,1), size(X_test_norm,2));

fprintf('✓ Phase 3 complete (%.1fs)\n\n', toc(phase3Time));

%% ========================================================================
%% PHASE 4: MODEL TRAINING & EVALUATION
%% ========================================================================

fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║ PHASE 4: MODEL TRAINING & EVALUATION                     ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

phase4Time = tic;

results = struct();

%% Logistic Regression
fprintf('   [1/5] Training Logistic Regression...\n');
tic;

[b, ~] = glmfit(X_train_norm, Y_train, 'binomial', 'link', 'logit');
probsVal = glmval(b, X_val_norm, 'logit');
probsTest = glmval(b, X_test_norm, 'logit');
Y_pred_val = double(probsVal > 0.5);
Y_pred_test = double(probsTest > 0.5);

valAcc = sum(Y_pred_val == Y_val) / length(Y_val);
valPrec = sum(Y_pred_val & Y_val) / (sum(Y_pred_val) + eps);
valRec = sum(Y_pred_val & Y_val) / (sum(Y_val) + eps);
valF1 = 2 * (valPrec * valRec) / (valPrec + valRec + eps);

results.LR.accuracy = sum(Y_pred_test == Y_test) / length(Y_test);
results.LR.precision = sum(Y_pred_test & Y_test) / (sum(Y_pred_test) + eps);
results.LR.recall = sum(Y_pred_test & Y_test) / (sum(Y_test) + eps);
results.LR.f1 = 2 * (results.LR.precision * results.LR.recall) / (results.LR.precision + results.LR.recall + eps);
results.LR.trainTime = toc;
results.LR.model = b;
results.LR.val_accuracy = valAcc;
results.LR.val_precision = valPrec;
results.LR.val_recall = valRec;
results.LR.val_f1 = valF1;

fprintf('         Val: Acc=%.4f, F1=%.4f | Test: Acc=%.4f\n\n', valAcc, valF1, results.LR.accuracy);

%% k-NN (with hyperparameter tuning)
fprintf('   [2/5] Training k-NN with hyperparameter tuning...\n');
tic;

% Test different k values
kValues = [3, 5, 7, 9, 11, 15];
bestK = 5;
bestValAcc = 0;

for k = kValues
    mdl_temp = fitcknn(X_train_norm, Y_train, 'NumNeighbors', k, 'Distance', 'euclidean');
    Y_pred_temp = predict(mdl_temp, X_val_norm);
    acc_temp = sum(Y_pred_temp == Y_val) / length(Y_val);
    if acc_temp > bestValAcc
        bestValAcc = acc_temp;
        bestK = k;
    end
end

fprintf('         Best k=%d (Val Acc=%.4f)\n', bestK, bestValAcc);

mdl = fitcknn(X_train_norm, Y_train, 'NumNeighbors', bestK, 'Distance', 'euclidean');
Y_pred_val = predict(mdl, X_val_norm);
Y_pred_test = predict(mdl, X_test_norm);

valAcc = sum(Y_pred_val == Y_val) / length(Y_val);
valPrec = sum(Y_pred_val & Y_val) / (sum(Y_pred_val) + eps);
valRec = sum(Y_pred_val & Y_val) / (sum(Y_val) + eps);
valF1 = 2 * (valPrec * valRec) / (valPrec + valRec + eps);

results.KNN.accuracy = sum(Y_pred_test == Y_test) / length(Y_test);
results.KNN.precision = sum(Y_pred_test & Y_test) / (sum(Y_pred_test) + eps);
results.KNN.recall = sum(Y_pred_test & Y_test) / (sum(Y_test) + eps);
results.KNN.f1 = 2 * (results.KNN.precision * results.KNN.recall) / (results.KNN.precision + results.KNN.recall + eps);
results.KNN.trainTime = toc;
results.KNN.model = mdl;
results.KNN.bestK = bestK;
results.KNN.val_accuracy = valAcc;
results.KNN.val_precision = valPrec;
results.KNN.val_recall = valRec;
results.KNN.val_f1 = valF1;

fprintf('         Val: Acc=%.4f, F1=%.4f | Test: Acc=%.4f\n\n', valAcc, valF1, results.KNN.accuracy);

%% Decision Tree (with pruning)
fprintf('   [3/5] Training Decision Tree with pruning...\n');
tic;

% Train with higher complexity first
mdl_full = fitctree(X_train_norm, Y_train, 'MaxNumSplits', 50, 'MinLeafSize', 1);

% Prune based on validation set
[~, ~, ~, best_level] = cvloss(mdl_full, 'SubTrees', 'all', 'KFold', 5);
mdl = prune(mdl_full, 'Level', best_level);

Y_pred_val = predict(mdl, X_val_norm);
Y_pred_test = predict(mdl, X_test_norm);

valAcc = sum(Y_pred_val == Y_val) / length(Y_val);
valPrec = sum(Y_pred_val & Y_val) / (sum(Y_pred_val) + eps);
valRec = sum(Y_pred_val & Y_val) / (sum(Y_val) + eps);
valF1 = 2 * (valPrec * valRec) / (valPrec + valRec + eps);

results.DT.accuracy = sum(Y_pred_test == Y_test) / length(Y_test);
results.DT.precision = sum(Y_pred_test & Y_test) / (sum(Y_pred_test) + eps);
results.DT.recall = sum(Y_pred_test & Y_test) / (sum(Y_test) + eps);
results.DT.f1 = 2 * (results.DT.precision * results.DT.recall) / (results.DT.precision + results.DT.recall + eps);
results.DT.trainTime = toc;
results.DT.model = mdl;
results.DT.val_accuracy = valAcc;
results.DT.val_precision = valPrec;
results.DT.val_recall = valRec;
results.DT.val_f1 = valF1;

fprintf('         Pruned to level %d\n', best_level);
fprintf('         Val: Acc=%.4f, F1=%.4f | Test: Acc=%.4f\n\n', valAcc, valF1, results.DT.accuracy);

%% SVM (with comprehensive hyperparameter tuning)
fprintf('   [4/5] Training SVM with hyperparameter tuning...\n');
tic;

% Grid search over kernel and C parameter
kernels = {'linear', 'rbf'};
C_values = [0.1, 0.5, 1, 5, 10];

bestValAcc = 0;
bestKernel = 'rbf';
bestC = 1;

fprintf('         Testing configurations...\n');
for k_idx = 1:length(kernels)
    for c_idx = 1:length(C_values)
        kernel = kernels{k_idx};
        C = C_values(c_idx);
        
        try
            if strcmp(kernel, 'linear')
                mdl_temp = fitcsvm(X_train_norm, Y_train, 'KernelFunction', kernel, ...
                    'BoxConstraint', C, 'Standardize', false);
            else
                mdl_temp = fitcsvm(X_train_norm, Y_train, 'KernelFunction', kernel, ...
                    'BoxConstraint', C, 'KernelScale', 'auto', 'Standardize', false);
            end
            
            Y_pred_temp = predict(mdl_temp, X_val_norm);
            acc_temp = sum(Y_pred_temp == Y_val) / length(Y_val);
            
            if acc_temp > bestValAcc
                bestValAcc = acc_temp;
                bestKernel = kernel;
                bestC = C;
            end
        catch
            continue;
        end
    end
end

fprintf('         Best: kernel=%s, C=%.1f (Val Acc=%.4f)\n', bestKernel, bestC, bestValAcc);

% Train final model with best parameters
if strcmp(bestKernel, 'linear')
    mdl = fitcsvm(X_train_norm, Y_train, 'KernelFunction', bestKernel, ...
        'BoxConstraint', bestC, 'Standardize', false);
else
    mdl = fitcsvm(X_train_norm, Y_train, 'KernelFunction', bestKernel, ...
        'BoxConstraint', bestC, 'KernelScale', 'auto', 'Standardize', false);
end

Y_pred_val = predict(mdl, X_val_norm);
Y_pred_test = predict(mdl, X_test_norm);

valAcc = sum(Y_pred_val == Y_val) / length(Y_val);
valPrec = sum(Y_pred_val & Y_val) / (sum(Y_pred_val) + eps);
valRec = sum(Y_pred_val & Y_val) / (sum(Y_val) + eps);
valF1 = 2 * (valPrec * valRec) / (valPrec + valRec + eps);

results.SVM.accuracy = sum(Y_pred_test == Y_test) / length(Y_test);
results.SVM.precision = sum(Y_pred_test & Y_test) / (sum(Y_pred_test) + eps);
results.SVM.recall = sum(Y_pred_test & Y_test) / (sum(Y_test) + eps);
results.SVM.f1 = 2 * (results.SVM.precision * results.SVM.recall) / (results.SVM.precision + results.SVM.recall + eps);
results.SVM.trainTime = toc;
results.SVM.model = mdl;
results.SVM.bestKernel = bestKernel;
results.SVM.bestC = bestC;
results.SVM.val_accuracy = valAcc;
results.SVM.val_precision = valPrec;
results.SVM.val_recall = valRec;
results.SVM.val_f1 = valF1;

fprintf('         Val: Acc=%.4f, F1=%.4f | Test: Acc=%.4f\n\n', valAcc, valF1, results.SVM.accuracy);

%% Naive Bayes
fprintf('   [5/5] Training Naive Bayes...\n');
tic;

X_train_nb = X_train_norm;
for class = [0, 1]
    classIdx = (Y_train == class);
    for feat = 1:size(X_train_nb, 2)
        featVar = var(X_train_nb(classIdx, feat));
        if featVar < 1e-10
            X_train_nb(classIdx, feat) = X_train_nb(classIdx, feat) + randn(sum(classIdx), 1) * 1e-6;
        end
    end
end

try
    mdl = fitcnb(X_train_nb, Y_train, 'DistributionNames', cfg.models.nb.distributionNames);
    Y_pred_val = predict(mdl, X_val_norm);
    Y_pred_test = predict(mdl, X_test_norm);
    
    valAcc = sum(Y_pred_val == Y_val) / length(Y_val);
    valPrec = sum(Y_pred_val & Y_val) / (sum(Y_pred_val) + eps);
    valRec = sum(Y_pred_val & Y_val) / (sum(Y_val) + eps);
    valF1 = 2 * (valPrec * valRec) / (valPrec + valRec + eps);
    
    results.NB.accuracy = sum(Y_pred_test == Y_test) / length(Y_test);
    results.NB.precision = sum(Y_pred_test & Y_test) / (sum(Y_pred_test) + eps);
    results.NB.recall = sum(Y_pred_test & Y_test) / (sum(Y_test) + eps);
    results.NB.f1 = 2 * (results.NB.precision * results.NB.recall) / (results.NB.precision + results.NB.recall + eps);
    results.NB.trainTime = toc;
    results.NB.model = mdl;
    results.NB.val_accuracy = valAcc;
    results.NB.val_precision = valPrec;
    results.NB.val_recall = valRec;
    results.NB.val_f1 = valF1;
    
    fprintf('         Val: Acc=%.4f, F1=%.4f | Test: Acc=%.4f\n\n', valAcc, valF1, results.NB.accuracy);
catch
    results.NB.accuracy = 0.50;
    results.NB.precision = 0.50;
    results.NB.recall = 0.50;
    results.NB.f1 = 0.50;
    results.NB.trainTime = toc;
    results.NB.model = [];
    results.NB.val_accuracy = 0.50;
    results.NB.val_precision = 0.50;
    results.NB.val_recall = 0.50;
    results.NB.val_f1 = 0.50;
    fprintf('         ⚠️  Failed - using defaults\n\n');
end

%% Summary
fprintf('📊 MODEL COMPARISON:\n');
fprintf('┌────────────────────┬──────────┬───────────┬────────┬────────┐\n');
fprintf('│ Model              │ Accuracy │ Precision │ Recall │   F1   │\n');
fprintf('├────────────────────┼──────────┼───────────┼────────┼────────┤\n');

models = {'LR', 'KNN', 'DT', 'SVM', 'NB'};
modelLabels = {'Logistic Reg', 'k-NN', 'Decision Tree', 'SVM', 'Naive Bayes'};

for i = 1:length(models)
    m = models{i};
    fprintf('│ %-18s │  %.4f  │   %.4f  │ %.4f │ %.4f │\n', ...
        modelLabels{i}, results.(m).accuracy, results.(m).precision, ...
        results.(m).recall, results.(m).f1);
end
fprintf('└────────────────────┴──────────┴───────────┴────────┴────────┘\n\n');

% Find best model (using F1 as tiebreaker for accuracy)
accuracies = [results.LR.accuracy, results.KNN.accuracy, results.DT.accuracy, results.SVM.accuracy, results.NB.accuracy];
f1_scores = [results.LR.f1, results.KNN.f1, results.DT.f1, results.SVM.f1, results.NB.f1];

[maxAcc, bestIdx] = max(accuracies);

% Check for ties in accuracy
tied_indices = find(abs(accuracies - maxAcc) < 1e-6);
if length(tied_indices) > 1
    % Use F1-score as tiebreaker
    [~, best_f1_rel_idx] = max(f1_scores(tied_indices));
    bestIdx = tied_indices(best_f1_rel_idx);
    fprintf('ℹ️  Tie in accuracy - using F1-score as tiebreaker\n');
    fprintf('   Tied models: ');
    for i = 1:length(tied_indices)
        fprintf('%s (F1=%.4f) ', modelLabels{tied_indices(i)}, f1_scores(tied_indices(i)));
    end
    fprintf('\n');
end
fprintf('\n');

fprintf('🏆 Best Model: %s (Accuracy: %.4f, F1: %.4f)\n\n', modelLabels{bestIdx}, maxAcc, f1_scores(bestIdx));

save(fullfile(cfg.paths.results, 'models_trained.mat'), 'results');
fprintf('✓ Phase 4 complete (%.1fs)\n\n', toc(phase4Time));

fprintf('✓ Phase 4 complete (%.1fs)\n\n', toc(phase4Time));

%% PHASE 5: VISUALIZATIONS (10 Figures - 2D + 3D)
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║ PHASE 5: GENERATING VISUALIZATIONS                       ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

phase5Time = tic;

%% Calculate statistical features needed for visualization
fprintf('   Calculating statistical metrics...\n');

cohens_d_vals = zeros(length(featureNames), 1);
p_values_vals = zeros(length(featureNames), 1);
correlations_vals = zeros(length(featureNames), 1);
f_stats_vals = zeros(length(featureNames), 1);

for i = 1:length(featureNames)
    class0 = X_train_norm(Y_train == 0, i);
    class1 = X_train_norm(Y_train == 1, i);
    
    % Cohen's d
    mean0 = mean(class0);
    mean1 = mean(class1);
    n0 = length(class0);
    n1 = length(class1);
    s0 = std(class0);
    s1 = std(class1);
    sp = sqrt(((n0-1)*s0^2 + (n1-1)*s1^2) / (n0 + n1 - 2));
    cohens_d_vals(i) = (mean1 - mean0) / sp;
    
    % p-value (t-test)
    [~, p_values_vals(i)] = ttest2(class0, class1);
    
    % Correlation with target
    correlations_vals(i) = corr(X_train_norm(:, i), double(Y_train), 'Type', 'Pearson');
    
    % F-statistic (from Phase 3)
    f_stats_vals(i) = fStats(i);
end

fprintf('   ✓ Statistical analysis complete\n\n');

%% Generate all visualizations
generate_phase5_visualizations(results, X_train_norm, Y_train, ...
                               X_test_norm, Y_test, ...
                               X_train_before_smote, Y_train_before_smote, ...
                               featureNames, cohens_d_vals, p_values_vals, ...
                               correlations_vals, cfg);

fprintf('✓ Phase 5 complete (%.1fs)\n\n', toc(phase5Time));

%% ========================================================================
%% PHASE 6: EXPORT FOR R

%% ========================================================================
%% PHASE 6: EXPORT FOR R
%% ========================================================================

fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║ PHASE 6: EXPORT DATA FOR R VISUALIZATION                 ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

phase6Time = tic;

%% Export model performance
modelData = table();
modelData.Model = {'Logistic_Regression'; 'KNN'; 'Decision_Tree'; 'SVM'; 'Naive_Bayes'};
modelData.Test_Accuracy = [results.LR.accuracy; results.KNN.accuracy; results.DT.accuracy; results.SVM.accuracy; results.NB.accuracy];
modelData.Test_Precision = [results.LR.precision; results.KNN.precision; results.DT.precision; results.SVM.precision; results.NB.precision];
modelData.Test_Recall = [results.LR.recall; results.KNN.recall; results.DT.recall; results.SVM.recall; results.NB.recall];
modelData.Test_F1 = [results.LR.f1; results.KNN.f1; results.DT.f1; results.SVM.f1; results.NB.f1];

writetable(modelData, fullfile(cfg.paths.forR, 'model_performance.csv'));
fprintf('   ✓ Exported model_performance.csv\n');

%% Export features
featureData = array2table(features, 'VariableNames', featureNames);
featureData.Label = cellstr(labels);
writetable(featureData, fullfile(cfg.paths.forR, 'features_with_labels.csv'));
fprintf('   ✓ Exported features_with_labels.csv\n\n');

fprintf('✓ Phase 6 complete (%.1fs)\n\n', toc(phase6Time));

%% ========================================================================
%% PHASE 7: EXTERNAL VALIDATION (APTOS 2019 DATASET)
%% ========================================================================

fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║ PHASE 7: EXTERNAL VALIDATION - APTOS Dataset             ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

phase7Time = tic;

%% Configuration
aptos_config = struct();
aptos_config.base_url = 'https://raw.githubusercontent.com/vkarthikeya16/data/main/aptos/';
aptos_config.images_url = 'https://raw.githubusercontent.com/vkarthikeya16/data/main/aptos/train_images/';
aptos_config.cache_dir = fullfile(cfg.paths.root, 'aptos_cache');
aptos_config.labels_file = 'train.csv';
aptos_config.num_samples = 800;  % 400 healthy + 400 DR

if ~exist(aptos_config.cache_dir, 'dir')
    mkdir(aptos_config.cache_dir);
end

fprintf('   [1/5] Downloading APTOS labels...\n');

%% Download labels
labels_url = [aptos_config.base_url, aptos_config.labels_file];
labels_cache = fullfile(aptos_config.cache_dir, aptos_config.labels_file);

if ~exist(labels_cache, 'file')
    try
        websave(labels_cache, labels_url);
        fprintf('      ✓ Downloaded labels\n\n');
    catch
        fprintf('      ⚠️  Could not download APTOS labels - skipping external validation\n\n');
        fprintf('✓ Phase 7 skipped (%.1fs)\n\n', toc(phase7Time));
        goto_summary = true;
    end
else
    fprintf('      ✓ Using cached labels\n\n');
end

if ~exist('goto_summary', 'var')
    labels_table = readtable(labels_cache);
    fprintf('      Total APTOS images: %d\n\n', height(labels_table));
    
    %% Create balanced subset
    fprintf('   [2/5] Creating balanced subset (%d samples)...\n', aptos_config.num_samples);
    
    aptos_binary_labels = labels_table.diagnosis > 0;
    healthy_idx = find(~aptos_binary_labels);
    dr_idx = find(aptos_binary_labels);
    
    num_per_class = aptos_config.num_samples / 2;
    
    if length(healthy_idx) >= num_per_class && length(dr_idx) >= num_per_class
        rng(42);
        selected_healthy = healthy_idx(randperm(length(healthy_idx), num_per_class));
        selected_dr = dr_idx(randperm(length(dr_idx), num_per_class));
        
        selected_idx_aptos = [selected_healthy; selected_dr];
        selected_labels_table = labels_table(selected_idx_aptos, :);
        
        fprintf('      ✓ Selected: %d healthy + %d DR = %d total\n\n', ...
            num_per_class, num_per_class, length(selected_idx_aptos));
        
        %% Download images
        fprintf('   [3/5] Downloading %d APTOS images...\n', length(selected_idx_aptos));
        fprintf('      (This may take 5-10 minutes)\n\n');
        
        aptos_images = cell(height(selected_labels_table), 1);
        aptos_labels = selected_labels_table.diagnosis > 0;
        failed_downloads = [];
        
        for i = 1:height(selected_labels_table)
            img_name = [selected_labels_table.id_code{i}, '.png'];
            img_cache = fullfile(aptos_config.cache_dir, img_name);
            
            if mod(i, 100) == 0
                fprintf('      Progress: %d/%d (%.1f%%)\n', i, height(selected_labels_table), ...
                    100*i/height(selected_labels_table));
            end
            
            if ~exist(img_cache, 'file')
                img_url = [aptos_config.images_url, img_name];
                try
                    websave(img_cache, img_url);
                catch
                    failed_downloads = [failed_downloads; i];
                    continue;
                end
            end
            
            try
                img = imread(img_cache);
                if isempty(img) || size(img, 3) ~= 3
                    failed_downloads = [failed_downloads; i];
                    continue;
                end
                aptos_images{i} = img;
            catch
                failed_downloads = [failed_downloads; i];
            end
        end
        
        if ~isempty(failed_downloads)
            aptos_images(failed_downloads) = [];
            aptos_labels(failed_downloads) = [];
        end
        
        fprintf('\n      ✓ Downloaded: %d/%d images\n', length(aptos_images), height(selected_labels_table));
        fprintf('      Failed: %d images\n\n', length(failed_downloads));
        
        %% Extract features
        fprintf('   [4/5] Extracting features from %d images...\n', length(aptos_images));
        
        aptos_features = zeros(length(aptos_images), num_features);
        
        for i = 1:length(aptos_images)
            try
                features_extracted = extract_real_features(aptos_images{i});
                aptos_features(i, :) = features_extracted;
            catch
                aptos_features(i, :) = nan(1, num_features);
            end
            
            if mod(i, 100) == 0
                fprintf('      Progress: %d/%d (%.1f%%)\n', i, length(aptos_images), 100*i/length(aptos_images));
            end
        end
        
        fprintf('      ✓ Feature extraction complete\n\n');
        
        %% Remove invalid samples
        invalid_samples = any(isnan(aptos_features) | isinf(aptos_features), 2);
        if any(invalid_samples)
            fprintf('      Removing %d invalid samples\n', sum(invalid_samples));
            aptos_features(invalid_samples, :) = [];
            aptos_labels(invalid_samples) = [];
        end
        
        %% Apply preprocessing (variance filter, normalization, NO feature selection)
        fprintf('   [5/5] Applying preprocessing...\n');
        
        % Apply same variance filter
        aptos_features = aptos_features(:, validFeaturesIdx);
        
        % Apply same normalization (frozen parameters)
        aptos_features_norm = (aptos_features - mu) ./ sigma;
        
        % Use ALL features (no selection) - same as training
        aptos_features_final = aptos_features_norm;
        
        fprintf('      ✓ Normalized - using ALL %d features\n\n', size(aptos_features_final, 2));
        
        %% Evaluate models
        fprintf('📊 EVALUATING MODELS ON APTOS:\n\n');
        fprintf('┌────────────────────┬──────────┬───────────┬────────┬────────┐\n');
        fprintf('│ Model              │ Accuracy │ Precision │ Recall │   F1   │\n');
        fprintf('├────────────────────┼──────────┼───────────┼────────┼────────┤\n');
        
        external_results = struct();
        aptos_labels_numeric = double(aptos_labels);
        
        for i = 1:5
            m = models{i};
            
            try
                % Predict
                if i == 1  % Logistic Regression
                    probs = glmval(results.LR.model, aptos_features_final, 'logit');
                    pred = double(probs > 0.5);
                else
                    pred = predict(results.(m).model, aptos_features_final);
                    if iscategorical(pred)
                        pred = double(pred) - 1;
                    elseif islogical(pred)
                        pred = double(pred);
                    end
                end
                
                % Calculate metrics
                tp = sum(pred == 1 & aptos_labels_numeric == 1);
                tn = sum(pred == 0 & aptos_labels_numeric == 0);
                fp = sum(pred == 1 & aptos_labels_numeric == 0);
                fn = sum(pred == 0 & aptos_labels_numeric == 1);
                
                acc = (tp + tn) / length(aptos_labels_numeric);
                prec = tp / (tp + fp + eps);
                rec = tp / (tp + fn + eps);
                f1 = 2 * prec * rec / (prec + rec + eps);
                
                external_results.(m).accuracy = acc;
                external_results.(m).precision = prec;
                external_results.(m).recall = rec;
                external_results.(m).f1 = f1;
                
                fprintf('│ %-18s │  %.4f  │   %.4f  │ %.4f │ %.4f │\n', ...
                    modelLabels{i}, acc, prec, rec, f1);
                
            catch ME
                fprintf('│ %-18s │  ERROR   │   -----   │ ------ │ ------ │\n', modelLabels{i});
                external_results.(m).accuracy = 0;
                external_results.(m).precision = 0;
                external_results.(m).recall = 0;
                external_results.(m).f1 = 0;
            end
        end
        
        fprintf('└────────────────────┴──────────┴───────────┴────────┴────────┘\n\n');
        
        % Find best on external
        ext_accuracies = [external_results.LR.accuracy, external_results.KNN.accuracy, ...
                          external_results.DT.accuracy, external_results.SVM.accuracy, ...
                          external_results.NB.accuracy];
        [best_ext_acc, best_ext_idx] = max(ext_accuracies);
        fprintf('🏆 Best on APTOS: %s (Accuracy: %.4f)\n\n', modelLabels{best_ext_idx}, best_ext_acc);
        
        %% Save results
        external_validation = struct();
        external_validation.results = external_results;
        external_validation.aptos_labels = aptos_labels;
        external_validation.model_names = modelLabels;
        external_validation.num_samples = length(aptos_labels);
        
        save(fullfile(cfg.paths.results, 'external_validation.mat'), 'external_validation');
        
        %% Export for R
        ext_modelData = table();
        ext_modelData.Model = {'Logistic_Regression'; 'KNN'; 'Decision_Tree'; 'SVM'; 'Naive_Bayes'};
        ext_modelData.Accuracy = [external_results.LR.accuracy; external_results.KNN.accuracy; ...
                                  external_results.DT.accuracy; external_results.SVM.accuracy; ...
                                  external_results.NB.accuracy];
        ext_modelData.Precision = [external_results.LR.precision; external_results.KNN.precision; ...
                                   external_results.DT.precision; external_results.SVM.precision; ...
                                   external_results.NB.precision];
        ext_modelData.Recall = [external_results.LR.recall; external_results.KNN.recall; ...
                                external_results.DT.recall; external_results.SVM.recall; ...
                                external_results.NB.recall];
        ext_modelData.F1 = [external_results.LR.f1; external_results.KNN.f1; ...
                            external_results.DT.f1; external_results.SVM.f1; ...
                            external_results.NB.f1];
        
        writetable(ext_modelData, fullfile(cfg.paths.forR, 'external_validation_metrics.csv'));
        fprintf('      ✓ Exported external_validation_metrics.csv\n\n');
        
    else
        fprintf('      ⚠️  Insufficient samples - skipping\n\n');
    end
end

fprintf('✓ Phase 7 complete (%.1fs)\n\n', toc(phase7Time));

%% ========================================================================
%% SUMMARY
%% ========================================================================

fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║ ✅ PIPELINE COMPLETE!                                      ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

fprintf('⏱️  Total: %.1fs (%.1f min)\n\n', toc(totalTime), toc(totalTime)/60);

fprintf('📁 OUTPUTS:\n');
fprintf('  • Results: %s\n', cfg.paths.results);
fprintf('  • Figures: %s\n', cfg.paths.figures);
fprintf('  • R Data:  %s\n\n', cfg.paths.forR);

fprintf('📊 FINAL RESULTS:\n');
fprintf('  • Best Model (Internal): %s\n', modelLabels{bestIdx});
fprintf('  • Test Accuracy: %.2f%%\n', maxAcc*100);
fprintf('  • Features Used: ALL %d features\n', num_features);
fprintf('  • Training Samples: %d (after SMOTE)\n', length(Y_train));
fprintf('  • Test Samples: %d\n', length(Y_test));

% Show external validation results if available
if exist('external_results', 'var')
    fprintf('\n📊 EXTERNAL VALIDATION (APTOS):\n');
    fprintf('  • Best Model (External): %s\n', modelLabels{best_ext_idx});
    fprintf('  • APTOS Accuracy: %.2f%%\n', best_ext_acc*100);
    fprintf('  • APTOS Samples: %d\n', length(aptos_labels));
    fprintf('  • Generalization Gap: %.2f%%\n', (maxAcc - best_ext_acc)*100);
end

fprintf('\n');

%% ========================================================================
%% HELPER FUNCTIONS
%% ========================================================================

function success = download_robust(url, filepath, max_retries, timeout_sec)
    success = false;
    for attempt = 1:max_retries
        try
            options = weboptions('Timeout', timeout_sec);
            websave(filepath, url, options);
            if exist(filepath, 'file')
                finfo = dir(filepath);
                if finfo.bytes > 0
                    success = true;
                    return;
                end
            end
        catch
            if attempt < max_retries
                pause(2);
            end
        end
    end
end

function features = extract_real_features(img)
    features = zeros(1, 20);
    
    %% Preprocessing
    img_resized = imresize(img, [512, 512]);
    
    %% Grayscale conversion
    if size(img_resized, 3) == 1
        grayImg = img_resized;
    else
        try
            grayImg = rgb2gray(img_resized);
        catch
            grayImg = 0.299 * double(img_resized(:,:,1)) + ...
                      0.587 * double(img_resized(:,:,2)) + ...
                      0.114 * double(img_resized(:,:,3));
            grayImg = uint8(grayImg);
        end
    end
    
    %% Intensity Statistics (Features 1-4)
    intensities = double(grayImg(:));
    features(1) = mean(intensities);
    features(2) = std(intensities);
    features(3) = skewness(intensities);
    features(4) = kurtosis(intensities);
    
    %% Color Channel Features (Features 5-7)
    if size(img_resized, 3) == 3
        features(5) = mean(mean(img_resized(:,:,1)));
        features(6) = mean(mean(img_resized(:,:,2)));
        features(7) = mean(mean(img_resized(:,:,3)));
    else
        features(5:7) = mean(intensities);
    end
    
    %% GLCM Texture Features (Features 8-11)
    try
        grayImg_glcm = imresize(grayImg, [256, 256]);
        glcm = graycomatrix(grayImg_glcm, 'NumLevels', 8, 'GrayLimits', [], 'Offset', [0 1], 'Symmetric', true);
        stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
        
        if length(stats.Contrast) > 1
            features(8) = mean(stats.Contrast);
            features(9) = mean(stats.Correlation);
            features(10) = mean(stats.Energy);
            features(11) = mean(stats.Homogeneity);
        else
            features(8) = stats.Contrast;
            features(9) = stats.Correlation;
            features(10) = stats.Energy;
            features(11) = stats.Homogeneity;
        end
    catch
        features(8:11) = [0, 0, 0, 0];
    end
    
    %% Additional Texture Features (Features 12-14)
    try
        img_quantized = round(double(grayImg) / 255 * 7);
        features(12) = entropy(uint8(img_quantized));
    catch
        features(12) = 0;
    end
    
    if size(img_resized, 3) == 3
        features(13) = features(5) / (features(6) + eps);
    else
        features(13) = 1.0;
    end
    
    features(14) = max(intensities);
    
    %% Lesion Morphology (Features 15-17)
    try
        threshold = graythresh(grayImg) * 0.7;
        bw = imbinarize(grayImg, threshold);
        bw = ~bw;
        bw = bwareaopen(bw, 50);
        
        if any(bw(:))
            props = regionprops(bw, 'Area', 'Perimeter', 'Eccentricity');
            if ~isempty(props)
                [~, idx] = max([props.Area]);
                features(15) = props(idx).Area;
                features(16) = props(idx).Perimeter;
                features(17) = props(idx).Eccentricity;
            else
                features(15:17) = [0, 0, 0];
            end
        else
            features(15:17) = [0, 0, 0];
        end
    catch
        features(15:17) = [0, 0, 0];
    end
    
    %% Vascular Features (Features 18-20)
    try
        if size(img_resized, 3) == 3
            greenChannel = img_resized(:,:,2);
        else
            greenChannel = grayImg;
        end
        
        se = strel('disk', 3);
        tophat = imtophat(greenChannel, se);
        vessel_threshold = graythresh(tophat);
        vessels = imbinarize(tophat, vessel_threshold);
        vessels = bwareaopen(vessels, 20);
        
        if any(vessels(:))
            features(18) = sum(vessels(:)) / numel(vessels) * 100;
            cc = bwconncomp(vessels);
            features(19) = cc.NumObjects;
            skel = bwmorph(vessels, 'skel', Inf);
            features(20) = sum(skel(:)) / (features(19) + eps);
        else
            features(18:20) = [0, 0, 0];
        end
    catch
        features(18:20) = [0, 0, 0];
    end
    
    %% Final validation
    features(isnan(features)) = 0;
    features(isinf(features)) = 0;
end
%% ========================================================================
%% PHASE 5 VISUALIZATION FUNCTION - All 10 Figures
%% ========================================================================

function generate_phase5_visualizations(results, X_train, Y_train, ...
                                       X_test, Y_test, ...
                                       X_before_smote, Y_before_smote, ...
                                       featureNames, cohens_d, p_vals, corrs, cfg)
    
    % Define colors
    color_healthy = [0 115 178]/255;
    color_dr = [222 143 5]/255;
    color_healthy_synth = [100 180 220]/255;
    color_dr_synth = [255 180 100]/255;
    
    colors_models = [
        0.12 0.47 0.71;  % Blue
        0.20 0.63 0.17;  % Green
        0.89 0.10 0.11;  % Red
        1.00 0.50 0.00;  % Orange
        0.42 0.24 0.60   % Purple
    ];
    
    model_names = {'Logistic Reg', 'k-NN', 'Decision Tree', 'SVM', 'Naive Bayes'};
    model_fields = {'LR', 'KNN', 'DT', 'SVM', 'NB'};
    
    %% Figure 1: Model Performance Comparison
    fprintf('   [1/10] Model performance comparison...\n');
    
    perf_data = zeros(5, 4);
    for i = 1:5
        perf_data(i, 1) = results.(model_fields{i}).accuracy;
        perf_data(i, 2) = results.(model_fields{i}).precision;
        perf_data(i, 3) = results.(model_fields{i}).recall;
        perf_data(i, 4) = results.(model_fields{i}).f1;
    end
    
    fig1 = figure('Position', [100 100 1200 600], 'Color', 'w', 'Visible', 'off');
    b = bar(perf_data, 'grouped');
    for i = 1:4
        b(i).FaceColor = colors_models(i, :);
        b(i).EdgeColor = 'k';
        b(i).LineWidth = 1;
    end
    set(gca, 'XTickLabel', model_names, 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Score', 'FontSize', 12, 'FontWeight', 'bold');
    title('Model Performance Comparison (Internal Test Set)', 'FontSize', 13, 'FontWeight', 'bold');
    ylim([0 1]);
    ytickformat('percentage');
    legend({'Accuracy', 'Precision', 'Recall', 'F1'}, 'Location', 'south', ...
           'Orientation', 'horizontal', 'FontSize', 10);
    grid on;
    saveas(fig1, fullfile(cfg.paths.figures, '1_Model_Performance.png'));
    close(fig1);
    
    %% Figure 2: Confusion Matrix (Best Model - SVM)
    fprintf('   [2/10] Confusion matrix (SVM)...\n');
    
    Y_pred = predict(results.SVM.model, X_test);
    if iscategorical(Y_pred), Y_pred = double(Y_pred) - 1; end
    cm = confusionmat(Y_test, Y_pred);
    
    fig2 = figure('Position', [100 100 800 700], 'Color', 'w', 'Visible', 'off');
    imagesc(cm);
    colormap([1 1 1; 0.9 0.9 1; 0.6 0.6 1; 0.4 0.4 1]);
    
    % Annotate cells
    for i = 1:2
        for j = 1:2
            text(j, i, sprintf('%d\n(%.1f%%)', cm(i,j), 100*cm(i,j)/sum(cm(:))), ...
                'HorizontalAlignment', 'center', 'FontSize', 14, ...
                'FontWeight', 'bold', 'Color', 'k');
        end
    end
    
    set(gca, 'XTick', 1:2, 'YTick', 1:2, ...
        'XTickLabel', {'Healthy', 'DR'}, ...
        'YTickLabel', {'Healthy', 'DR'}, ...
        'FontSize', 11, 'FontWeight', 'bold');
    xlabel('Predicted', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Actual', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('Confusion Matrix: SVM (Accuracy: %.2f%%)', results.SVM.accuracy*100), ...
          'FontSize', 13, 'FontWeight', 'bold');
    saveas(fig2, fullfile(cfg.paths.figures, '2_Confusion_Matrix_SVM.png'));
    close(fig2);
    
   %% Figure 3: Feature Importance (Cohen's d)
fprintf('   [3/10] Feature importance (Cohen''s d)...\n');

[~, sort_idx] = sort(abs(cohens_d), 'descend');
top10 = sort_idx(1:min(10, length(featureNames)));

% Clean feature names (remove underscores)
clean_names = cell(size(top10));
for i = 1:length(top10)
    clean_names{i} = strrep(featureNames{top10(i)}, '_', ' ');
end

fig3 = figure('Position', [100 100 1000 650], 'Color', 'w', 'Visible', 'off');

% Color bars by direction (positive = DR higher, negative = Healthy higher)
colors_bar = zeros(length(top10), 3);
for i = 1:length(top10)
    if cohens_d(top10(i)) > 0
        colors_bar(i, :) = [0.3 0.6 0.8];  % Blue for DR higher
    else
        colors_bar(i, :) = [0.8 0.4 0.3];  % Red for Healthy higher
    end
end

barh(1:length(top10), cohens_d(top10), 'FaceColor', 'flat', 'CData', colors_bar, ...
     'EdgeColor', 'k', 'LineWidth', 1);
hold on;

% Add p-value annotations
for i = 1:length(top10)
    idx = top10(i);
    if p_vals(idx) < 0.001
        sig_text = 'p<0.001';
    else
        sig_text = sprintf('p=%.3f', p_vals(idx));
    end
    
    % Position text to the right of positive bars, left of negative bars
    if cohens_d(idx) > 0
        x_pos = cohens_d(idx) + 0.12;
        h_align = 'left';
    else
        x_pos = cohens_d(idx) - 0.12;
        h_align = 'right';
    end
    
    text(x_pos, i, sig_text, 'FontSize', 8, 'FontWeight', 'bold', ...
         'HorizontalAlignment', h_align);
end
hold off;

set(gca, 'YTick', 1:length(top10), 'YTickLabel', clean_names, ...
    'FontSize', 10, 'FontWeight', 'bold', 'TickLabelInterpreter', 'none');
xlabel('Cohen''s d (Effect Size)', 'FontSize', 12, 'FontWeight', 'bold');
title({'Top 10 Discriminative Features (Cohen''s d)', ...
       'Blue = Higher in DR | Red = Higher in Healthy'}, ...
       'FontSize', 13, 'FontWeight', 'bold');
grid on;
xlim([min(cohens_d(top10))-0.3, max(cohens_d(top10))+0.3]);

saveas(fig3, fullfile(cfg.paths.figures, '3_Feature_Importance_Cohens_d.png'));
close(fig3);

    %% Figure 4: PCA 2D Projection
    fprintf('   [4/10] PCA 2D projection...\n');
    
    [~, score, ~, ~, explained] = pca(X_train);
    
    fig4 = figure('Position', [100 100 900 700], 'Color', 'w', 'Visible', 'off');
    scatter(score(Y_train==0, 1), score(Y_train==0, 2), 50, color_healthy, 'filled', ...
            'MarkerEdgeColor', 'k', 'LineWidth', 0.5, 'DisplayName', 'Healthy');
    hold on;
    scatter(score(Y_train==1, 1), score(Y_train==1, 2), 50, color_dr, 'filled', ...
            'MarkerEdgeColor', 'k', 'LineWidth', 0.5, 'DisplayName', 'DR');
    hold off;
    
    xlabel(sprintf('PC1 (%.1f%% variance)', explained(1)), 'FontSize', 12, 'FontWeight', 'bold');
    ylabel(sprintf('PC2 (%.1f%% variance)', explained(2)), 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('PCA 2D Projection (Total: %.1f%% variance)', sum(explained(1:2))), ...
          'FontSize', 13, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 10);
    grid on;
    saveas(fig4, fullfile(cfg.paths.figures, '4_PCA_2D_Projection.png'));
    close(fig4);
    
    %% Figure 5: Correlation Matrix
    fprintf('   [5/10] Correlation matrix...\n');
    
    % Calculate correlation matrix
    corr_matrix = corr(X_train);
    
    fig5 = figure('Position', [100 100 1000 900], 'Color', 'w', 'Visible', 'off');
    imagesc(corr_matrix);
    colormap(redblue);
    colorbar;
    caxis([-1 1]);
    
    % Clean feature names for display
    clean_feat_names = cell(size(featureNames));
    for i = 1:length(featureNames)
        clean_feat_names{i} = strrep(featureNames{i}, '_', ' ');
    end
    
    set(gca, 'XTick', 1:length(featureNames), 'YTick', 1:length(featureNames), ...
        'XTickLabel', clean_feat_names, 'YTickLabel', clean_feat_names, ...
        'XTickLabelRotation', 45, 'TickLabelInterpreter', 'none', 'FontSize', 8);
    title('Feature Correlation Matrix', 'FontSize', 13, 'FontWeight', 'bold');
    saveas(fig5, fullfile(cfg.paths.figures, '5_Correlation_Matrix.png'));
    close(fig5);
    
    %% Figure 6: Feature Distributions (Top 3)
fprintf('   [6/10] Feature distributions...\n');

[~, top_feat_idx] = sort(abs(cohens_d), 'descend');
top3 = top_feat_idx(1:3);

fig6 = figure('Position', [100 100 1200 400], 'Color', 'w', 'Visible', 'off');

for i = 1:3
    subplot(1, 3, i);
    
    feat_idx = top3(i);
    healthy_data = X_train(Y_train == 0, feat_idx);
    dr_data = X_train(Y_train == 1, feat_idx);
    
    % Violin plot simulation with histogram
    bins = 20;
    histogram(healthy_data, bins, 'FaceColor', color_healthy, ...
              'FaceAlpha', 0.5, 'EdgeColor', 'k');
    hold on;
    histogram(dr_data, bins, 'FaceColor', color_dr, ...
              'FaceAlpha', 0.5, 'EdgeColor', 'k');
    hold off;
    
    feat_name = strrep(featureNames{feat_idx}, '_', ' ');
    title(feat_name, 'Interpreter', 'none', 'FontWeight', 'bold');
    xlabel('Feature Value');
    ylabel('Frequency');
    legend({'Healthy', 'DR'}, 'Location', 'best');
    grid on;
end

sgtitle('Top 3 Feature Distributions', 'FontSize', 14, 'FontWeight', 'bold');
saveas(fig6, fullfile(cfg.paths.figures, '6_Feature_Distributions.png'));
close(fig6);


%% Figure 7: Model Comparison Radar Chart
fprintf('   [7/10] Model metrics radar chart...\n');

fig7 = figure('Position', [100 100 900 800], 'Color', 'w', 'Visible', 'off');

% Create radar chart data
metrics = {'Accuracy', 'Precision', 'Recall', 'F1'};
model_data = zeros(5, 4);
for i = 1:5
    model_data(i, 1) = results.(model_fields{i}).accuracy;
    model_data(i, 2) = results.(model_fields{i}).precision;
    model_data(i, 3) = results.(model_fields{i}).recall;
    model_data(i, 4) = results.(model_fields{i}).f1;
end

% Create polar axes
ax = polaraxes;
hold on;

% FIXED: Plot each model with correct angles
angles = linspace(0, 2*pi, length(metrics)+1);  % 4 metrics + close loop = 5 points
theta_plot = angles(1:end-1);  % Remove last duplicate for plotting
theta_plot = [theta_plot, theta_plot(1)];  % Close the loop

for i = 1:5
    values = [model_data(i, :), model_data(i, 1)];  % Close the loop
    polarplot(theta_plot, values, 'o-', 'LineWidth', 2, ...
              'MarkerSize', 8, 'Color', colors_models(i, :), ...
              'DisplayName', model_names{i});
end
hold off;

% Formatting
ax.ThetaTick = rad2deg(angles(1:end-1));  % Don't include duplicate
ax.ThetaTickLabel = metrics;
ax.RLim = [0 1];
ax.RTick = 0:0.2:1;
ax.RTickLabel = arrayfun(@(x) sprintf('%.1f', x), 0:0.2:1, 'UniformOutput', false);
ax.FontSize = 11;
ax.FontWeight = 'bold';

title('Model Performance Metrics (Radar Chart)', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'southoutside', 'Orientation', 'horizontal', 'FontSize', 10);
grid on;

saveas(fig7, fullfile(cfg.paths.figures, '7_Model_Metrics_Radar.png'));
close(fig7);

%% Figure 8: Class Balance Visualization
fprintf('   [8/10] Class balance analysis...\n');

fig8 = figure('Position', [100 100 1000 600], 'Color', 'w', 'Visible', 'off');

% Data for bar chart
original_counts = [sum(Y_before_smote == 0), sum(Y_before_smote == 1)];
smote_counts = [sum(Y_train == 0), sum(Y_train == 1)];
test_counts = [sum(Y_test == 0), sum(Y_test == 1)];

% Create grouped bar chart
data_matrix = [original_counts; smote_counts; test_counts];
bar_handle = bar(data_matrix, 'grouped');
bar_handle(1).FaceColor = color_healthy;
bar_handle(1).EdgeColor = 'k';
bar_handle(1).LineWidth = 1;
bar_handle(2).FaceColor = color_dr;
bar_handle(2).EdgeColor = 'k';
bar_handle(2).LineWidth = 1;

% Add value labels on bars
for i = 1:3
    for j = 1:2
        x_pos = i + (j-1.5)*0.15;
        y_pos = data_matrix(i, j);
        text(x_pos, y_pos + 5, sprintf('%d', data_matrix(i, j)), ...
             'HorizontalAlignment', 'center', 'FontSize', 11, ...
             'FontWeight', 'bold', 'Color', 'k');
    end
end

% Formatting
set(gca, 'XTickLabel', {'Pre-SMOTE\n(Training)', 'Post-SMOTE\n(Training)', 'Test Set'}, ...
    'FontSize', 11, 'FontWeight', 'bold');
ylabel('Number of Samples', 'FontSize', 12, 'FontWeight', 'bold');
title({'Dataset Class Distribution Analysis', ...
       sprintf('SMOTE Balancing: %d → %d samples (%.1f%% increase)', ...
       sum(original_counts), sum(smote_counts), ...
       100*(sum(smote_counts)/sum(original_counts)-1))}, ...
       'FontSize', 13, 'FontWeight', 'bold');

legend({'Healthy (Class 0)', 'DR (Class 1)'}, 'Location', 'northwest', 'FontSize', 10);
ylim([0 max(data_matrix(:)) + 30]);
grid on;
box on;

% Add balance ratio annotations
for i = 1:3
    ratio = data_matrix(i, 2) / data_matrix(i, 1);
    text(i, max(data_matrix(i, :)) + 15, sprintf('Ratio: 1:%.2f', ratio), ...
         'HorizontalAlignment', 'center', 'FontSize', 9, ...
         'FontWeight', 'bold', 'Color', [0.5 0.5 0.5]);
end

saveas(fig8, fullfile(cfg.paths.figures, '8_Class_Balance_Analysis.png'));
close(fig8);

    
    %% Figure 9: 3D SMOTE Sample Distribution
fprintf('   [9/10] 3D SMOTE sample distribution...\n');

% PCA on pre-SMOTE data
[coeff, ~, ~, ~, explained_3d] = pca(X_before_smote);

% Project post-SMOTE data onto same PC space
score_after = X_train * coeff;
n_orig = size(X_before_smote, 1);

fig9 = figure('Position', [100 100 1200 900], 'Color', 'w', 'Visible', 'off');

% Identify sample types
orig_healthy_idx = (Y_train(1:n_orig) == 0);
orig_dr_idx = (Y_train(1:n_orig) == 1);

% Plot original samples (large filled circles)
h1 = scatter3(score_after(orig_healthy_idx, 1), score_after(orig_healthy_idx, 2), ...
         score_after(orig_healthy_idx, 3), 80, color_healthy, 'filled', ...
         'MarkerEdgeColor', 'k', 'LineWidth', 1);
hold on;
h2 = scatter3(score_after(orig_dr_idx, 1), score_after(orig_dr_idx, 2), ...
         score_after(orig_dr_idx, 3), 80, color_dr, 'filled', ...
         'MarkerEdgeColor', 'k', 'LineWidth', 1);

% Plot synthetic samples (small open circles)
h3 = [];
if size(score_after, 1) > n_orig
    synth_idx = n_orig+1:size(score_after, 1);
    h3 = scatter3(score_after(synth_idx, 1), score_after(synth_idx, 2), ...
             score_after(synth_idx, 3), 40, color_healthy_synth, 'o', ...
             'MarkerEdgeColor', color_healthy, 'LineWidth', 1);
end
hold off;

xlabel(sprintf('PC1 (%.1f%%)', explained_3d(1)), 'FontSize', 12, 'FontWeight', 'bold');
ylabel(sprintf('PC2 (%.1f%%)', explained_3d(2)), 'FontSize', 12, 'FontWeight', 'bold');
zlabel(sprintf('PC3 (%.1f%%)', explained_3d(3)), 'FontSize', 12, 'FontWeight', 'bold');

title({'3D SMOTE Sample Distribution (PCA Projection)', ...
       sprintf('Original: %d → After SMOTE: %d (%.1f%% increase)', ...
       n_orig, size(score_after, 1), 100*(size(score_after,1)/n_orig-1))}, ...
       'FontSize', 13, 'FontWeight', 'bold');

% Fixed legend
if ~isempty(h3)
    legend([h1, h2, h3], {'Original Healthy', 'Original DR', 'Synthetic (SMOTE)'}, ...
           'Location', 'best', 'FontSize', 10);
else
    legend([h1, h2], {'Original Healthy', 'Original DR'}, ...
           'Location', 'best', 'FontSize', 10);
end

grid on;
view(45, 30);
saveas(fig9, fullfile(cfg.paths.figures, '9_SMOTE_3D_Distribution.png'));
close(fig9);
    
    %% Figure 10: 3D Feature Selection Landscape
    fprintf('   [10/10] 3D feature selection landscape...\n');
    
    % Prepare 3D volcano plot data
    x_data = corrs;  % Correlation with target
    y_data = -log10(p_vals);  % Significance
    y_data(isinf(y_data)) = 100;  % Cap infinite values
    z_data = abs(cohens_d);  % Effect size
    
    fig10 = figure('Position', [100 100 1200 900], 'Color', 'w', 'Visible', 'off');
    
    % 3D scatter colored by effect size
    scatter3(x_data, y_data, z_data, 150, z_data, 'filled', ...
             'MarkerEdgeColor', 'k', 'LineWidth', 2);
    
    % Label top 5 features
    [~, top_idx] = sort(abs(cohens_d), 'descend');
    for i = 1:min(5, length(featureNames))
        idx = top_idx(i);
        feat_name = strrep(featureNames{idx}, '_', ' ');
        text(x_data(idx), y_data(idx), z_data(idx)+0.1, feat_name, ...
             'FontSize', 9, 'FontWeight', 'bold', 'Interpreter', 'none', ...
             'HorizontalAlignment', 'center', 'BackgroundColor', 'white', ...
             'EdgeColor', 'k');
    end
    
    xlabel('Correlation with DR (Pearson r)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('-log₁₀(p-value) [Significance]', 'FontSize', 12, 'FontWeight', 'bold');
    zlabel('|Cohen''s d| [Effect Size]', 'FontSize', 12, 'FontWeight', 'bold');
    title({'3D Feature Selection Landscape (Volcano Plot)', ...
           sprintf('Top feature: %s (d=%.2f, p<%.0e)', ...
           strrep(featureNames{top_idx(1)}, '_', ' '), ...
           abs(cohens_d(top_idx(1))), p_vals(top_idx(1)))}, ...
           'FontSize', 13, 'FontWeight', 'bold');
    
    % Colorbar
    colormap(jet);
    cb = colorbar;
    cb.Label.String = '|Cohen''s d| (Effect Size)';
    cb.Label.FontSize = 11;
    cb.Label.FontWeight = 'bold';
    caxis([0 max(z_data)]);
    
    grid on;
    view(45, 30);
    saveas(fig10, fullfile(cfg.paths.figures, '10_Feature_Selection_3D_Landscape.png'));
    close(fig10);
    
    fprintf('\n   ✓ All 10 figures saved to: %s\n', cfg.paths.figures);
end

%% Red-Blue colormap for correlation matrix
function c = redblue(m)
    if nargin < 1
        m = size(get(gcf,'colormap'),1);
    end
    
    if mod(m,2) == 0
        m1 = m/2;
        r = (0:m1-1)'/max(m1-1,1);
        g = r;
        r = [ones(m1,1); flipud(r)];
        g = [g; flipud(g)];
        b = flipud(r);
    else
        m1 = floor(m/2);
        r = (0:m1-1)'/max(m1,1);
        g = r;
        r = [ones(m1+1,1); flipud(r)];
        g = [g; 1; flipud(g)];
        b = flipud(r);
    end
    
    c = [r g b];
end