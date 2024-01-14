%% CLASSIFICATORE BINARIO CON RETE NEURALE 
clear; clc
load workspace_breast.mat

% Count Features and Samples
nFeatures = size(genData, 1);
nPatients = size(genData, 2);
fprintf("\nCounting %d features and %d samples", nFeatures, nPatients);

y=zeros(size(clinicalData,1),1)';
labels_low_mask = zeros(nPatients,1);
labels_high_mask = zeros(nPatients,1);
labels_low = [];
labels_high = [];
label_mask=[];
for i = 1:nPatients
    label = clinicalData{i,1}{1};
    if label == "low"
        labels_low = [labels_low, i];
        label_mask(i)=0;
        labels_low_mask(i) = 1;
    else
        y(i)=1;
        labels_high = [labels_high, i];
        labels_high_mask(i) = 1;
        label_mask(i)=1;
    end
end
nLows = numel(labels_low);
nHighs = numel(labels_high);


% normalizzazione con media geometrica
genDataArray = table2array(genData);
counts = genDataArray; %counts=x

pseudoRefSample = geomean(counts,2);
nz = pseudoRefSample > 0;
ratios = bsxfun(@rdivide,counts(nz,:),pseudoRefSample(nz));
sizeFactors = median(ratios,1);

% transform to common scale
normCounts = bsxfun(@rdivide,counts,sizeFactors);
fs_bonferroni=false;
fs_bh=false;
%% BOXPLOT NORMALIZZAZIONE
figure;

subplot(2,1,1)
maboxplot(log2(counts(:,1:4)),'title','Raw Read Count','orientation','horizontal')
ylabel('sample');
xlabel('log2(counts)');

subplot(2,1,2)
maboxplot(log2(normCounts(:,1:4)),'title','Normalized Read Count','orientation','horizontal')
ylabel('sample');
xlabel('log2(counts)');

%% Constant Variance Link
x=normCounts;
%%
tConstant = nbintest(x(:,labels_low),x(:,labels_high),...
    'VarianceLink','Constant');
%h = plotVarianceLink(tConstant, 'Compare', 0);
% set custom title
%h(1).Title.String = 'Variance Link on Low Samples';
%h(2).Title.String = 'Variance Link on High Samples';
p_original = tConstant.pValue;
alpha = 0.05;

%% FDR Benjamini-Hochberg
    % Function 'fdr_BH' computes the Benjamini-Hochberg correction of the     %
    % False Discovery Rate for multiple comparisons. The adjusted p-values of %
    % Gordon Smyth are also provided.     
fs_bh=false;
if fs_bh 
    [c_pvalues, c_alpha, bh] = fdr_BH(p_original, alpha, true);
    x_bh=x(bh,:);
    fprintf('\nBenjamini-hochberg correction selects %d features',sum(c_pvalues < alpha))  
end

% FWER Bonferroni
% Function 'fwer_bonf' computes the Bonferroni correction of Family-Wise  %
% Error Rate for multiple comparisons.
fs_bonferroni=true;
if fs_bonferroni
    [c_pvalues, c_alpha, fs_bonf] = fwer_bonf(p_original, alpha);
    x_bonf=x(fs_bonf,:);
   
    fprintf('\nBonferroni correction selects %d features',sum(c_pvalues < alpha))
    %numel(c_pvalues)
end

%% ----------------------CROSS VALIDATION ON NN-------------------------
%clearvars -except normCounts label_mask x_bonf t_bonf fs_bonferroni x_bh
%rng('default') % For reproducibility
if fs_bonferroni
    x=x_bonf'; 
elseif fs_bh
    x=x_bh';
else
    x=normCounts'; 
end
t=label_mask';
k=10; 
AUC_cv=[]; roc_train={}; roc_test={};accuracy=[];precision=[];recall=[];jaccard=[];fscore=[];
tn=[];fp=[];fn=[];tp=[];
cv = cvpartition(length(label_mask),'KFold',k,'Stratify',false)
for i=1:k
    cv_test=cv.test(i);
    x_train=x(not(cv_test),:)'; 
    x_test=x(cv_test,:)';
    t_train=[]; t_test=[];
    for j=1:length(cv_test)
        if cv_test(j) == 0
           t_train(end+1)=t(j,:);
        else
           t_test(end+1)=t(j,:);
        end
    end
    
    %oversampling?
    [x_train_os,t_train_os]=oversampling(x_train',t_train');
    x_train=x_train_os'; t_train=t_train_os';
    
    %LDA
    do_lda=false;
    if do_lda
        [x_train_new,x_test_new]=lda_fun(x_train',x_test',t_train);
        x_train=x_train_new; x_test=x_test_new;
    end
    %PCA
    do_pca=false;
    if do_pca && do_lda
        [x_train_new,x_test_new]=pca_fun(x_train,x_test);
        x_train=x_train_new; x_test=x_test_new;
    elseif do_pca
        [x_train_new,x_test_new]=pca_fun(x_train',x_test');
        x_train=x_train_new; x_test=x_test_new;
    elseif do_lda
        x_train=x_train'; x_test=x_test';
    end
   
    
    hiddenLayerSize = [10 10];
  
    trainFcn = 'traingdx';
    
    costFunction = 'crossentropy';
    epochs = 200;
    lastLayerActivationFunction = 'logsig';
    
    net = patternnet(hiddenLayerSize, trainFcn);
    net.trainParam.epochs = epochs;
    net.performFcn = costFunction;
    net.layers{end}.transferFcn = lastLayerActivationFunction;
    
    net.divideFcn = 'dividerand';
    net.divideParam.trainRatio = 100/100;
    net.divideParam.valRatio   = 0/100;
    net.divideParam.testRatio  = 0/100;
    
    
    net = configure(net,x_train,t_train);
    net=init(net);
    init_LW=net.LW;
    init_IW=net.IW;
    init_B=net.B;
    
    [net,tr] = train(net,x_train,t_train);
    
    y_train = net(x_train);
    y_test = net(x_test);
    
    % AUC
    [X1,Y1,T1,AUC1]=perfcurve(t_train,y_train,1);
    roc_train{i,1}=X1; roc_train{i,2}=Y1;
    
    [X,Y,T,AUC] = perfcurve(t_test,y_test,1);
    roc_test{i,1}=X; roc_test{i,2}=Y;
    AUC_cv(i)=AUC;
    %yhat_train = classify(x_train,x_train,t_train)
    % calcparam train
    y_pred_train = double( y_train > 0.5 );
    [C_train, G] = confusionmat(t_train,y_pred_train);
    [accuracy(i,1),precision(i,1),recall(i,1),jaccard(i,1),fscore(i,1)]=calcparam(C_train);
    % calcparam test
    y_pred_test = double ( y_test > 0.5 );
    C_test = confusionmat(t_test,y_pred_test);
    [accuracy(i,2),precision(i,2),recall(i,2),jaccard(i,2),fscore(i,2)]=calcparam(C_test);
    
    tn(i,1)= C_train(1,1);tn(i,2)= C_test(1,1);
    fp(i,1)= C_train(1,2);fp(i,2)= C_test(1,2);
    fn(i,1)= C_train(2,1);fn(i,2)= C_test(2,1);
    tp(i,1)= C_train(2,2);tp(i,2)= C_test(2,2);
   
end
if do_lda && do_pca
fprintf('\nAfter LDA&PCA and %d training, we obtain a mean AUC equal to %.4f\n',k,mean(AUC_cv))
elseif do_lda
    fprintf('\nAfter LDA and %d training, we obtain a mean AUC equal to %.4f\n',k,mean(AUC_cv))
elseif do_pca
    fprintf('\nAfter PCA and %d training, we obtain a mean AUC equal to %.4f\n',k,mean(AUC_cv))
else
    fprintf('\nAfter %d training and no PCA or LDA, we obtain a mean AUC equal to %.4f\n',k,mean(AUC_cv))
end
fprintf('Train\nAccuracy: %.3f \nPrecision: %.3f \nRecall: %.3f \nJaccard Similarity Index: %.3f \nFScore: %.3f ',mean(accuracy(:,1)),mean(precision(:,1)),mean(recall(:,1)),mean(jaccard(:,1)),mean(fscore(:,1)))
fprintf('\nTest\nAccuracy: %.3f \nPrecision: %.3f \nRecall: %.3f \nJaccard Similarity Index: %.3f \nFScore: %.3f ',mean(accuracy(:,2)),mean(precision(:,2)),mean(recall(:,2)),mean(jaccard(:,2)),mean(fscore(:,2)))
C_train_mean=[mean(tn(:,1)),mean(fp(:,1));mean(fn(:,1)),mean(tp(:,1))];
C_test_mean=[mean(tn(:,2)),mean(fp(:,2));mean(fn(:,2)),mean(tp(:,2))];
C_train_mean
C_test_mean
%% ROC CURVES PLOT - TRAIN

%color=['k','b','r','c','m','y','g','w','b+','gx'];
for i=1:k 
    X=roc_train{i,1};
    Y=roc_train{i,2};
    plot(X,Y)
    title('TRAIN ROC')
    refline(1,0)
    hold on
end
hold off
%% ROC CURVES PLOT - TEST
for i=1:k
    X=roc_test{i,1};
    Y=roc_test{i,2};
    plot(X,Y)
    title('TEST ROC')
    refline(1,0)
    hold on
end
hold off


%% -------------------------------FUNZIONI---------------------------------
%----------PCA
function [x_train_new,x_test_new]=pca_fun(x_train,x_test)
var_p = 99;
x_var=var(x_train);
x_mean = mean(x_train);
x_centered = x_train - x_mean;
warning('off')
[coeff,score,latent,tsquared,explained,mu] = pca(x_centered,'Centered',false);
warning('on')
x_train_PCA = x_centered*coeff;
n_comp = numel(explained);
for ii = 1:n_comp
    % Calcolo la somma dei primi "ii" elementi di explained
    sum_ = sum(explained(1:ii));
    if sum_ > var_p, break; end
end
num_pc = ii;

fprintf("\nTo preserve %.2f%s of variance, PCA reduces genes down to %d PC", var_p, "%", num_pc);
x_train_PCA_r = x_train_PCA(:,1:num_pc);

x_test_centered = x_test - x_mean;
x_test_PCA = x_test_centered*coeff;
x_test_PCA_r = x_test_PCA(:,1:num_pc);

% Dataset after PCA
x_train_new = x_train_PCA_r';
x_test_new = x_test_PCA_r';

end

%--------------LDA
function [x_train_new,x_test_new]=lda_fun(x_train,x_test,t_train)

Mdl = fitcdiscr(x_train,t_train);
weigths = abs(Mdl.Coeffs(2,1).Linear);
length(weigths);
[weigths_sorted,indexes] = sort(weigths, 'descend');
feature_out = 30;
weights_selected = weigths_sorted(1:feature_out);
indexes_selected = indexes(1:feature_out);
sum(weights_selected == weigths(indexes_selected));
%numel(indexes_selected)

threshold_weight = 0.0001;
weights_thresholded = weigths(weigths > threshold_weight);
fprintf("\nLDA reduces features down to %d",numel(weights_thresholded))

feature_mask = weigths > threshold_weight;

x_train_reduced_lda = x_train(:, feature_mask);
x_test_reduced_lda = x_test(:,feature_mask);

x_train_new=x_train_reduced_lda;
x_test_new=x_test_reduced_lda;
end

%----------OVER-SAMPLING(SMOTE)
function [x_train_os,t_train_os]=oversampling(x_train,t_train)
% APPLYING OVERSAMPLING ON TRAIN SET 
%controlliamo la distribuzione low/high nel train
nHigh_train=0; nLow_train=0;
labels_low_train = zeros(size(x_train,1),1);
labels_high_train = zeros(size(x_train,1),1);
for i = 1:size(x_train,1)
    if t_train(i) == 0
        nLow_train=nLow_train+1;
        labels_low_train(i) = 1;
    else
        nHigh_train=nHigh_train+1;
        labels_high_train(i) = 1;
    end
end
fprintf('\nTraining set counts %d high risk and %d low risk samples-->\t',nHigh_train,nLow_train)
if abs(nHigh_train-nLow_train)<50
    fprintf('\nTrain set is balanced ---> no over-sampling needed')
    x_train_os=x_train; t_train_os=t_train;
else 
    if nHigh_train>nLow_train
        fprintf('\nTrain set is unbalaced: over-sampling low risk samples')
        [x_train_os,t_train_os]=SMOTE(x_train,labels_low_train);
    else
        fprintf('\nTrain set is unbalaced: over-sampling high risk samples\n')
        [x_train_os,t_train_os]=SMOTE(x_train,labels_high_train);
    end
    nHigh_new=0; nLow_new=0;
    for i=1:length(t_train_os)
        if t_train_os(i)== 1
            nHigh_new=nHigh_new+1;
        else
            nLow_new=nLow_new+1;
        end
    end
    fprintf("\nAfter over-sampling train set:\t %d high risk \t %d low risk",nHigh_new,nLow_new)
end
end
function[accuracy,precision,recall,jaccard,fscore]=calcparam(C)
tn = C(1,1);
fp = C(1,2);
fn = C(2,1);
tp = C(2,2);
accuracy=(tp+tn)/(tp+tn+fp+fn);
precision=tp/(tp+fp);
recall=tp/(tp+fn);
jaccard=tp/(tp+fp+fn);
fscore=(2*tp)/((2*tp)+fn+fp);
%fprintf('\nAccuracy: %.3f \nPrecision: %.3f \nRecall: %.3f \nJaccard Similarity Index: %.3f \nFScore: %.3f ',accuracy,precision,recall,jaccard,fscore)
end

