function acc = get_cv_id_acc(X,Y,nCV,varargin)

pnames = {'mdlType','lambda'};
dflts  = {'glm_fit_log',NaN};
[mdlType,lambda] = internal.stats.parseArgs(pnames,dflts,varargin{:});

nCall = length(Y);

rand_call_idx = randperm(nCall);
cvBreaks = 1:round(nCall/nCV):nCall;
if length(cvBreaks) < nCV + 1
    cvBreaks(end+1) = nCall;
else
    cvBreaks(end) = nCall;
end

acc = nan(1,nCV);

for cv_k = 1:nCV
    if cv_k < nCV
        cvFold = rand_call_idx(cvBreaks(cv_k):cvBreaks(cv_k+1)-1);
    else
        cvFold = rand_call_idx(cvBreaks(cv_k):cvBreaks(cv_k+1));
    end
    
    trainIdx = setdiff(1:nCall,cvFold);
    testIdx = cvFold;
    
    switch mdlType
        
        case 'glm_fit_log'
            
            currentY = Y(trainIdx);
            weights = ones(length(currentY),1);
            for k = 0:1
                weights(currentY==k) = (weights(currentY==k)/sum(weights(currentY==k)))/2;
            end
            
            if isnan(lambda)
                b = glmfit(X(trainIdx,:),Y(trainIdx,:),'binomial','link','logit','weights',weights);
            else
                try
                    [b,fitInfo] = lassoglm(X(trainIdx,:),Y(trainIdx,:),'binomial','link','logit','weights',weights,'Lambda',0.01);
                    b = [fitInfo.Intercept;b]; %#ok<AGROW>
                catch err
                    switch err.identifier
                        case 'stats:lasso:MaxIterReached'
                            b = glmfit(X(trainIdx,:),Y(trainIdx,:),'binomial','link','logit','weights',weights);
                        otherwise
                            rethrow(err)
                    end
                end
            end
            pp = glmval(b,X(testIdx,:),'logit');
            yy = round(pp);
            
        case 'svm'
            
            mdl = fitcsvm(X(trainIdx,:),Y(trainIdx,:),'Prior','uniform');
            yy = predict(mdl,X(testIdx,:));
            
        case 'fitc_log'
            
            mdl = fitclinear(X(trainIdx,:),Y(trainIdx,:),'Learner','logistic','Prior','uniform');
            yy = predict(mdl,X(testIdx,:));
            
        case 'fitc_discr'
            
            mdl = fitcdiscr(X(trainIdx,:),Y(trainIdx,:),'Prior','uniform','DiscrimType','diaglinear');
            yy = predict(mdl,X(testIdx,:));
            
    end
    
    
    acc(cv_k) = sum(Y(testIdx) == yy)/length(testIdx);
    
end

end