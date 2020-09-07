function prediction_accuracy = predict_bat_id_from_fr(activation,varargin)

pnames = {'n_acc_rep','n_population_reps','minCalls','trainFrac','exclude_cell_frac','cell_ks','exclude_cell_ks','exclude_n_resp'};
dflts  = {1e2,1e2,20,0.8,NaN,[],[],false};
[n_acc_rep,n_population_reps,minCalls,trainFrac,exclude_cell_frac,cell_ks,exclude_cell_ks,exclude_n_resp_flag] = internal.stats.parseArgs(pnames,dflts,varargin{:});

nBat = length(activation);

if length(exclude_cell_frac) == 1
   exclude_cell_frac = repmat(exclude_cell_frac,1,nBat); 
end

prediction_accuracy = zeros(nBat,n_acc_rep);

for bat_k = 1:nBat
    current_activation = activation{bat_k};
    n_pred_cats = size(current_activation,2);
    n_time_bins = unique(cellfun(@(x) size(x,2),current_activation));
    if ~isempty(exclude_cell_ks) && ~exclude_n_resp_flag
       current_activation = current_activation(~ismember(cell_ks,exclude_cell_ks),:); 
    elseif ~isempty(exclude_cell_ks) && exclude_n_resp_flag
        exclude_cell_frac(bat_k) = sum(ismember(cell_ks,exclude_cell_ks))/size(current_activation,1);
    end
    for acc_k = 1:n_acc_rep
        
        if ~isnan(exclude_cell_frac(bat_k))
            current_activation = activation{bat_k};
            nCell = size(current_activation,1);
            subsetIdx = randperm(nCell,nCell - round(nCell*exclude_cell_frac(bat_k)));
            current_activation = current_activation(subsetIdx,:);
        end
        
        usable_idx = all(cellfun(@(x) size(x,1),current_activation)>minCalls,2);
        current_activation = current_activation(usable_idx,:);
        nCell = sum(usable_idx);
        
        
        trainIdx = cellfun(@(x) randperm(size(x,1),floor(size(x,1)*trainFrac)),current_activation,'un',0);
        testIdx = cellfun(@(x,y) setdiff(1:size(x,1),y),current_activation,trainIdx,'un',0);
        
        activation_split = cell(1,2);
        
        activation_split{1} = cellfun(@(fr,idx) fr(idx,:),current_activation,trainIdx,'un',0);
        activation_split{2} = cellfun(@(fr,idx) fr(idx,:),current_activation,testIdx,'un',0);
        
        population_activity_split = repmat({nan(n_pred_cats*n_population_reps,nCell*n_time_bins)},1,2);
        for rep_k = 1:n_population_reps
            for split_k = 1:2
                current_activity = nan(n_pred_cats,nCell*n_time_bins);
                for cell_k = 1:nCell
                    for pred_k = 1:n_pred_cats
                        activationTmp = activation_split{split_k}{cell_k,pred_k};
                        idx = randi(size(activationTmp,1),1);
                        for time_bin_k = 1:n_time_bins
                            time_bin_idx = (cell_k-1)*n_time_bins + time_bin_k;
                            current_activity(pred_k,time_bin_idx) = activationTmp(idx,time_bin_k);
                        end
                    end
                end
                rep_idx = rep_k*n_pred_cats - (n_pred_cats - 1);
                population_activity_split{split_k}(rep_idx:rep_idx+n_pred_cats-1,:) = current_activity;
            end
        end
        population_activity_split = cellfun(@(activity) zscore(activity,[],1),population_activity_split,'un',0);
        bhvType = mod(1:n_pred_cats*n_population_reps,n_pred_cats);
        if n_pred_cats == 2
            mdl = fitclinear(population_activity_split{1},bhvType,'Learner','svm','Regularization','ridge');
        else
            mdl = fitcecoc(population_activity_split{1},bhvType,'Learners','svm');
%             mdl = fitcdiscr(population_activity_split{1},bhvType,'DiscrimType','pseudolinear');
        end
        [~,betaIdx(acc_k)] = max(abs(mdl.Beta));
        pred_bhv_type = predict(mdl,population_activity_split{2});
        prediction_accuracy(bat_k,acc_k) = sum(pred_bhv_type == bhvType')/length(bhvType);
        
    end
end