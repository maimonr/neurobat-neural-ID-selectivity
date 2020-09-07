function bat_id_pred = predict_bat_id_corr(cross_brain_corr,all_bat_pairs,included_bat_nums,expDays,t,varargin)

pnames = {'n_boot_rep','nCV','minCalls','mdlType','min_boot_thresh','timeWin'};
dflts  = {1e3,5,10,'glm_fit_log',0.9,[-1 1]};
[n_boot_rep,nCV,minCalls,mdlType,min_boot_thresh,timeWin] = internal.stats.parseArgs(pnames,dflts,varargin{:});

[~,t_idx] = inRange(t,timeWin);
cross_brain_corr = cross_brain_corr(:,:,t_idx);

row_k = 1;
varNames = {'acc','bootAcc','batPair','targetBNum','exp_k'};
nVar = length(varNames);
maxRows = 1e5;
bat_id_pred_mat = cell(maxRows,nVar);

all_used_call_idx = ~all(isnan(cross_brain_corr),3);

n_bat_pairs = size(all_bat_pairs,1);
all_exp_days = unique(expDays);
n_exp_day = length(all_exp_days);


for exp_k = 1:n_exp_day
    exp_day_idx = expDays == all_exp_days(exp_k);
    target_bat_nums = unique(cellflat(included_bat_nums(exp_day_idx)));
    for pair_k = 1:n_bat_pairs        
        target_bat_nums = setdiff(target_bat_nums,[all_bat_pairs(pair_k,:), 'unidentified']);
        n_target_bats = length(target_bat_nums);
        for target_bat_k = 1:n_target_bats
            target_bat_num = target_bat_nums(target_bat_k);
            used_call_idx = all_used_call_idx(:,pair_k) & exp_day_idx;
            
            targetIdx = strcmp(included_bat_nums(used_call_idx),target_bat_num)';
            nontargetIdx = ~strcmp(included_bat_nums(used_call_idx),target_bat_num)' & ~contains(included_bat_nums(used_call_idx),all_bat_pairs(pair_k,:))';

            if sum(targetIdx) > minCalls && sum(nontargetIdx) > minCalls
                
                X = squeeze(cross_brain_corr(used_call_idx,pair_k,:));
                Y = nan(sum(used_call_idx),1);
                Y(targetIdx) = 1; Y(nontargetIdx) = 0;
                
                cvAcc = get_cv_id_acc(X,Y,nCV,'mdlType',mdlType);
                bat_id_pred_mat{row_k,1} = mean(cvAcc);
                
                for boot_thresh_k = 1:length(n_boot_rep)
                    bootAccTmp = zeros(1,n_boot_rep(boot_thresh_k));
                    parfor boot_k = 1:n_boot_rep(boot_thresh_k)
                        label_perm_idx = randperm(length(Y));
                        Y_perm = Y(label_perm_idx);
                        cvAcc = get_cv_id_acc(X,Y_perm,nCV,'mdlType',mdlType);
                        bootAccTmp(boot_k) = mean(cvAcc);
                    end
                    p = sum(bat_id_pred_mat{row_k,1} > bootAccTmp)/n_boot_rep(boot_thresh_k);
                    
                    if p < min_boot_thresh
                        break
                    end
                end
                
                bat_id_pred_mat{row_k,2} = bootAccTmp;
                
                bat_id_pred_mat{row_k,3} = strjoin(all_bat_pairs(pair_k,:));
                bat_id_pred_mat{row_k,4} = target_bat_num;
                bat_id_pred_mat{row_k,5} = all_exp_days(exp_k);
                row_k = row_k + 1;
            end
        end
    end
end

bat_id_pred_mat = bat_id_pred_mat(1:row_k-1,:);
bat_id_pred = cell2table(bat_id_pred_mat,'VariableNames',varNames);

end