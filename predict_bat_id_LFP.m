function bat_id_pred = predict_bat_id_LFP(lfp_power,batNums,target_bat_nums,varargin)

pnames = {'n_boot_rep','nCV','minCalls','mdlType','min_boot_thresh'};
dflts  = {1e2,5,10,'glm_fit_log',0.9};
[n_boot_rep,nCV,minCalls,mdlType,min_boot_thresh] = internal.stats.parseArgs(pnames,dflts,varargin{:});

row_k = 1;
varNames = {'acc','bootAcc','batNum','targetBNum','exp_k','ch_k'};
nVar = length(varNames);
maxRows = 1e5;
bat_id_pred_mat = cell(maxRows,nVar);

for bat_k = 1:length(lfp_power)
    
    for exp_k = 1:length(lfp_power{bat_k})
        
        for target_bat_k = 1:length(lfp_power{bat_k}{exp_k})
            
            for ch_k = 1:length(lfp_power{bat_k}{exp_k}{target_bat_k})
                
                current_lfp_power = lfp_power{bat_k}{exp_k}{target_bat_k}(ch_k,:);
                
                if all(cellfun(@(x) size(x,1),current_lfp_power)>minCalls)
                    
                    X = vertcat(current_lfp_power{:});
                    Y = [zeros(size(current_lfp_power{1},1),1); ones(size(current_lfp_power{2},1),1)];
                    
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
                    
                    bat_id_pred_mat{row_k,3} = batNums{bat_k};
                    bat_id_pred_mat{row_k,4} = target_bat_nums{bat_k}{exp_k}{target_bat_k};
                    bat_id_pred_mat{row_k,5} = exp_k;
                    bat_id_pred_mat{row_k,6} = ch_k;
                    row_k = row_k + 1;
                end
            end
        end
    end
end

bat_id_pred_mat = bat_id_pred_mat(1:row_k-1,:);
bat_id_pred = cell2table(bat_id_pred_mat,'VariableNames',varNames);

end