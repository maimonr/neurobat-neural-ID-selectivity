function bat_id_pred = predict_bat_id_multi_SU(vd,varargin)

warning('error','stats:lasso:MaxIterReached') %#ok<CTPCT>
cleanObj = onCleanup(@cleanupFun);

pnames = {'n_boot_rep','nCV','minCalls','timeWin','winSize','mdlType','min_boot_thresh','lambda'};
dflts  = {1e2,5,10,[-1 1],0.25,'glm_fit_log',0.9,NaN};
[n_boot_rep,nCV,minCalls,timeWin,winSize,mdlType,min_boot_thresh,lambda] = internal.stats.parseArgs(pnames,dflts,varargin{:});
[t,t_idx] = inRange(vd.time,timeWin);

sliding_win_idx = slidingWin(length(t) - 2*vd.constantBW/vd.dT,round(winSize/vd.dT),0);
sliding_win_idx = sliding_win_idx + vd.constantBW/vd.dT;

nBat = length(vd.batNums);

varNames = {'acc','callAcc','bootAcc','bootCallAcc','targetBNum','batNum','cell_k','cellInfo'};
nVar = length(varNames);

all_target_bat_nums = [vd.call_bat_num{:}];
n_target_bat = length(unique(all_target_bat_nums(~cellfun(@iscell,all_target_bat_nums)))) - 1;
maxRows = length(unique(vd.expDay))*nBat*n_target_bat;
bat_id_pred_mat = cell(maxRows,nVar);
row_k = 1;
lastProgress = 0;
tic
for bat_k = 1:nBat
    self_bat_num = vd.batNums(bat_k);
    batIdx = strcmp(vd.batNum,self_bat_num) & vd.usable;
    expDays = unique(vd.expDay(batIdx));
    nExp = length(expDays);
    
    progress_counter_k = 1;
    for exp_k = 1:nExp
        
        cell_ks = find(batIdx & vd.expDay == expDays(exp_k));
        
        nCalls = cellfun(@sum,vd.usedCalls(cell_ks));
        used_cell_idx = nCalls == mode(nCalls);
        
        cell_ks = cell_ks(used_cell_idx);
        nCell = length(cell_ks);
        
        all_call_nums = unique([vd.callNum{cell_ks}]);
        assert(all(cellfun(@(x) isempty(setxor(all_call_nums,x)),vd.callNum(cell_ks))))
        
        single_cell_k = cell_ks(1);
        call_bat_nums = vd.call_bat_num{single_cell_k};
        target_bat_nums = unique(cellflat(call_bat_nums));
        target_bat_nums = setdiff(target_bat_nums,[self_bat_num,'unidentified']);
                
        n_target_bats = length(target_bat_nums);
        
        trial_fr_binned = cell(1,nCell);
        k = 1;
        for cell_k = cell_ks
            trialFR = vd.trialFR(cell_k);
            trialFR = trialFR(:,t_idx);
            trial_fr_binned{k} = nan(size(trialFR,1),size(sliding_win_idx,1));
            
            for trial_k = 1:size(trialFR,1)
                trial_fr_tmp = trialFR(trial_k,:);
                trial_fr_binned{k}(trial_k,:) = mean(trial_fr_tmp(sliding_win_idx'));
            end
            k = k + 1;
        end
        trial_fr_binned = [trial_fr_binned{:}];
        
        for target_bat_k = 1:n_target_bats
            target_bat_num = target_bat_nums(target_bat_k);
            
            targetIdx = vd.usedCalls{single_cell_k} & strcmp(vd.call_bat_num{single_cell_k},target_bat_num)';
            nontargetIdx = vd.usedCalls{single_cell_k} & ~strcmp(vd.call_bat_num{single_cell_k},target_bat_num)';
            
            if sum(targetIdx) > minCalls && sum(nontargetIdx) > minCalls
                
                X = [trial_fr_binned(targetIdx,:); trial_fr_binned(nontargetIdx,:)];
                Y = [zeros(sum(targetIdx),1); ones(sum(nontargetIdx),1)];
                
                cvAcc = get_cv_id_acc(X,Y,nCV,'mdlType',mdlType,'lambda',lambda);
                bat_id_pred_mat{row_k,1} = mean(cvAcc);
                
                for boot_thresh_k = 1:length(n_boot_rep)
                    bootAccTmp = deal(zeros(1,n_boot_rep(boot_thresh_k)));
                    parfor boot_k = 1:n_boot_rep(boot_thresh_k)
                        label_perm_idx = randperm(length(Y));
                        Y_perm = Y(label_perm_idx);
                        cvAcc = get_cv_id_acc(X,Y_perm,nCV,'mdlType',mdlType,'lambda',lambda);
                        bootAccTmp(boot_k) = mean(cvAcc);
                    end
                    p = sum(bat_id_pred_mat{row_k,1} > bootAccTmp)/n_boot_rep(boot_thresh_k);
                    
                    if p < min_boot_thresh
                        break
                    end
                end
                bat_id_pred_mat{row_k,3} = bootAccTmp;
                
                bat_id_pred_mat{row_k,4} = target_bat_num{1};
                bat_id_pred_mat{row_k,5} = self_bat_num{1};
                bat_id_pred_mat{row_k,6} = cell_ks;
                bat_id_pred_mat{row_k,7} = vd.cellInfo(cell_ks);
                row_k = row_k + 1;
            end
        end
        progress = round(10*(progress_counter_k/nCell))*10;
        if mod(progress,10) < mod(lastProgress,10)
            fprintf('%d %% of cells from bat %d / %d\n',progress,bat_k,nBat)
            toc
        end
        lastProgress = progress;
        progress_counter_k = progress_counter_k + 1;
        
    end
end

bat_id_pred_mat = bat_id_pred_mat(1:row_k-1,:);
bat_id_pred = cell2table(bat_id_pred_mat,'VariableNames',varNames);

end

function cleanupFun
warning('on','stats:lasso:MaxIterReached')
end
