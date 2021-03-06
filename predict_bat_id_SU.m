function bat_id_pred = predict_bat_id_SU(vd,varargin)

pnames = {'n_boot_rep','nCV','minCalls','cData','timeWin','winSize','predType',...
    'mdlType','select_cell_table','min_boot_thresh','exclude_by_bhv','callPred',...
    'benchmark_exclusion','call_dist_map','call_dist_mode','call_dist_thresh',...
    'selectBats'};
dflts  = {1e3,5,15,[],[-1 1],0.25,'one_vs_all',...
    'glm_fit_log',[],0.9,[],false,...
    false,[],'none',10,...
    []};
[n_boot_rep,nCV,minCalls,cData,timeWin,winSize,predType,...
    mdlType,select_cell_table,min_boot_thresh,exclude_by_bhv_type,callPred,...
    benchmark_exclusion_flag,call_dist_map,call_dist_mode,call_dist_thresh,...
    selectBats] = internal.stats.parseArgs(pnames,dflts,varargin{:});

[t,t_idx] = inRange(vd.time,timeWin);

sliding_win_idx = slidingWin(length(t) - 2*vd.constantBW/vd.dT,round(winSize/vd.dT),0);
sliding_win_idx = sliding_win_idx + vd.constantBW/vd.dT;

nBat = length(vd.batNums);

varNames = {'acc','callAcc','bootAcc','bootCallAcc','targetBNum','batNum','cell_k','cellInfo'};
nVar = length(varNames);
switch predType
    case 'one_vs_all'
        all_target_bat_nums = [vd.call_bat_num{:}];
        all_target_bat_nums = all_target_bat_nums(~cellfun(@isempty,all_target_bat_nums));
        n_target_bat = length(unique(all_target_bat_nums(~cellfun(@iscell,all_target_bat_nums)))) - 1;
    case 'self_vs_other'
        n_target_bat = 1;
    case 'selectBats'
        n_target_bat = 1;
end
maxRows = vd.nCells*n_target_bat;
bat_id_pred_mat = cell(maxRows,nVar);
row_k = 1;
lastProgress = 0;
ticH = tic;
f = waitbar(0,'0 cells processed - 0s elapsed');
n_total_cell = sum(vd.usable);
all_cell_k = 1;
for bat_k = 1:nBat
    self_bat_num = vd.batNums(bat_k);
    cell_ks = find(strcmp(vd.batNum,self_bat_num) & vd.usable);
    
    nCell = length(cell_ks);
    k = 1;
    for cell_k = cell_ks
        switch predType
            case 'one_vs_all'
                call_bat_nums = vd.call_bat_num{cell_k};
                target_bat_nums = unique(cellflat(call_bat_nums));
                target_bat_nums = setdiff(target_bat_nums,[self_bat_num,'unidentified']);
                n_target_bats = length(target_bat_nums);
                
            case 'self_vs_other'
                target_bat_nums = self_bat_num;
                n_target_bats = 1;
                
            case 'selectBats'
                current_cell_table = table(self_bat_num,vd.cellInfo(cell_k),'VariableNames',{'batNum','cellInfo'});
                current_cell_idx = ismember(selectBats(:,1:2),current_cell_table);
                if ~any(current_cell_idx)
                    continue
                end
                   
                target_bat_nums = selectBats.target_bat_nums(current_cell_idx,:);
                n_target_bats = 1;
        end
        
        call_bat_nums = vd.call_bat_num{cell_k};
        call_bat_nums(cellfun(@iscell,call_bat_nums)) = {'unidentified'};
        
        trialFR = vd.trialFR(cell_k);
        trialFR = trialFR(:,t_idx);
        trialPredictors = nan(size(trialFR,1),size(sliding_win_idx,1));
        for trial_k = 1:size(trialFR,1)
            trial_fr_tmp = trialFR(trial_k,:);
            trialPredictors(trial_k,:) = mean(trial_fr_tmp(sliding_win_idx'));
        end
        
        for target_bat_k = 1:n_target_bats
            target_bat_num = target_bat_nums(target_bat_k);
            
            if ~isempty(select_cell_table)
                if length(select_cell_table.Properties.VariableNames) == 3
                    current_cell_table = table(target_bat_num,self_bat_num,vd.cellInfo(cell_k),'VariableNames',{'targetBNum','batNum','cellInfo'});
                elseif length(select_cell_table.Properties.VariableNames) == 2
                    current_cell_table = table(self_bat_num,vd.cellInfo(cell_k),'VariableNames',{'batNum','cellInfo'});
                end
                if ~ismember(current_cell_table,select_cell_table)
                    continue
                end
            end
            
            targetIdx = vd.usedCalls{cell_k} & ismember(call_bat_nums,target_bat_num{1});
            
            switch predType
                case 'one_vs_all'
                    nontargetIdx = vd.usedCalls{cell_k} & ~strcmp(call_bat_nums,target_bat_num) & ~strcmp(vd.call_bat_num{cell_k},self_bat_num);
                case 'self_vs_other'
                    nontargetIdx = vd.usedCalls{cell_k} & ~strcmp(call_bat_nums,target_bat_num);
                case 'selectBats'
                    nontargetIdx = vd.usedCalls{cell_k} & ismember(call_bat_nums,target_bat_nums{2});
            end
            
            if ~(sum(targetIdx) > minCalls && sum(nontargetIdx) > minCalls)
                continue
            end
            
            if ~isempty(exclude_by_bhv_type)
                target_idx_orig = targetIdx;
                nontarget_idx_orig = nontargetIdx;
                [targetIdx,nontargetIdx] = exclude_by_bhv(vd,cData,cell_k,target_idx_orig,nontarget_idx_orig,exclude_by_bhv_type);
            end
            
            switch call_dist_mode
                case 'exclude_by_dist'
                    target_idx_orig = targetIdx;
                    nontarget_idx_orig = nontargetIdx;
                    [targetIdx,nontargetIdx] = exclude_by_dist(vd,cell_k,target_idx_orig,nontarget_idx_orig,call_dist_map,call_dist_thresh);
                case 'use_dist_pred'
                    callDist = get_call_dist(vd,cell_k,call_dist_map);
                    
                    used_call_dist_idx = ~isnan(callDist);
                    targetIdx = targetIdx & used_call_dist_idx;
                    nontargetIdx = nontargetIdx & used_call_dist_idx;
                    trialPredictors = [trialPredictors callDist'];
            end
            
            if ~(sum(targetIdx) > minCalls && sum(nontargetIdx) > minCalls)
                continue
            end
            
            X = [trialPredictors(targetIdx,:); trialPredictors(nontargetIdx,:)];
            Y = [zeros(sum(targetIdx),1); ones(sum(nontargetIdx),1)];
            
            if callPred
                [callX, callY] = init_call_pred(vd,cData,cell_k,timeWin);
            else
                [callX,callY] = deal([]);
            end
            
            cvAcc = get_cv_id_acc(X,Y,nCV,'mdlType',mdlType);
            bat_id_pred_mat{row_k,1} = mean(cvAcc);
            
            if benchmark_exclusion_flag
                bootAcc = get_benchmark_acc(trialPredictors,n_boot_rep,nCV,mdlType,target_idx_orig,nontarget_idx_orig,targetIdx,nontargetIdx);
                bootCallAcc = [];
            else
                [bootAcc,bootCallAcc] = get_boot_acc(X,Y,bat_id_pred_mat{row_k,1},n_boot_rep,nCV,mdlType,callPred,min_boot_thresh);
            end
            
            if callPred
                cvCallAcc = get_cv_id_acc(callX,callY,nCV,'mdlType',mdlType);
                bat_id_pred_mat{row_k,2} = mean(cvCallAcc);
            end
            
            bat_id_pred_mat{row_k,3} = bootAcc;
            bat_id_pred_mat{row_k,4} = bootCallAcc;
            
            bat_id_pred_mat{row_k,5} = target_bat_num{1};
            bat_id_pred_mat{row_k,6} = self_bat_num{1};
            bat_id_pred_mat{row_k,7} = cell_k;
            bat_id_pred_mat{row_k,8} = vd.cellInfo{cell_k};
            row_k = row_k + 1;
        end
        
        progress = round(10*(k/nCell))*10;
        if mod(progress,10) < mod(lastProgress,10)
            fprintf('%d %% of cells from bat %d / %d\n',progress,bat_k,nBat)
            toc
        end
        lastProgress = progress;
        k = k + 1;
        all_cell_k = all_cell_k + 1;
        f = waitbar(all_cell_k/n_total_cell,f,sprintf('%d cells processed - %ds elapsed',all_cell_k,toc(ticH)));
    end
end

bat_id_pred_mat = bat_id_pred_mat(1:row_k-1,:);
bat_id_pred = cell2table(bat_id_pred_mat,'VariableNames',varNames);
close(f)
end

function [targetIdx,nontargetIdx] = exclude_by_bhv(vd,cData,cell_k,targetIdx,nontargetIdx,exclude_by_bhv_type)

all_call_info = get_all_bhv(cData);
expDate = vd.expDay(cell_k);

all_call_info = all_call_info([all_call_info.expDate] == expDate);
if isempty(all_call_info)
    [targetIdx,nontargetIdx] = deal(false(1,length(targetIdx)));
    return
end

n_bhv_bouts = length(all_call_info);
bhv_bout_call_nums = cell(1,n_bhv_bouts);

for k = 1:n_bhv_bouts
    if ~isnan(all_call_info(k).callID)
        bhv_bout_call_nums{k} = get_call_bout_nums(cData,all_call_info(k).callID,1);
    else
        bhv_bout_call_nums{k} = NaN;
    end
end

switch exclude_by_bhv_type
    
    case 'exclude_self'
        
        self_bat_num = vd.batNum(cell_k);
        included_interaction_idx = cellfun(@(bats) ~isempty(bats) && ~any(strcmp(bats,self_bat_num)),{all_call_info.batsInvolved});
        
    case 'exclude_spread'
        
        included_interaction_idx = cellfun(@(bhv) ~isempty(bhv) && contains(bhv,'Grouped','IgnoreCase',true),{all_call_info.behaviors});
        
end

included_bout_call_nums = vertcat(bhv_bout_call_nums{included_interaction_idx});
callNums = vd.callNum{cell_k};
included_bout_idx = ismember(callNums,included_bout_call_nums);

targetIdx = targetIdx & included_bout_idx;
nontargetIdx = nontargetIdx & included_bout_idx;


end

function [targetIdx,nontargetIdx] = exclude_by_dist(vd,cell_k,targetIdx,nontargetIdx,call_dist_map,call_dist_thresh)

callDist = get_call_dist(vd,cell_k,call_dist_map);

call_dist_idx = sign(call_dist_thresh)*callDist < call_dist_thresh;

targetIdx = targetIdx & call_dist_idx;
nontargetIdx = nontargetIdx & call_dist_idx;
end

function callDist = get_call_dist(vd,cell_k,call_dist_map)
callIDs = vd.callNum{cell_k};
calling_bat_nums = vd.call_bat_num{cell_k};
self_bat_num = vd.batNum(cell_k);
callDist = nan(1,length(callIDs));
call_k = 1;
for callID = callIDs
    if isKey(call_dist_map,callID) && ~iscell(calling_bat_nums{call_k})
        bat_pair_str = strjoin(sort([calling_bat_nums(call_k),self_bat_num]),'-');
        current_call_dist = call_dist_map(callID);
        if isKey(current_call_dist,bat_pair_str)
            callDist(call_k) = current_call_dist(bat_pair_str);
        end
    end
    call_k = call_k + 1;
end
end

function [callX, callY] = init_call_pred(vd,cData,cell_k,timeWin)

callNums = vd.callNum{cell_k};
callNums_self = callNums(targetIdx);
callNums_other = callNums(nontargetIdx);
predMat = cell(1,2);
target_k = 1;
for cNums = {callNums_self,callNums_other}
    call_bout_nums = cell(1,length(cNums{1}));
    for call_k = 1:length(cNums{1})
        try
            call_bout_nums{call_k} = get_call_bout_nums(cData,cNums{1}(call_k),max(timeWin));
        catch
            call_bout_nums{call_k} = [];
        end
    end
    call_bout_nums = vertcat(call_bout_nums{:});
    idx = ismember(cData.callID,call_bout_nums);
    F0 = cData.yinF0(idx);
    WE = cData.weinerEntropy(idx);
    RMS = cData.RMS(idx);
    callLength = cData.callLength(idx);
    SE = cData.spectralEntropy(idx);
    centroid = cData.centroid(idx);
    ap0 = cData.ap0(idx);
    
    predMat{target_k} = [F0 WE RMS callLength SE centroid ap0];
    target_k = target_k + 1;
end

callX = vertcat(predMat{:});
callY = cellfun(@(predMat,label) label*ones(size(predMat,1),1),predMat,{0,1},'un',0);
callY = vertcat(callY{:});

end

function [bootAcc,bootCallAcc] = get_boot_acc(X,Y,acc,n_boot_rep,nCV,mdlType,callPred,min_boot_thresh)

calculate_call_pred_boot = false;
for boot_thresh_k = 1:length(n_boot_rep)
    [bootAcc,bootCallAcc] = deal(zeros(1,n_boot_rep(boot_thresh_k)));
    parfor boot_k = 1:n_boot_rep(boot_thresh_k)
        label_perm_idx = randperm(length(Y));
        Y_perm = Y(label_perm_idx);
        cvAcc = get_cv_id_acc(X,Y_perm,nCV,'mdlType',mdlType);
        bootAcc(boot_k) = mean(cvAcc);
        
        if callPred && calculate_call_pred_boot
            label_perm_idx = randperm(length(callY));
            Y_perm = callY(label_perm_idx);
            cvCallAcc = get_cv_id_acc(callX,Y_perm,nCV,'mdlType',mdlType);
            bootCallAcc(boot_k) = mean(cvCallAcc);
        end
    end
    p = sum(acc > bootAcc)/n_boot_rep(boot_thresh_k);
    
    if p < min_boot_thresh
        break
    else
        calculate_call_pred_boot = true;
    end
end

end

function bootAcc = get_benchmark_acc(trial_fr_binned,n_boot_rep,nCV,mdlType,target_idx_orig,nontarget_idx_orig,targetIdx,nontargetIdx)

bootAcc = zeros(1,n_boot_rep(1));

n_target_incl = sum(targetIdx);
n_nontarget_incl = sum(nontargetIdx);

n_target_orig = sum(target_idx_orig);
n_nontarget_orig = sum(nontarget_idx_orig);

target_idx_orig = find(target_idx_orig);
nontarget_idx_orig = find(nontarget_idx_orig);

parfor boot_k = 1:n_boot_rep(1)
    
    targetIdx = target_idx_orig(randperm(n_target_orig,n_target_incl));
    nontargetIdx = nontarget_idx_orig(randperm(n_nontarget_orig,n_nontarget_incl));
    
    X = [trial_fr_binned(targetIdx,:); trial_fr_binned(nontargetIdx,:)];
    Y = [zeros(length(targetIdx),1); ones(length(nontargetIdx),1)];
    cvAcc = get_cv_id_acc(X,Y,nCV,'mdlType',mdlType);
    bootAcc(boot_k) = mean(cvAcc);
end

end

function [xSub,idx] = inRange(x,bounds)
% function to get values and index of  vector within a given bounds
bounds = sort(bounds);
idx = x>bounds(1) & x<= bounds(2);
xSub = x(idx);
end

function idx = slidingWin(N, winSize, overlap)

nWins = floor((N)/(winSize - overlap));
idx = zeros(nWins,winSize);

for w = 1:nWins
   idx(w,:) = (winSize-overlap)*(w-1) +( 1:winSize); 
end

badWins = find(idx>N);
[I,~] = ind2sub(size(idx),badWins);
idx(unique(I),:) = [];


end
