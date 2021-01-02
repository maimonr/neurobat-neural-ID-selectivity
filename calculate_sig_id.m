function [sigIdx,callSigIdx] = calculate_sig_id(bat_id_pred_all,varIdx,varargin)

pnames = {'callAccFlag','correctionType','alpha0'};
dflts  = {false,'BH',0.05};
[callAccFlag,correctionType,alpha0] = internal.stats.parseArgs(pnames,dflts,varargin{:});

all_cells = unique(bat_id_pred_all(:,varIdx), 'rows');
[sigIdx,callSigIdx] = deal(false(1,size(bat_id_pred_all,1)));
for k = 1:size(all_cells,1)
    idx = find(ismember(bat_id_pred_all(:,varIdx),all_cells(k,:)));
    if iscell(bat_id_pred_all.bootAcc)
        p = 1 - cellfun(@(acc,bootAcc) sum(acc > bootAcc)/length(bootAcc),num2cell(bat_id_pred_all.acc(idx)),bat_id_pred_all.bootAcc(idx,:));
    else
        p = 1 - cellfun(@(acc,bootAcc) sum(acc > bootAcc)/length(bootAcc),num2cell(bat_id_pred_all.acc(idx)),num2cell(bat_id_pred_all.bootAcc(idx,:),2));
    end
    sigIdx(idx) = calculate_sig(p,alpha0,correctionType);
    
    if callAccFlag
        usedIdx = cellfun(@(bootAcc) ~all(bootAcc==0),bat_id_pred_all.bootCallAcc(idx,:));
        idx = idx(usedIdx);
        if ~isempty(idx)
            p = 1 - cellfun(@(acc,bootAcc) sum(acc > bootAcc)/length(bootAcc),num2cell(bat_id_pred_all.callAcc(idx)),bat_id_pred_all.bootCallAcc(idx,:));
            callSigIdx(idx) = calculate_sig(p,alpha0,correctionType);
        end
    end
end
end

function sigIdx = calculate_sig(p,alpha0,correctionType)

m = length(p);
switch correctionType
    case 'none'
        sigIdx = p < alpha0;
    case 'BH'
        [pSort,p_sort_idx] = sort(p,'ascend');
        BH_critical_value = alpha0*((1:m)/m)';
        max_sig_idx = find(pSort < BH_critical_value, 1, 'last' );
        
        sigIdx = false(1,m);
        
        current_sig_idx = p_sort_idx(1:max_sig_idx);
        sigIdx(current_sig_idx) = true;
    case 'bonferroni'
        sigIdx = p < alpha0/m;
end

end