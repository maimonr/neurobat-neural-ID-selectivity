function [acc,IDacc] = get_pos_data_decoding_acc(posData,cData,bat_id_pred,pos_data_type)

switch pos_data_type
    case 'dist'
        callMap = get_call_dist(posData,cData,'inter_call_int',0.5);
    case 'pos'
        callMap = get_call_pos(posData,cData,'inter_call_int',0.5);
end
callMap = posData.collect_by_calls(callMap);

%%
callNums = callMap.keys;
call_bat_nums = cellfun(@(cNum) cData('callID',cNum).batNum,callNums,'un',0);
call_bat_nums = cellfun(@(x) x{1},call_bat_nums,'un',0);

idx = ~cellfun(@iscell,call_bat_nums);
callNums = callNums(idx);
call_bat_nums = call_bat_nums(idx);

call_exp_dates = cellfun(@(cNum) cData('callID',cNum).expDay,callNums);

%%
exclDate = datetime(2020,8,3);
sigIdx = calculate_sig_id(bat_id_pred,6:8,'correctionType','BH');
expDates = cellfun(@(s) datetime(s(1:8),'InputFormat','yyyyMMdd'),bat_id_pred.cellInfo);
used_date_idx = expDates > datetime(2020,7,28) & expDates ~= exclDate;
sigIdx = sigIdx & used_date_idx';
acc = nan(1,sum(sigIdx));
k = 1;
for cell_k = find(sigIdx)
    expDate = expDates(cell_k);
    target_bat_num = bat_id_pred.targetBNum{cell_k};
    current_bat_num = bat_id_pred.batNum{cell_k};
    bat_pair_key = posData.get_pair_keys(str2double({target_bat_num current_bat_num}));
    
    dateIdx = find(call_exp_dates == expDate);
    switch pos_data_type
        case 'dist'
            call_map_key = bat_pair_key{1};
            d = nan(length(dateIdx),1);
        case 'pos'
            call_map_key = str2double(target_bat_num);
            d = nan(length(dateIdx),2);
    end
    for date_k = 1:length(dateIdx)
        current_call_map = callMap(callNums{dateIdx(date_k)});
        d(date_k,:) = current_call_map(call_map_key);
    end
    Y = strcmp(call_bat_nums(dateIdx),target_bat_num)';
    nanIdx = ~any(isnan(d),2);
    if sum(nanIdx) < 20
        continue
    end
    mdl_log = fitclinear(d(nanIdx,:),Y(nanIdx),'Learner','logistic','Prior','uniform','KFold',5);
    acc(k) = 1 - kfoldLoss(mdl_log);
    k = k + 1;
end

IDacc = bat_id_pred.acc(sigIdx);

end
