function [all_lfp_power,all_target_bat_nums] = prepare_lfp_power_for_prediction(baseDir,batNums,cData,varargin)

pnames = {'f_bin','predictionType','used_call_type','callSeparation','minICI','timeLims','winSize'};
dflts  = {[70 150],'one_vs_all','ICI_inter_bat',0.1,0.1,[-0.1 0.1],0.1};
[f_bin,predictionType,used_call_type,callSeparation,minICI,timeLims,winSize] = internal.stats.parseArgs(pnames,dflts,varargin{:});

n_time_bin = floor(diff(timeLims)/winSize);

fName_str = '*call_trig_ps_corr.mat';

call_trig_lfp_fnames = dir(fullfile(baseDir,fName_str));
nExp = length(call_trig_lfp_fnames);

max_artifact_frac = 0.01;
nBat = length(batNums);

[all_lfp_power,all_target_bat_nums] = deal(cell(1,nBat));
[all_lfp_power{:}] = deal(cell(1,nExp));
[all_target_bat_nums{:}] = deal(cell(1,nExp));

t1 = tic;
for date_k = 1:nExp
    
    lfpData = load(fullfile(call_trig_lfp_fnames(date_k).folder,call_trig_lfp_fnames(date_k).name),'ps','expParams','specParams','n_call_artifact_times');
    if ~isfield(lfpData.expParams,'included_bat_nums')
        continue
    end
    
    exp_bat_nums = [lfpData.expParams.batNums{:}];

    artifact_removed_ps = prepare_ps_data_for_corr(lfpData,max_artifact_frac);
    [lfp_power,used_trial_artifact_idx] = get_f_bin_lfp_power(artifact_removed_ps,lfpData.specParams.freqs,f_bin);
    
    expDay = regexp(call_trig_lfp_fnames(date_k).name,'\d{8}','match');
    expDay = datetime(expDay{1},'InputFormat','yyyyMMdd');
    callIDs = unique([lfpData.expParams.included_call_IDs{:}]','stable');
    call_bat_nums = cData('callID',callIDs,'expDay',expDay).batNum;
    idx = cellfun(@iscell,call_bat_nums);
    session_target_bat_nums = unique([call_bat_nums(~idx); [call_bat_nums{idx}]']);
    ps_dT = mean(diff(lfpData.expParams.ps_time));
    usedChannels = squeeze(~all(isnan(lfp_power),[2 4]));

    for bat_k = 1:nBat
        self_bat_num = batNums{bat_k};
        switch predictionType
            case 'one_vs_all'
                target_bat_nums = setdiff(session_target_bat_nums,{self_bat_num,'unidentified'});
                n_target_bat_nums = length(target_bat_nums);
            case 'prod_vs_percep'
                target_bat_nums = batNums(bat_k);
                n_target_bat_nums = 1;
        end
        
        [all_lfp_power{bat_k}{date_k},all_target_bat_nums{bat_k}{date_k}] = deal(cell(1,n_target_bat_nums));
        
        if ismember(batNums{bat_k},exp_bat_nums)
            batIdx = strcmp(batNums{bat_k},exp_bat_nums);
            nChannel = sum(usedChannels(batIdx,:));
            channelIdx = find(usedChannels(batIdx,:));
            for pred_cat_k = 1:n_target_bat_nums
                all_target_bat_nums{bat_k}{date_k}{pred_cat_k} = target_bat_nums(pred_cat_k);
                all_lfp_power{bat_k}{date_k}{pred_cat_k} = cell(nChannel,2);
                used_call_idx = get_used_calls(used_call_type,callIDs,cData,expDay,callSeparation,minICI,target_bat_nums(pred_cat_k),batNums(bat_k));
                
                if strcmp(predictionType,'one_vs_all')
                    used_call_idx = used_call_idx & ~cellfun(@(bNum) any(strcmp(bNum,batNums{bat_k})),call_bat_nums');
                end
                
                used_call_idx = used_call_idx & ~strcmp(call_bat_nums','unidentified');
                
                used_call_IDs = callIDs(used_call_idx);
                used_call_bat_nums = call_bat_nums(used_call_idx);
                nCall = sum(used_call_idx);                
                
                call_pred_idx = nan(1,nCall);
                trial_binned_lfp_power = nan(nChannel,nCall,n_time_bin);
                
                for call_k = 1:length(used_call_IDs)
                    trialIdx = cellfun(@(trial_call_IDs) ismember(used_call_IDs(call_k),trial_call_IDs),lfpData.expParams.included_call_IDs);
                    if sum(trialIdx) ~= 1
                        continue
                    end
                    lead_call_pos = cData('callID',lfpData.expParams.used_call_IDs(trialIdx),'expDay',expDay).callPos;
                    callPos = cData('callID',used_call_IDs(call_k),'expDay',expDay).callPos;
                    call_dT = callPos(1) - lead_call_pos(1);
                    call_t = lfpData.expParams.ps_time - call_dT;
                    [~,t_idx] = inRange(call_t,timeLims);
                    slidingWinIdx = slidingWin(sum(t_idx),round(winSize/ps_dT),0);
                    
                    if size(slidingWinIdx,1) < n_time_bin
                        continue
                    elseif size(slidingWinIdx,1) > n_time_bin
                        slidingWinIdx = slidingWinIdx(1:n_time_bin,:);
                    end
                    
                    for ch_k = channelIdx
                        if used_trial_artifact_idx(batIdx,trialIdx,ch_k)
                            current_lfp_power = lfp_power(batIdx,trialIdx,ch_k,t_idx);
                            trial_binned_lfp_power(ch_k,call_k,:) = mean(current_lfp_power(slidingWinIdx'));
                        end
                    end
                    
                    switch predictionType
                        case 'one_vs_all'
                            call_pred_idx(call_k) = any(strcmp(target_bat_nums(pred_cat_k),used_call_bat_nums{call_k}));
                        case 'prod_vs_percep'
                            call_pred_idx(call_k) = any(strcmp(batNums(bat_k),used_call_bat_nums{call_k}));
                    end
                end
                
                artifact_idx = all(isnan(trial_binned_lfp_power),[1 3]);
                call_pred_idx(artifact_idx) = NaN;
                
                for ch_k = channelIdx
                    for pred_type_k = 1:2
                        if pred_type_k == 1
                            pred_lfp_power = squeeze(trial_binned_lfp_power(ch_k,call_pred_idx == 1,:));
                        else
                            pred_lfp_power = squeeze(trial_binned_lfp_power(ch_k,call_pred_idx == 0,:));
                        end
                        all_lfp_power{bat_k}{date_k}{pred_cat_k}{ch_k,pred_type_k} = pred_lfp_power;
                    end
                end
                
            end
        end
    end
    toc(t1)
end

end

function artifact_removed_ps = prepare_ps_data_for_corr(lfpData,max_artifact_frac)

lfp_call_time_s = abs(diff(lfpData.expParams.call_t_win));
call_t_length = lfpData.expParams.fs*lfp_call_time_s;
max_n_artifact = call_t_length*max_artifact_frac;
artifact_trial_idx = any(lfpData.n_call_artifact_times>max_n_artifact,3);
artifact_removed_ps = lfpData.ps;

for bat_k = 1:size(lfpData.ps,1)
    artifact_removed_ps(bat_k,artifact_trial_idx(bat_k,:),:,:,:) = NaN;
end

end

function [all_lfp_power,used_trial_idx] = get_f_bin_lfp_power(ps,freqs,f_bin)

[~,f_idx] = inRange(freqs,f_bin);

freq_band_ps = ps(:,:,:,:,f_idx);
mu = nanmean(freq_band_ps,4);
sigma = nanstd(freq_band_ps,[],4);
freq_band_ps_zscore = (freq_band_ps - mu)./sigma;
all_lfp_power = mean(freq_band_ps_zscore,5);

used_trial_idx = ~any(isnan(all_lfp_power),ndims(all_lfp_power));

end

function used_call_idx = get_used_calls(used_call_type,callIDs,cData,expDay,callSeparation,minICI,target_bat_num,self_bat_num)

switch used_call_type
        case 'allCalls'
            used_call_idx = true(1,length(callIDs));
        case 'vd_used_calls'
            used_call_idx = callIDs;
        case 'ICI'
            cPos = cData('callID',callIDs,'expDay',expDay).callPos;
            interCallInterval = [0; cPos(2:end,1) - cPos(1:end-1,2)];
            used_call_idx = interCallInterval > callSeparation;
        case 'ICI_inter_bat'
            cPos = cData('callID',callIDs,'expDay',expDay).callPos;
            interCallInterval = [0; cPos(2:end,1) - cPos(1:end-1,2)]';
            
            call_bat_nums = cData('callID',callIDs,'expDay',expDay).batNum;
            target_call_idx = cellfun(@(bNum) any(strcmp(bNum,target_bat_num)),call_bat_nums);
            self_call_idx = cellfun(@(bNum) any(strcmp(bNum,self_bat_num)),call_bat_nums);
            target_call_cPos = cPos(target_call_idx,1);
            other_bat_cPos = cPos(~target_call_idx & ~self_call_idx,1);
            self_call_cPos = cPos(self_call_idx,1);
            
            ICI_inter_bat = nan(1,length(callIDs));
            ICI_self_call = nan(1,length(callIDs));
            for call_k = 1:length(callIDs)
                if target_call_idx(call_k)
                    ICI = cPos(call_k,1) - other_bat_cPos;
                else
                    ICI = cPos(call_k,1) - target_call_cPos;
                end
                ICI = ICI(ICI>0);
                if ~isempty(ICI)
                    ICI_inter_bat(call_k) = min(ICI);
                else
                    ICI_inter_bat(call_k) = Inf;
                end
                
                ICI_self = cPos(call_k,1) - self_call_cPos;
                ICI_self = ICI_self(ICI_self>0);
                if ~isempty(ICI_self)
                    ICI_self_call(call_k) = min(ICI_self);
                else
                    ICI_self_call(call_k) = Inf;
                end
                
            end
            
            used_call_idx = ICI_inter_bat > callSeparation & ICI_self_call > callSeparation & interCallInterval > minICI;
end

end