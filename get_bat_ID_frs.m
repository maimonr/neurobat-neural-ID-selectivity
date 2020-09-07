function [trialFR, used_call_nums, cell_ks] = get_bat_ID_frs(vd,cData,batNum,varargin)

pnames = {'predictionType','timeLims','used_bat_nums','used_call_type','winSize','callSeparation','minICI','shuffle_bat_nums','shuffle_spikes','shift_spikes','keep_time_bins','loadSpikes'};
dflts  = {'prod_vs_percep',[-0.25 0.25],vd.batNums,'allCalls',0.25,1,0.05,false,false,false,false,false};
[predictionType,timeLims,used_bat_nums,used_call_type,winSize,callSeparation,minICI,shuffle_bat_nums,shuffle_spikes,shift_spikes,keep_time_bins,loadSpikes] = internal.stats.parseArgs(pnames,dflts,varargin{:});
slidingWins = timeLims(1):winSize:timeLims(2);
cell_ks = find(vd.usable & ismember(vd.batNum,batNum));
nCell = length(cell_ks);
switch predictionType
    case 'one_vs_all'
        assert(length(used_bat_nums) == 1);
        n_call_type = 2;
    case 'prod_vs_percep'
        n_call_type = 2;
    case 'singleBat'
        n_call_type = length(used_bat_nums)-1;
end

trialFR = cell(nCell,n_call_type );
used_call_nums = cell(1,nCell);
k = 1;
cData.exp_date_num = datenum(cData.expDay);

all_call_nums = unique([vd.callNum{cell_ks}]);
nCall = length(all_call_nums);
all_call_bat_nums = cell(1,nCall);

for call_k = 1:nCall
   all_call_bat_nums{call_k} = cData.batNum{cData.callID == all_call_nums(call_k)};  
end

% for call_k = 1:nCall
%     try
%         bout_nums = get_call_bout_nums(cData,all_call_nums(call_k),abs(diff(timeLims)));
%     catch
%         bout_nums = NaN;
%     end
%     bout_bat_nums = cData('callID',bout_nums).batNum;
%     if any(cellfun(@iscell,bout_bat_nums))
%         all_call_bat_nums{call_k} = unique(cellflat(bout_bat_nums));
%     else
%         all_call_bat_nums{call_k} = unique(bout_bat_nums);
%     end
% end

if shuffle_bat_nums
    selfIdx = cellfun(@(x) any(strcmp(x,batNum)),all_call_bat_nums);
    other_bat_nums = all_call_bat_nums(~selfIdx);
    permIdx = randperm(sum(~selfIdx));
    all_call_bat_nums(~selfIdx) = other_bat_nums(permIdx);
end

for cell_k = cell_ks
    
    used_call_idx = get_used_calls(used_call_type,vd.callNum{cell_k}',cData,vd.expDay(cell_k),callSeparation,minICI,used_bat_nums,batNum);
    callIDs = vd.callNum{cell_k}(used_call_idx);
    
    fr_binned = get_used_spikes(loadSpikes,vd,cell_k,used_call_idx,cData,slidingWins,shuffle_spikes,shift_spikes,winSize);
    
    used_call_bat_nums = cell(1,length(callIDs));
    for call_k = 1:length(callIDs)
        used_call_bat_nums{call_k} = all_call_bat_nums{all_call_nums == callIDs(call_k)};
    end
    
    used_call_nums{k} = cell(1,n_call_type);
    
    for call_type_k = 1:n_call_type
        
        switch predictionType
            case 'one_vs_all'
                self_call_idx = cellfun(@(bNum) any(strcmp(bNum,batNum)),used_call_bat_nums);
                if call_type_k == 1
                    callIdx = cellfun(@(bNum) any(strcmp(bNum,used_bat_nums)),used_call_bat_nums);
                else
                    callIdx = ~cellfun(@(bNum) any(strcmp(bNum,used_bat_nums)),used_call_bat_nums);
                end
                callIdx = callIdx & ~self_call_idx;
            case 'prod_vs_percep'
                if call_type_k == 1
                    callIdx = cellfun(@(bNum) any(strcmp(bNum,batNum)),used_call_bat_nums);
                else
                    callIdx = ~cellfun(@(bNum) any(strcmp(bNum,batNum)),used_call_bat_nums);
                end
            case 'singleBat'
                predBats = setdiff(used_bat_nums,batNum);
                callIdx = cellfun(@(bNum) any(strcmp(bNum,predBats{call_type_k})),used_call_bat_nums);
                trialFR{k,call_type_k} = reshape(fr_binned(callIdx,:),[],1);
        end
        
        if keep_time_bins
            trialFR{k,call_type_k} = fr_binned(callIdx,:);
        else
            trialFR{k,call_type_k} = reshape(fr_binned(callIdx,:),[],1);
        end
        
        used_call_nums{k}{call_type_k} = callIDs(callIdx);
    end
    k = k + 1;
end
end

function fr_binned = get_used_spikes(loadSpikes,vd,cell_k,used_call_idx,cData,slidingWins,shuffle_spikes,shift_spikes,winSize)
if loadSpikes
    cPos = cData('callID',vd.callNum{cell_k}','expDay',vd.expDay(cell_k)).callPos;
    cPos = cPos(used_call_idx,1);
    
    timestamps = vd.getSpikes(cell_k);
    timestamps = timestamps*1e-3;
    timestamps = inRange(timestamps,[min(cPos); max(cPos)] + slidingWins([1 end]));
    
    if shuffle_spikes
        ISI = diff(timestamps);
        [f,xi] = ecdf(ISI);
        u = rand(length(timestamps),1);
        ISI_rand = interp1(f,xi,u,'linear')';
        timestamps = timestamps(1) + cumsum(ISI_rand);
    end
    
    if shift_spikes
        shiftAmount = 1e3;
        tsRange = range(timestamps);
        minTs = min(timestamps);
        timestamps = mod((timestamps-minTs)+shiftAmount,tsRange)+minTs;
    end
    
    fr_binned = nan(sum(used_call_idx),length(slidingWins)-1);
    for call_k = 1:sum(used_call_idx)
        fr_binned(call_k,:) = histcounts(timestamps,cPos(call_k) + slidingWins)/winSize;
    end
    
else
    
    if shift_spikes || shuffle_spikes
        disp('Can''t shuffle spikes without loading all spikes')
        keyboard
    else
        
        used_spikes = vd.callSpikes{cell_k}(used_call_idx);
        fr_binned = cellfun(@(spikes) histcounts(spikes,slidingWins)/winSize,used_spikes,'un',0);
        fr_binned = vertcat(fr_binned{:});
        
    end
end
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