function [pred_acc,pred_err] = calculate_bat_ID_pred_over_time(vdList,cdList,varargin)

pnames = {'timeWins','used_call_type','callSeparation','minICI','winSize','n_population_reps','n_acc_rep','predType','excluded_cells','exclude_cell_frac','exclude_n_resp','shuffle_bat_nums','shuffle_spikes','shift_spikes','keep_time_bins'};
dflts  = {[-0.1; 0.1],'ICI_inter_bat',0.1,0.05,0.1,1e2,1e2,'one_vs_all',[],NaN,false,false,false,false,false};
[timeWins,used_call_type,callSeparation,minICI,winSize,n_population_reps,n_acc_rep,predType,excluded_cell_list,exclude_cell_frac,exclude_n_resp,shuffle_bat_nums,shuffle_spikes,shift_spikes,keep_time_bins] = internal.stats.parseArgs(pnames,dflts,varargin{:});

timeWins = sort(timeWins,2);
timeWins = unique(timeWins','rows')';

minCalls = 10;

if ~keep_time_bins
    minCalls = round(minCalls*abs(diff(timeWins(:,1)))/winSize);
end

n_time_reps = size(timeWins,2);
n_mdl_reps = length(exclude_cell_frac);

batNums = {vdList(1).batNums(1:3),vdList(2).batNums};
expStrs = {'adult','adult_operant'};

pred_acc = struct('adult',[],'adult_operant',[]);
err_k = 1;
pred_err = {};
for vd_k = 1:length(vdList)
    
    vd = vdList(vd_k);
    cData = cdList(vd_k);
    bNums = batNums{vd_k};
    nBat = length(bNums);
    
    if ~isempty(excluded_cell_list)
        excluded_cells = excluded_cell_list{vd_k};
    else
        excluded_cells = [];
    end
    
    pred_acc.(expStrs{vd_k}) = cell(n_mdl_reps,n_time_reps);
    t = tic;
    
    for time_k = 1:n_time_reps
        
        call_fr_params = {'timeLims',timeWins(:,time_k),'used_call_type',used_call_type,'winSize',winSize,'callSeparation',callSeparation,'minICI',minICI,'predictionType',predType,'shuffle_bat_nums',shuffle_bat_nums,'shuffle_spikes',shuffle_spikes,'shift_spikes',shift_spikes,'keep_time_bins',keep_time_bins};
        
        switch predType
            
            case 'singleBat'
                trialFR = cell(nBat,1);
            case 'one_vs_all'
                trialFR = cell(nBat,nBat-1);
            case 'prod_vs_percep'
                trialFR = cell(nBat,1);
        end
        
        cell_ks = cell(1,nBat);
        for bat_k = 1:nBat
            switch predType
                case 'singleBat'
                    current_bat_nums = {bNums};
                    n_select_bat = 1;
                case 'one_vs_all'
                    current_bat_nums = num2cell(setdiff(bNums,bNums(bat_k)));
                    n_select_bat = length(current_bat_nums);
                case 'prod_vs_percep'
                    current_bat_nums = {bNums};
                    n_select_bat = 1;
            end
            
            for select_bat_k = 1:n_select_bat
                selectBat = current_bat_nums{select_bat_k};
                [trialFR{bat_k,select_bat_k}, used_call_nums{bat_k,select_bat_k}, cell_ks{bat_k}] = get_bat_ID_frs(vd,cData,bNums(bat_k),'used_bat_nums',selectBat,call_fr_params{:});
            end
        end
        
        for mdl_k = 1:n_mdl_reps
            pred_params = {'minCalls',minCalls,'n_population_reps',n_population_reps,'n_acc_rep',n_acc_rep,'exclude_cell_frac',exclude_cell_frac(mdl_k),'exclude_cell_ks',excluded_cells,'exclude_n_resp',exclude_n_resp};
            
            pred_acc.(expStrs{vd_k}){mdl_k,time_k} = nan(nBat,n_select_bat,n_acc_rep);
            
            for bat_k = 1:nBat
                try 
                    pred_acc.(expStrs{vd_k}){mdl_k,time_k}(bat_k,:,:) = predict_bat_id_from_fr(trialFR(bat_k,:),pred_params{:},'cell_ks',cell_ks{bat_k});
                catch err
                    pred_err{err_k} = err;
                    err_k = err_k + 1;
                end
            end
            
            fprintf('%d time wins elapsed',time_k)
            toc(t)
            
        end
    end
end