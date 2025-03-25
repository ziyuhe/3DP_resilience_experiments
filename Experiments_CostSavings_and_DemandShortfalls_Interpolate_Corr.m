% =========================================================================
% Script Name:       Experiments_CostSavings_and_DemandShortfalls_Interpolate_Corr.m
% Date:              02/01/2025
% Description:       
%   - This is part of a series of experiments prefixed with "Experiments_CostSavings_and_DemandShortfalls".
%   - Evaluates the impact of key hyperparameters on the performance of the 3DP resilience strategy.
%   - Specifically, here we analyze effect of varying:
%           - marginal failure rate p
%           - yield loss rate 
%       on resilience performance.
%   - Focuses on cost savings and reduction in demand shortfalls under different conditions.
%   - Outputs:
%       - Cost savings analysis compared to traditional backup (TM-only benchmark).
%       - Demand shortfall distributions across different 3DP capacity settings.
%       - Boxplots and histograms illustrating performance trade-offs.
%
% Hyperparameters and Default Values:
%   - **c_cap** (Fixed cost per unit 3DP capacity): 
%       Default = cost_of_3dp_per_machine_month(1) / speed_per_machine_month(1).
%   - **c_3DP** (3DP variable cost): Default = c_source.
%   - **Disruption Parameters** (p, yield_loss_rate): Default = (0.05, 0.05).
%   - **Disruption Correlation**: Default = independent failures.
%
% Experiment Scope:
%   - **Focus**: Analyzing the impact of changes in correlations
%   - **Comparisons**:
%       - Different 3DP capacity investments and their effect on resilience.
%       - Trade-offs between cost savings and demand fulfillment.
%
% Notes:
%   - 3DP capacity represents the total monthly output of printing materials.
%   - Costs and weights are converted from grams to kilograms for better cost scaling.
%   - Adjustments:
%       - "weight_all" divided by 1000 (per-unit product weight).
%       - "speed_per_machine_month" divided by 1000 (material consumption per printer per month).
%       - "Monthly_Weight_3scenarios_all" divided by 1000 (demand scenarios in weight).
% =========================================================================


addpath('Utilities')





%% Read Data (General Information)
filename = 'Problem_Data/All/Mattel_All_Suppliers_Ave_Weight_Quantity.csv';
all_csvdata = readtable(filename);

% Product weight (converted from grams to kg)
weight_all = all_csvdata.SyntheticProductWeight_gram_ / 1000;

% Cost parameters
c_source_all = all_csvdata.SyntheticSourcingCost;      % Primary sourcing cost
c_3DP_source_all = all_csvdata.Synthetic3DPCost;       % 3DP production cost
c_TM_source_all = all_csvdata.SyntheticExpeditionCost; % TM production cost
c_price_all = all_csvdata.SyntheticPrice;              % Selling price

% Derived cost parameters
c_3DP_all = c_3DP_source_all - c_source_all; % Extra cost of 3DP vs. primary sourcing
c_TM_all = c_TM_source_all - c_source_all;   % Extra cost of TM vs. primary sourcing
v_all = c_price_all - c_source_all;          % Lost-sale penalty (sales margin)
h_all = c_source_all;                         % Holding cost (assuming zero salvage value)

num_suppliers_all = length(h_all);


% 3DP fixed cost modeled as linear in capacity (monthly depreciation per printer)
cost_of_3dp_per_machine_month = [5000, 10000, 15000, 17500, 20000, 22500, ...
                                 25000, 30000, 35000, 40000] / 120;

% 3DP speed (output capacity per month in kg)
speed_per_machine_month = [18000, 50000, 90000] / 1000;

% Default disruption parameters (medium disruption case)
p_medium = 0.05; 
yield_loss_rate_medium = 0.05;

%% Load 3-Scenario Random Demand Data
filename = 'Problem_Data/All/Mattel_All_Suppliers_Ave_Month_Weight_Quantity_3scenarios.csv';
all_csvdata_3scenarios = readtable(filename);
num_scenarios = 3;

% Extract column names for weight, quantity, and probabilities
weight_sceanarios_col_names = strcat('WeightScenario', string(1:num_scenarios), '_grams_');
quantity_scenarios_col_names = strcat('QuantityScenario', string(1:num_scenarios));
probability_col_names = strcat('Scenario', string(1:num_scenarios), 'Probability');

% Read scenario data
Monthly_Weight_3scenarios_all = zeros(height(all_csvdata_3scenarios), num_scenarios);
Monthly_Quantity_3scenarios_all = zeros(height(all_csvdata_3scenarios), num_scenarios);
Demand_Probability_3scenarios_all = zeros(height(all_csvdata_3scenarios), num_scenarios);

for k = 1:num_scenarios
    Monthly_Weight_3scenarios_all(:, k) = all_csvdata_3scenarios.(weight_sceanarios_col_names{k}) / 1000; % Convert to kg
    Monthly_Quantity_3scenarios_all(:, k) = all_csvdata_3scenarios.(quantity_scenarios_col_names{k});
    Demand_Probability_3scenarios_all(:, k) = all_csvdata_3scenarios.(probability_col_names{k});
end

% Normalize probabilities to prevent numerical precision issues
Demand_Probability_3scenarios_all = Demand_Probability_3scenarios_all ./ sum(Demand_Probability_3scenarios_all, 2);

% Compute mean demand per supplier
mean_demand_3scenarios_all = sum(Monthly_Quantity_3scenarios_all .* Demand_Probability_3scenarios_all, 2);










%% Do a trial on sampler

% num_suppliers = num_suppliers_all;
% supplier_subset_idx = false(num_suppliers_all, 1);
% 
% Monthly_Weight_3scenarios = Monthly_Weight_3scenarios_all;
% Monthly_Quantity_3scenarios = Monthly_Quantity_3scenarios_all;
% Demand_Probability_3scenarios = Demand_Probability_3scenarios_all;
% mean_demand_3scenarios = mean_demand_3scenarios_all;
% 
% c_source = c_source_all;
% 
% c_3DP = c_3DP_all; 
% c_TM = c_TM_all;   
% v = v_all;         
% h = h_all;  
% 
% weight = weight_all;

num_suppliers = 5;
supplier_subset_idx = false(num_suppliers_all, 1);
supplier_subset_idx(randperm(num_suppliers_all, num_suppliers)) = true;

Monthly_Weight_3scenarios = Monthly_Weight_3scenarios_all(supplier_subset_idx, :);
Monthly_Quantity_3scenarios = Monthly_Quantity_3scenarios_all(supplier_subset_idx, :);
Demand_Probability_3scenarios = Demand_Probability_3scenarios_all(supplier_subset_idx, :);
mean_demand_3scenarios = mean_demand_3scenarios_all(supplier_subset_idx);

c_source = c_source_all(supplier_subset_idx);

c_3DP = c_3DP_all(supplier_subset_idx); 
c_TM = c_TM_all(supplier_subset_idx);   
v = v_all(supplier_subset_idx);         
h = h_all(supplier_subset_idx);  

weight = weight_all(supplier_subset_idx);


%% Sample some data for BoE Submod Max SAA
input_preprocess_medium_sampled.num_suppliers = num_suppliers;
input_preprocess_medium_sampled.num_scenarios = num_scenarios;

input_preprocess_medium_sampled.yield_loss_rate = yield_loss_rate_medium;
input_preprocess_medium_sampled.p_disrupt = p_medium;

input_preprocess_medium_sampled.Monthly_Quantity = Monthly_Quantity_3scenarios;
input_preprocess_medium_sampled.Demand_Probability = Demand_Probability_3scenarios;

input_preprocess_medium_sampled.sample_mode = 5;


input_preprocess_medium_sampled.disruption_sample_flag = 1;
input_preprocess_medium_sampled.demand_sample_flag = 1;
demand_samplesize_saa = 1;
disruption_samplesize_saa = 5000;         

Corr_interploate_rate_set = [0:0.1:0.9, 0.91:0.01:1];

success_rate = [];
marg_succ_rate = [];

for cc = 1 : length(Corr_interploate_rate_set)

    input_preprocess_medium_sampled.p0 = Corr_interploate_rate_set(cc)*p_medium;

    input_preprocess_medium_sampled.demand_num_per_disruption = demand_samplesize_saa;
    input_preprocess_medium_sampled.disruption_samplesize = disruption_samplesize_saa;
    
    output_preprocess_medium_sampled = Data_prep_for_MIP(input_preprocess_medium_sampled);
    
    disruption_demand_joint_prob_medium_sampled = output_preprocess_medium_sampled.disruption_demand_joint_prob;
    failure_data_medium_sampled = output_preprocess_medium_sampled.failure_data;
    demand_data_medium_sampled = output_preprocess_medium_sampled.demand_data;
    

    for ll = 1:length(disruption_demand_joint_prob_medium_sampled)
        success_rate(cc,ll) = sum(failure_data_medium_sampled(:,ll)==1)/num_suppliers*100;
    end

    for i = 1:num_suppliers
        marg_succ_rate(cc,i) = sum(failure_data_medium_sampled(i,:)==1)/length(disruption_demand_joint_prob_medium_sampled)*100;
    end

end


for cc = 1 : length(Corr_interploate_rate_set)

    boxplot(success_rate(cc,:), 'Positions', 100*Corr_interploate_rate(cc), width=2.5);
    hold on

end

ylim([90,101])









%% Series of Experiments: "Experiments_CostSavings_and_DemandShortfalls"  
% We conduct a series of experiments using files with the prefix  "Experiments_CostSavings_and_DemandShortfalls" 
% to study the impact of hyperparameters on the performance of the 3DP resilience strategy. 


%% Key hyperparameters and their default values:  
%   - **c_cap** (per-unit fixed cost of 3DP reservation):  
%     Default = cost_of_3dp_per_machine_month(1) / speed_per_machine_month(1).  
%   - **c_3DP**: Default = c_source.  
%   - **(p, yield_loss_rate)**: Default = (0.05, 0.05).  
%   - **Correlation among disruptions**: Default = independent.  
%  
% Performance of the 3DP resilience strategy is evaluated based on:  
%   - **Cost savings**  
%   - **Reduction in demand shortfalls**  
  

%% In this experiment, we focus on analyzing the impact of:
%   - Correlation Structure

%% Here we apply a special treatment that allows us to simulate the shift of correlation (among disruptions) from weak to strong:
%   - Let Yj, j=1...n be independent Bernoulli, with success rate (p_j-p_0)/(1-p0)
%   - Let Y0, be Bernoulli (ind. of Yj's), with success rate p0
%   - Denote disruption s_j = 1 - max(Yj, Y0) * yield_loss
% This treatment ensures that the marginal distribution of s_j is:
%   - P(s_j = 1) = p
%   - P(s_j = 1-yield_loss) = 1 - p
% When p_0/p_j = 0 => independence
% Wehn p_0/p_j = 1 => comonotonicity


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VARYING CORRELATION INTERPOLATION RATIO: COMPUTATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

capacity_3dp_percentage = [1e-2, 1e-1*[1:9], 1:2:25, 30:5:50, 75, 100]/100;

Corr_interploate_rate_set = [0:0.1:1];

OUTPUT_MEDIUM_BOE_VARYING_CORR_INTERP_RATE = {};
COST_TMONLY_VARYING_CORR_INTERP_RATE = {};

%% We consider two yield loss regimes: 
%       - yield_loss_rate = 0.05
%       - yield_loss_rate = 1

yield_loss_rate_set = [0.05, 1];

for ii = 1 : length(yield_loss_rate_set)

    yield_loss_rate_medium = yield_loss_rate_set(ii);
    
    for ll = 1 : length(Corr_interploate_rate_set)
    
        num_suppliers = num_suppliers_all;
        supplier_subset_idx = false(num_suppliers_all, 1);
        
        Monthly_Weight_3scenarios = Monthly_Weight_3scenarios_all;
        Monthly_Quantity_3scenarios = Monthly_Quantity_3scenarios_all;
        Demand_Probability_3scenarios = Demand_Probability_3scenarios_all;
        mean_demand_3scenarios = mean_demand_3scenarios_all;
                
        c_source = c_source_all;
        
        c_3DP = c_3DP_all; 
        c_TM = c_TM_all;   
        v = v_all;         
        h = h_all;  
        
        weight = weight_all;
    
        
        %% (PART OF PRE-PROCESSING) Compute the costs of each supplier when they are
        %% - Backed-up by TM 
        %% - No back-up at all 
        %% - Backed-up by 3DP but inf. capacity
        % Medium disruptions
        % Without backup (op. and total costs: individual product and sum)
        input_medium_no3dp.n = num_suppliers;
        input_medium_no3dp.v = v;
        input_medium_no3dp.h = h;
        input_medium_no3dp.p = p_medium;
        input_medium_no3dp.yield_loss_rate = yield_loss_rate_medium;
        input_medium_no3dp.Demand_atoms = Monthly_Quantity_3scenarios;
        input_medium_no3dp.Demand_prob = Demand_Probability_3scenarios;
        input_medium_no3dp.Demand_mean = mean_demand_3scenarios;
        input_medium_no3dp.TM_flag = 0;
        output_medium_no3dp = Cost_No3DP_or_TM(input_medium_no3dp);
                
        % All backed-up by 3DP (inf. capacity, we only care about the solution)
        input_medium_3DP_infK = input_medium_no3dp;
        input_medium_3DP_infK.v = c_3DP;
        output_medium_3DP_infK = Cost_No3DP_or_TM(input_medium_3DP_infK);
        
        % All backed-up by TM (op. and total costs: individual product and sum)
        TM_retainer_ratio = 0.75;
        C_TM = TM_retainer_ratio*c_source.*mean_demand_3scenarios;
        input_medium_TM = input_medium_no3dp;
        input_medium_TM.TM_flag = 1;
        input_medium_TM.c_TM = c_TM; 
        input_medium_TM.C_TM = C_TM; 
        output_medium_TM = Cost_No3DP_or_TM(input_medium_TM);
        
        
        
        %% The products that are originally backed-up by TM
        TM_backup_set = output_medium_TM.TM_cost < output_medium_no3dp.opt_val;
        nobackup_set = 1 - TM_backup_set;
        COST_TMONLY_VARYING_CORR_INTERP_RATE{ll} = sum(output_medium_TM.TM_cost(TM_backup_set))+sum(output_medium_no3dp.opt_val(logical(nobackup_set)));
        
                
        %% Sample some data for BoE Submod Max SAA
        input_preprocess_medium_sampled.num_suppliers = num_suppliers;
        input_preprocess_medium_sampled.num_scenarios = num_scenarios;
        
        input_preprocess_medium_sampled.yield_loss_rate = yield_loss_rate_medium;
        input_preprocess_medium_sampled.p_disrupt = p_medium;
    
        input_preprocess_medium_sampled.p0 = Corr_interploate_rate_set(ll)*p_medium;
        
        input_preprocess_medium_sampled.Monthly_Quantity = Monthly_Quantity_3scenarios;
        input_preprocess_medium_sampled.Demand_Probability = Demand_Probability_3scenarios;
        
        input_preprocess_medium_sampled.sample_mode = 5;
        input_preprocess_medium_sampled.disruption_sample_flag = 1;
        input_preprocess_medium_sampled.demand_sample_flag = 1;
      
        if num_suppliers < 20
            demand_samplesize_saa = 1;
            disruption_samplesize_saa = 20000;
        elseif num_suppliers < 30
            demand_samplesize_saa = 1;
            disruption_samplesize_saa = 45000;        
        elseif num_suppliers < 40
            demand_samplesize_saa = 1;
            disruption_samplesize_saa = 80000; 
        else
            demand_samplesize_saa = 1;
            disruption_samplesize_saa = 125000;         
        end
    
        
        input_preprocess_medium_sampled.disruption_samplesize = disruption_samplesize_saa;
        input_preprocess_medium_sampled.demand_num_per_disruption = demand_samplesize_saa;
        
        output_preprocess_medium_sampled = Data_prep_for_MIP(input_preprocess_medium_sampled);
        
        disruption_demand_joint_prob_medium_sampled = output_preprocess_medium_sampled.disruption_demand_joint_prob;
        failure_data_medium_sampled = output_preprocess_medium_sampled.failure_data;
        demand_data_medium_sampled = output_preprocess_medium_sampled.demand_data;
        
        input_boe.disruption_demand_joint_prob = disruption_demand_joint_prob_medium_sampled;
        input_boe.failure_data = failure_data_medium_sampled;
        input_boe.demand_data = demand_data_medium_sampled;
        
        
        input_boe.p0 = Corr_interploate_rate_set(ll)*p_medium;
                
                
                
        %% Prepare data for BoE Submod Max
        Obj_const = - output_medium_no3dp.opt_val;
        U0_with_vmean = output_medium_no3dp.opt_val + v.*mean_demand_3scenarios;
        U0_no_vmean = output_medium_no3dp.opt_val;
        TM_Delta = output_medium_TM.TM_cost - output_medium_no3dp.opt_val;
        
        ratio_over_weight = (v-c_3DP)./weight;
        
        % Optimal primal ordering when no backup
        q0 = output_medium_no3dp.opt_q;
        
        % Compute the probability that [Dj - q0_j*s_j]^+ > 0 and == 0 (The probability of having unfilled demand)
        % - Call P([Dj - q0_j*s_j]^+ > 0) "pi_p"
        % - Call P([Dj - q0_j*s_j]^+ = 0) "pi_0"
        
        pi_p = [];
        pi_0 = [];
        
        for j = 1 : num_suppliers
            
            tmp1 = max(0, Monthly_Quantity_3scenarios(j,:)' - q0(j)*[1-yield_loss_rate_medium, 1]);
            tmp2 = Demand_Probability_3scenarios(j,:)'*[p_medium, 1-p_medium];
        
            pi_p(j,:) = sum(sum(tmp2(tmp1 > 1e-5)));
            pi_0(j,:) = sum(sum(tmp2(tmp1 <= 1e-5)));
            
        end
        
        input_boe.U0_with_vmean = U0_with_vmean;
        input_boe.U0_no_vmean = U0_no_vmean;
        input_boe.U0 = U0_with_vmean;
        input_boe.q0 = q0;
        input_boe.Obj_const = Obj_const;
        input_boe.TM_Delta = TM_Delta;
        input_boe.ratio_over_weight = ratio_over_weight;
        input_boe.pi_p = pi_p;
        input_boe.pi_0 = pi_0;
        
        input_boe.speed_per_machine_month = speed_per_machine_month;
        input_boe.cost_of_3dp_per_machine_month = cost_of_3dp_per_machine_month;
        
        input_boe.p = p_medium;
        input_boe.yield_loss_rate = yield_loss_rate_medium;
        input_boe.num_suppliers = num_suppliers;
        input_boe.num_scenarios = num_scenarios;
        
        input_boe.Monthly_Quantity = Monthly_Quantity_3scenarios;
        input_boe.Monthly_Weight = Monthly_Weight_3scenarios;
        input_boe.Demand_Probability = Demand_Probability_3scenarios;
        input_boe.mean_demand = mean_demand_3scenarios;
        input_boe.TM_cost = output_medium_TM.TM_cost;
        input_boe.nobackup_cost = output_medium_no3dp.opt_val;
        
        input_boe.c_3DP = c_3DP;
        input_boe.v = v;
        input_boe.h = h;
        input_boe.weight = weight;
        
        input_boe.q_ub = output_medium_no3dp.opt_q;    % HERE WE ADD BOUNDS FOR 1ST STAGE DECISIONS
        input_boe.q_lb = output_medium_3DP_infK.opt_q; % HERE WE ADD BOUNDS FOR 1ST STAGE DECISIONS
        
             
        
        for k = 1 : length(capacity_3dp_percentage)
        
            fprintf("%3.2f Percent of Max Yield Shortfall \n\n", capacity_3dp_percentage(k)*100)
        
            capacity_percentage = capacity_3dp_percentage(k);
            K_3DP_medium = capacity_percentage*sum(max(Monthly_Weight_3scenarios'));
            C_3DP_medium = (K_3DP_medium./speed_per_machine_month)'*cost_of_3dp_per_machine_month;
            
            input_boe.K_3DP = K_3DP_medium;
            input_boe.C_3DP = C_3DP_medium;
        
            input_boe.add_max_pivot_rule = 1;
            input_boe.delete_max_pivot_rule = 0;
        
            input_boe.GRB_display = 0;
        
            input_boe.auto_recompute = 1;
            
            input_boe.recompute_flag = 2;
    
            input_boe.recompute_distr = 3;
        
            input_boe.recompute_sample_mode = 1;
            input_boe.recompute_disruption_sample_flag = 0;
            input_boe.recompute_demand_sample_flag = 0;
            
            input_boe.recompute_disruption_samplesize_eval = 1000;
            input_boe.recompute_demand_samplesize_eval = 500; 
            input_boe.recompute_disruption_samplesize_finaleval = 1000;
            input_boe.recompute_demand_samplesize_finaleval = 500;
        
            input_boe.recompute_sgd_Maxsteps = 5e5;
            
            if k == 1
                one2n = [1:num_suppliers];
                input_boe.A_init = one2n(logical(nobackup_set));
                % input_boe.A_init = [];
            else
                input_boe.A_init = OUTPUT_MEDIUM_BOE_VARYING_CORR_INTERP_RATE{ll, k-1}.A_t;
            end
        
            if num_suppliers <= 10
                output_boe = BoE_Approx_Max_Submod_Exact(input_boe);
            else
                output_boe = BoE_Approx_Max_Submod_SAA_alternative(input_boe);
            end
    
            OUTPUT_MEDIUM_BOE_VARYING_CORR_INTERP_RATE{ll, k} = output_boe; 
        
            fprintf(fileID2, 'p0 = %3.2f * p,    k=%3.2f %% \n', Corr_interploate_rate_set(ll), capacity_3dp_percentage(k));
            fprintf(fileID2, '3DP Set: %d', OUTPUT_MEDIUM_BOE_VARYING_CORR_INTERP_RATE{ll, k}.A_t);
            fprintf(fileID2, '\n\n')
        
            disp("TO NEXT ONE! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ")
        
        end
    
    end

    X_BOE_CORR_INTER = {}; 
    SWITCHED_BOE_CORR_INTER = {};
    for ll = 1 : length(Corr_interploate_rate_set)
        for i = 1 : length(speed_per_machine_month)  
            for j = 1 : length(cost_of_3dp_per_machine_month)
                X_BOE_CORR_INTER{ll,i,j} = zeros(num_suppliers,length(capacity_3dp_percentage));
                SWITCHED_BOE_CORR_INTER{ll,i,j} = zeros(num_suppliers,length(capacity_3dp_percentage));
            end
        end
    end
    TOTAL_COST_CORR_INTER = {};
    COST_SAVINGS_CORR_INTER = {};
    for ll = 1 : length(Corr_interploate_rate_set)
        for k = 1 : length(capacity_3dp_percentage)
        
            for i = 1 : length(speed_per_machine_month)  
                for j = 1 : length(cost_of_3dp_per_machine_month)
        
                    TOTAL_COST_CORR_INTER{ll,i,j}(k) = OUTPUT_MEDIUM_BOE_VARYING_CORR_INTERP_RATE{ll,k}.TOTAL_COST(i,j);
                    COST_SAVINGS_CORR_INTER{ll,i,j}(k) = (COST_TMONLY_VARYING_CORR_INTERP_RATE{ll}-OUTPUT_MEDIUM_BOE_VARYING_CORR_INTERP_RATE{ll,k}.TOTAL_COST_NONZERO(i,j))/abs(COST_TMONLY_VARYING_CORR_INTERP_RATE{ll})*100;
                    X_BOE_CORR_INTER{ll, i,j}(:, k) = OUTPUT_MEDIUM_BOE_VARYING_CORR_INTERP_RATE{ll,k}.X_FINAL{i,j};
                    if ~isinf(sum(X_BOE_CORR_INTER{i,j}(:,k)))
                        SWITCHED_BOE_CORR_INTER{ll,i,j}(:,k) = TM_backup_set - (1-X_BOE_CORR_INTER{ll,i,j}(:,k));
                    end
        
                end
            end
        
        end
    end
    
    MAX_COST_SAVINGS_CORR_INTER = {};
    OPT_K_MAX_COST_SAVINGS_CORR_INTER = {};
    OPT_K_IDX_MAX_COST_SAVINGS_CORR_INTER = {};
    for ll = 1 : length(Corr_interploate_rate_set)
        for i = 1 : length(speed_per_machine_month)  
            for j = 1 : length(cost_of_3dp_per_machine_month)
                MAX_COST_SAVINGS_CORR_INTER{i,j}(ll) = max(COST_SAVINGS_CORR_INTER{ll,i,j});
                [~, OPT_K_IDX_MAX_COST_SAVINGS_CORR_INTER{i,j}(ll)] = max(COST_SAVINGS_CORR_INTER{ll,i,j});
                OPT_K_MAX_COST_SAVINGS_CORR_INTER{i,j}(ll) = capacity_3dp_percentage(OPT_K_IDX_MAX_COST_SAVINGS_CORR_INTER{i,j}(ll))*100;
            end
        end
    end

    if ii == 1
        save("Experiment_Results/Relative_Cost_Savings_Shortfalls_Corr_Interpolate/Corr_Interpolation(small_p_small_yieldloss).mat");
    else
        save("Experiment_Results/Relative_Cost_Savings_Shortfalls_Corr_Interpolate/Corr_Interpolation(small_p_large_yieldloss).mat");
    end

end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VARYING CORRELATION INTERPOLATION RATIO: COST-SAVINGS VS. K
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subset_for_plot = [1:2:11];

%% Save to csv data
DDD = load("Experiment_Results/Relative_Cost_Savings_Shortfalls_Corr_Interpolate/Corr_Interpolation(small_p_small_yieldloss).mat");
xaxis_data = DDD.Corr_interploate_rate_set(subset_for_plot);
yaxis_data = DDD.MAX_COST_SAVINGS_CORR_INTER{1,1}(subset_for_plot);
data_to_save = [xaxis_data(:), yaxis_data(:)];
csvwrite('Experiment_Results/Relative_Cost_Savings_Shortfalls_Corr_Interpolate/corr_interpoloate_costsavings(small_p_small_yieldloss).csv', data_to_save);

DDD = load("Experiment_Results/Relative_Cost_Savings_Shortfalls_Corr_Interpolate/Corr_Interpolation(small_p_large_yieldloss).mat");;
xaxis_data = DDD.Corr_interploate_rate_set(subset_for_plot);
yaxis_data = DDD.MAX_COST_SAVINGS_CORR_INTER{1,1}(subset_for_plot);
data_to_save = [xaxis_data(:), yaxis_data(:)];
csvwrite('Experiment_Results/Relative_Cost_Savings_Shortfalls_Corr_Interpolate/corr_interpoloate_costsavings(small_p_large_yieldloss).csv', data_to_save);



















%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VARYING CORRELATION INTERPOLATION RATIO: DEMAND SHORTFALL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Demand Shortfall

for ii = 1 : length(yield_loss_rate_set)

    if ii == 1
        load("Experiment_Results/Relative_Cost_Savings_Shortfalls_Corr_Interpolate/Corr_Interpolation(small_p_small_yieldloss).mat")
    else
        load("Experiment_Results/Relative_Cost_Savings_Shortfalls_Corr_Interpolate/Corr_Interpolation(small_p_large_yieldloss).mat")
    end

    yield_loss_rate_medium = yield_loss_rate_set(ii);

    Demand_shortfall_corr_inter = {};
    
    for i = 1 : length(speed_per_machine_month)  
        for j = 1 : length(cost_of_3dp_per_machine_month)
            
            fprintf("Working with case %d, %d\n\n", i,j)
    
            for ll = 1 : length(Corr_interploate_rate_set)
    
                %% Get the optimal x and q_SP
                kkk = OPT_K_IDX_MAX_COST_SAVINGS_CORR_INTER{i,j}(ll);
                x_3DP = logical(OUTPUT_MEDIUM_BOE_VARYING_CORR_INTERP_RATE{ll,kkk}.X_FINAL{i,j});
                q_SP = OUTPUT_MEDIUM_BOE_VARYING_CORR_INTERP_RATE{ll,kkk}.Q_FINAL{i,j}(x_3DP);
    
                %% Preparation
                num_suppliers = sum(x_3DP);
                supplier_subset_idx = x_3DP;
                Monthly_Weight_3scenarios = Monthly_Weight_3scenarios_all(supplier_subset_idx, :);
                Monthly_Quantity_3scenarios = Monthly_Quantity_3scenarios_all(supplier_subset_idx, :);
                Demand_Probability_3scenarios = Demand_Probability_3scenarios_all(supplier_subset_idx, :);
                mean_demand_3scenarios = mean_demand_3scenarios_all(supplier_subset_idx);
                       
                c_source = c_source_all(supplier_subset_idx);        
                c_3DP = c_3DP_all(supplier_subset_idx); 
                c_TM = c_TM_all(supplier_subset_idx);   
                v = v_all(supplier_subset_idx);         
                h = h_all(supplier_subset_idx);  
                weight = weight_all(supplier_subset_idx);
            
                capacity_percentage = capacity_3dp_percentage(kkk);
                K_3DP_medium = capacity_percentage*sum(max(Monthly_Weight_3scenarios'));
    
    
                %% First sample (D,s)
                input_preprocess_medium_sampled.num_suppliers = num_suppliers;
                input_preprocess_medium_sampled.num_scenarios = num_scenarios;
    
                input_preprocess_medium_sampled.yield_loss_rate = yield_loss_rate_medium;
                input_preprocess_medium_sampled.p_disrupt = p_medium;
    
                input_preprocess_medium_sampled.p0 = Corr_interploate_rate_set(ll)*p_medium;
    
                input_preprocess_medium_sampled.Monthly_Quantity = Monthly_Quantity_3scenarios;
                input_preprocess_medium_sampled.Demand_Probability = Demand_Probability_3scenarios;
                
                input_preprocess_medium_sampled.sample_mode = 5;
                input_preprocess_medium_sampled.disruption_sample_flag = 1; % We don't sample disruption (keep all combinations)
                input_preprocess_medium_sampled.demand_sample_flag = 1;     % For each disruption scenario, sample a fixed number of demand combos
                
                disruption_samplesize_saa = 100000;    
                demand_samplesize_saa = 1;            
                
                input_preprocess_medium_sampled.demand_num_per_disruption = demand_samplesize_saa;
                input_preprocess_medium_sampled.disruption_samplesize = disruption_samplesize_saa;
                
                output_preprocess_medium_sampled = Data_prep_for_MIP(input_preprocess_medium_sampled);
                
                disruption_demand_joint_prob_medium_sampled = output_preprocess_medium_sampled.disruption_demand_joint_prob;
                failure_data_medium_sampled = output_preprocess_medium_sampled.failure_data;
                demand_data_medium_sampled = output_preprocess_medium_sampled.demand_data;
                
                %% For each combo of (D,s), calculate the demand shortfall
                input_b2b.x = logical(ones(sum(x_3DP), 1));
                input_b2b.c_3DP = c_3DP;
                input_b2b.v = v;
                input_b2b.h = h;
                input_b2b.weight = weight;
                input_b2b.K_3DP = K_3DP_medium;
            
                for ss = 1 : length(disruption_demand_joint_prob_medium_sampled)
                    
            
                    D_sample = demand_data_medium_sampled(:, ss); 
                    s_sample = failure_data_medium_sampled(:, ss);   
                    D_bar = D_sample - q_SP.*s_sample;
                    input_b2b.D_bar = D_bar;
            
                    output_b2b = V3DP_b2b(input_b2b);
                    q_3DP = output_b2b.q_3DP;
                    
                    Demand_shortfall_corr_inter{i,j,ll}(ss) = sum( max( D_bar - q_3DP, 0 ) );
            
                end
    
            end
    
        end
    end
    
    
    %% Demand shortfall for SINGLE system
    num_suppliers = sum(nobackup_set);
    supplier_subset_idx = logical(nobackup_set);
    Monthly_Weight_3scenarios = Monthly_Weight_3scenarios_all(supplier_subset_idx, :);
    Monthly_Quantity_3scenarios = Monthly_Quantity_3scenarios_all(supplier_subset_idx, :);
    Demand_Probability_3scenarios = Demand_Probability_3scenarios_all(supplier_subset_idx, :);
    mean_demand_3scenarios = mean_demand_3scenarios_all(supplier_subset_idx);
           
    %% Get some sample first 
    input_preprocess_medium_sampled.num_suppliers = num_suppliers;
    input_preprocess_medium_sampled.num_scenarios = num_scenarios;
    input_preprocess_medium_sampled.yield_loss_rate = yield_loss_rate_medium;
    input_preprocess_medium_sampled.p_disrupt = p_medium;
    input_preprocess_medium_sampled.Monthly_Quantity = Monthly_Quantity_3scenarios;
    input_preprocess_medium_sampled.Demand_Probability = Demand_Probability_3scenarios;
    
    input_preprocess_medium_sampled.sample_mode = 5;
    input_preprocess_medium_sampled.disruption_sample_flag = 1; % We don't sample disruption (keep all combinations)
    input_preprocess_medium_sampled.demand_sample_flag = 1;     % For each disruption scenario, sample a fixed number of demand combos
    
    disruption_samplesize_saa = 100000;    
    demand_samplesize_saa = 1;
    
    input_preprocess_medium_sampled.disruption_samplesize = disruption_samplesize_saa;
    input_preprocess_medium_sampled.demand_num_per_disruption = demand_samplesize_saa;
    
    %% Get the distribution of demand shorfall under no 3DP
    opt_q_nobackup = output_medium_no3dp.opt_q(logical(nobackup_set));
    Demand_shortfall_no3DP_corr_inter = {};
    
    for ll = 1 : length(Corr_interploate_rate_set)
    
        input_preprocess_medium_sampled.p0 = Corr_interploate_rate_set(ll)*p_medium;
    
        output_preprocess_medium_sampled = Data_prep_for_MIP(input_preprocess_medium_sampled);
        
        disruption_demand_joint_prob_medium_sampled = output_preprocess_medium_sampled.disruption_demand_joint_prob;
        failure_data_medium_sampled = output_preprocess_medium_sampled.failure_data;
        demand_data_medium_sampled = output_preprocess_medium_sampled.demand_data;
    
        for ss = 1 : length(disruption_demand_joint_prob_medium_sampled)
            
            D_sample = demand_data_medium_sampled(:, ss); 
            s_sample = failure_data_medium_sampled(:, ss);
            
            Demand_shortfall_no3DP_corr_inter{1,ll}(ss,1) = sum( max( D_sample - opt_q_nobackup.*s_sample, 0 ) );
        
        end
    
    end
    
    Demand_shortfall_no_disruption = [];
    Relative_Demand_shortfall_no_disruption = [];
    for ss = 1 : length(disruption_demand_joint_prob_medium_sampled)
        D_sample = demand_data_medium_sampled(:, ss);
        Demand_shortfall_no_disruption(ss) = sum( max( D_sample - opt_q_nobackup, 0 ) );
        Relative_Demand_shortfall_no_disruption(ss) = Demand_shortfall_no_disruption(ss)/sum(max(Monthly_Quantity_3scenarios_all'))*100;
    end
    
    
    
    %% Get the relative demand shortfall (relative to total max demand)
    Relative_Demand_shortfall_corr_inter = {};
    Mean_Relative_Demand_shortfall_corr_inter = {};
    for ll = 1 : length(Corr_interploate_rate_set)
        for i = 1 : length(speed_per_machine_month)  
            for j = 1 : length(cost_of_3dp_per_machine_month)
                Relative_Demand_shortfall_corr_inter{i,j,ll} = Demand_shortfall_corr_inter{i,j,ll}/sum(max(Monthly_Quantity_3scenarios_all'))*100;
                Mean_Relative_Demand_shortfall_corr_inter{i,j}(ll) = mean(Relative_Demand_shortfall_corr_inter{i,j,ll});
            end
        end
    end
    Relative_Demand_shortfall_no3DP_corr_inter = {};
    Mean_Relative_Demand_shortfall_no3DP_corr_inter = [];
    for ll = 1 : length(Corr_interploate_rate_set)
        Relative_Demand_shortfall_no3DP_corr_inter{1,ll} = Demand_shortfall_no3DP_corr_inter{1,ll}/sum(max(Monthly_Quantity_3scenarios_all'))*100;
        Mean_Relative_Demand_shortfall_no3DP_corr_inter(ll) = mean(Relative_Demand_shortfall_no3DP_corr_inter{1,ll});
    end
    

    subset_for_plot = [1:2:11];
    for l = 1 : length(subset_for_plot)
        ll = subset_for_plot(l);
        boxplot(Relative_Demand_shortfall_corr_inter{i,j,ll},'Positions', Corr_interploate_rate_set(ll)*10+0.25, 'Widths', 0.5,'Whisker', 1, 'OutlierSize', 2)
        hold on
        boxplot(Relative_Demand_shortfall_no3DP_corr_inter{1,ll}, 'Positions', Corr_interploate_rate_set(ll)*10-0.25, 'Widths', 0.5,'Whisker', 1, 'OutlierSize', 2)
        hold on
    end
    
    plot( 10*Corr_interploate_rate_set(subset_for_plot)+0.25, Mean_Relative_Demand_shortfall_corr_inter{i,j}(subset_for_plot), '-o' )
    hold on
    plot( 10*Corr_interploate_rate_set(subset_for_plot)-0.25, Mean_Relative_Demand_shortfall_no3DP_corr_inter(subset_for_plot), '-o' )
    
    
    % Adjust x-ticks and labels
    set(gca, 'XTick', x_positions, 'XTickLabel', Corr_interploate_rate_set(subset_for_plot));
    
    ylim([0,5])
    
    
    %% Save data for csv (for the baseline C3DP case)
    i = 1; j = 1;

    subset_for_plot = [1:2:11];
    boxdata1 = [];
    boxdata2 = [];
    for l = 1 : length(subset_for_plot)
        ll = subset_for_plot(l);
        boxdata1(l,:) = Relative_Demand_shortfall_corr_inter{i,j,ll};
        boxdata2(l,:) = Relative_Demand_shortfall_no3DP_corr_inter{1,ll};
        boxposition1(l,:) = Corr_interploate_rate_set(ll)*10+0.25;
        boxposition2(l,:) = Corr_interploate_rate_set(ll)*10-0.25;
    end
    meandata1 = Mean_Relative_Demand_shortfall_corr_inter{i,j}(subset_for_plot);
    meandata2 = Mean_Relative_Demand_shortfall_no3DP_corr_inter(subset_for_plot);
    xtick_position = Corr_interploate_rate_set(subset_for_plot)*10;
    xtick_display = Corr_interploate_rate_set(subset_for_plot);
    
    
    for l = 1 : length(subset_for_plot)
         boxplot(boxdata1(l,:),'Positions', boxposition1(l), 'Widths', 0.5,'Whisker', 1, 'OutlierSize', 2)
         hold on
         boxplot(boxdata2(l,:),'Positions', boxposition2(l), 'Widths', 0.5,'Whisker', 1, 'OutlierSize', 2)
         hold on
    end
    plot(boxposition1, meandata1, '-o')
    hold on
    plot(boxposition2, meandata2, '-o')
    
    % Adjust x-ticks and labels
    set(gca, 'XTick', xtick_position, 'XTickLabel', xtick_display);
    
    ylim([-0.333,5])
    
    if ii == 1

        % Combine data into one table
        % Save boxplot data (concatenated)
        boxplot_data = [boxdata1, boxdata2];
        writematrix(boxplot_data, 'Experiment_Results/Relative_Cost_Savings_Shortfalls_Corr_Interpolate/corr_inter_shortfall(small_p_small_yieldloss)_boxdata.csv');
        
        % Save metadata (vectors)
        metadata = [boxposition1, boxposition2, meandata1', meandata2', xtick_position', xtick_display'];
        writematrix(metadata, 'Experiment_Results/Relative_Cost_Savings_Shortfalls_Corr_Interpolate/corr_inter_shortfall(small_p_small_yieldloss)_otherdata.csv');

    else

        % Combine data into one table
        % Save boxplot data (concatenated)
        boxplot_data = [boxdata1, boxdata2];
        writematrix(boxplot_data, 'Experiment_Results/Relative_Cost_Savings_Shortfalls_Corr_Interpolate/corr_inter_shortfall(small_p_large_yieldloss)_boxdata.csv');
        
        % Save metadata (vectors)
        metadata = [boxposition1, boxposition2, meandata1', meandata2', xtick_position', xtick_display'];
        writematrix(metadata, 'Experiment_Results/Relative_Cost_Savings_Shortfalls_Corr_Interpolate/corr_inter_shortfall(small_p_large_yieldloss)_otherdata.csv');

    end



end
