% =========================================================================
% Script Name:       Experiments_Compare_MIP_BoE.m
% Author:            Ziyu He
% Date:              02/04/2025
% Description:       
%   - Compares the computational performance of GRB MIP, Benders MIP, and BoE (Supermodular Approximation).
%   - Varies the number of suppliers (n) from 3 to 55.
%   - Methodology:
%       - Run BoE first to establish a baseline time ("BoE_total_time").
%       - Use "BoE_total_time" as a cutoff for GRB MIP and Benders MIP.
%       - For n > 40, BoE uses Sample Average Approximation (SAA) for feasibility.
%   - Outputs:
%       - Computation time comparison across methods.
%       - Optimality gap analysis between BoE and Benders MIP.
%       - Trends as supplier count increases.
%   - Saves intermediate data and generates comparative plots.
%
% Notes:
%   - For larger instances (n > 40), we approximate the BoE solution using Sample Average Approximation (SAA) to improve tractability.
%   - "Dedicated Backup" is labeled as "TM" (Traditional Manufacturing).
%   - 3DP capacity is the total weight of printing materials output per month.
%   - Initially measured in grams, converted to kilograms for cost scaling.
%   - Adjustments:
%       - "weight_all" divided by 1000 (per unit product weight).
%       - "speed_per_machine_month" divided by 1000 (material consumption per printer per month).
%       - "Monthly_Weight_3scenarios_all" divided by 1000 (demand scenarios in weight).
% =========================================================================






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

% 3DP capacity as a percentage of max yield shortfall or demand
capacity_3dp_percentage = [0.1:0.1:10, 10.2:0.2:15, 15.5:0.5:20, 21:50, 52:2:100] / 100;

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











%% Run BoE for n = 3:55 as a benchmark
%% - Set K = 5% of the max total demand
%% - Use Sample Average Approximation (SAA)
OUTPUT_BOE = {};
for nn = 3:55

    num_suppliers = nn;
    supplier_subset_idx = 1:nn; % Select the first nn suppliers
    
    % Extract relevant data for the selected suppliers
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
    
    %% Compute costs for each backup strategy:
    % - No backup
    % - 3DP backup (infinite capacity assumption)
    % - TM backup
    
    % No backup scenario
    input_medium_no3dp = struct('n', num_suppliers, 'v', v, 'h', h, ...
        'p', p_medium, 'yield_loss_rate', yield_loss_rate_medium, ...
        'Demand_atoms', Monthly_Quantity_3scenarios, ...
        'Demand_prob', Demand_Probability_3scenarios, ...
        'Demand_mean', mean_demand_3scenarios, 'TM_flag', 0);
    output_medium_no3dp = Cost_No3DP_or_TM(input_medium_no3dp);
    
    % 3DP backup (unconstrained capacity)
    input_medium_3DP_infK = input_medium_no3dp;
    input_medium_3DP_infK.v = c_3DP;
    output_medium_3DP_infK = Cost_No3DP_or_TM(input_medium_3DP_infK);
    
    % TM backup
    TM_retainer_ratio = 0.8;
    C_TM = TM_retainer_ratio * c_source .* mean_demand_3scenarios;
    input_medium_TM = input_medium_no3dp;
    input_medium_TM.TM_flag = 1;
    input_medium_TM.c_TM = c_TM; 
    input_medium_TM.C_TM = C_TM; 
    output_medium_TM = Cost_No3DP_or_TM(input_medium_TM);
    
    %% Sample data for BoE optimization using SAA
    input_preprocess_medium_sampled = struct('num_suppliers', num_suppliers, ...
        'num_scenarios', num_scenarios, 'yield_loss_rate', yield_loss_rate_medium, ...
        'p_disrupt', p_medium, 'Monthly_Quantity', Monthly_Quantity_3scenarios, ...
        'Demand_Probability', Demand_Probability_3scenarios, ...
        'sample_mode', 2, 'disruption_sample_flag', 1, 'demand_sample_flag', 1);

    % Set sample sizes based on number of suppliers
    if nn < 20
        demand_samplesize_saa = 200;
        disruption_samplesize_saa = 100;
    elseif nn < 30
        demand_samplesize_saa = 300;
        disruption_samplesize_saa = 150;        
    elseif nn < 40
        demand_samplesize_saa = 400;
        disruption_samplesize_saa = 200; 
    else
        demand_samplesize_saa = 500;
        disruption_samplesize_saa = 250;         
    end
    
    input_preprocess_medium_sampled.demand_num_per_disruption = demand_samplesize_saa;
    input_preprocess_medium_sampled.disruption_samplesize = disruption_samplesize_saa;
    
    % Preprocess data for the optimization problem
    output_preprocess_medium_sampled = Data_prep_for_MIP(input_preprocess_medium_sampled);
    
    input_boe = struct();
    input_boe.disruption_demand_joint_prob = output_preprocess_medium_sampled.disruption_demand_joint_prob;
    input_boe.failure_data = output_preprocess_medium_sampled.failure_data;
    input_boe.demand_data = output_preprocess_medium_sampled.demand_data;
    
    %% Compute parameters for BoE optimization
    input_boe.Obj_const = -output_medium_no3dp.opt_val;
    input_boe.U0_with_vmean = output_medium_no3dp.opt_val + v .* mean_demand_3scenarios;
    input_boe.U0_no_vmean = output_medium_no3dp.opt_val;
    input_boe.TM_Delta = output_medium_TM.TM_cost - output_medium_no3dp.opt_val;
    input_boe.ratio_over_weight = (v - c_3DP) ./ weight;
    input_boe.q0 = output_medium_no3dp.opt_q;
    
    % Compute probability of unfilled demand
    pi_p = zeros(num_suppliers,1);
    pi_0 = zeros(num_suppliers,1);
    
    for j = 1:num_suppliers
        tmp1 = max(0, Monthly_Quantity_3scenarios(j,:)' - q0(j) * [1 - yield_loss_rate_medium, 1]);
        tmp2 = Demand_Probability_3scenarios(j,:)' * [p_medium, 1 - p_medium];
    
        pi_p(j,:) = sum(tmp2(tmp1 > 1e-5));
        pi_0(j,:) = sum(tmp2(tmp1 <= 1e-5));
    end
    
    input_boe.pi_p = pi_p;
    input_boe.pi_0 = pi_0;
    
    % Additional parameters
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
    input_boe.c_3DP = c_3DP;
    input_boe.v = v;
    input_boe.h = h;
    input_boe.weight = weight;
    
    % Set bounds for first-stage decisions
    input_boe.q_ub = output_medium_no3dp.opt_q;
    input_boe.q_lb = output_medium_3DP_infK.opt_q;
    
    % Set 3DP capacity and cost
    capacity_percentage = 0.05; % K = 5% of max total demand
    K_3DP_medium = capacity_percentage * sum(max(Monthly_Weight_3scenarios'));
    C_3DP_medium = (K_3DP_medium ./ speed_per_machine_month)' * cost_of_3dp_per_machine_month;
    
    input_boe.K_3DP = K_3DP_medium;
    input_boe.C_3DP = C_3DP_medium;

    % Configuration parameters
    input_boe.add_max_pivot_rule = 1;
    input_boe.delete_max_pivot_rule = 0;
    input_boe.GRB_display = 0;
    input_boe.auto_recompute = 1;
    input_boe.recompute_flag = 2;
    input_boe.recompute_sample_mode = 2;
    input_boe.recompute_disruption_sample_flag = 0;
    input_boe.recompute_demand_sample_flag = 0;
    input_boe.recompute_disruption_samplesize_eval = 1000;
    input_boe.recompute_demand_samplesize_eval = 500; 
    input_boe.recompute_disruption_samplesize_finaleval = 1000;
    input_boe.recompute_demand_samplesize_finaleval = 500;
    input_boe.recompute_sgd_Maxsteps = 2e5;
    input_boe.A_init = [];
    
    % Solve BoE optimization problem
    if nn < 5
        output_boe = BoE_Approx_Max_Submod_Exact(input_boe);
    else
        output_boe = BoE_Approx_Max_Submod_SAA(input_boe);
    end

    OUTPUT_BOE{1,nn} = output_boe;

end

% Compute total computation time
BOE_time_vs_n = zeros(1, 55);
BOE_total_time = sum(cellfun(@(x) x.solving_time, OUTPUT_BOE(3:55)));




%% NOW WE USE "BOE_total_time" as a limit for computing time for Benders
%%      - Also solves the SAA problem
%%      - We cutoff if Benders are not done in "BOE_total_time"

OUTPUT_MEDIUM_BENDERS_UNREG_SAMPLED = {};

for nn = 3 : 55
    
    num_suppliers = nn;
    supplier_subset_idx = [1:nn]; % Choose the first nn suppplier
    
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

    %% Prepare for Benders
    input_medium.n = num_suppliers;
    input_medium.c_3DP = c_3DP;
    input_medium.v = v;
    input_medium.h = h;
    input_medium.weight = weight;
    input_medium.mean_demand = mean_demand_3scenarios;
    input_medium.cost_of_3dp_per_machine_month = cost_of_3dp_per_machine_month;
    input_medium.speed_per_machine_month = speed_per_machine_month;
    input_medium.num_suppliers = num_suppliers;
    input_medium.num_scenarios = num_scenarios;
    input_medium.Monthly_Quantity = Monthly_Quantity_3scenarios;
    input_medium.Monthly_Weight = Monthly_Weight_3scenarios;
    input_medium.Demand_Probability = Demand_Probability_3scenarios;
    
    bigM_scale = 1e2;
    input_medium.TM_cost = output_medium_TM.TM_cost;
    input_medium.bigM = bigM_scale*max(sum(max(Monthly_Weight_3scenarios'))/min(weight) , max(Monthly_Quantity_3scenarios(:)));
    
    input_medium.q_ub = output_medium_no3dp.opt_q;    % HERE WE ADD BOUNDS FOR 1ST STAGE DECISIONS
    input_medium.q_lb = output_medium_3DP_infK.opt_q; % HERE WE ADD BOUNDS FOR 1ST STAGE DECISIONS
    
    input_medium.p = p_medium;
    input_medium.yield_loss_rate = yield_loss_rate_medium;
    
    input_medium_benders = input_medium;

    


    %% Setup the stopping time !!!
    input_medium_benders.stopping_time_flag = 1;
    input_medium_benders.stopping_time = BOE_total_time;
    

   

    %% Sample some data for Benders (MIP SAA)
    input_preprocess_medium_sampled.num_suppliers = num_suppliers;
    input_preprocess_medium_sampled.num_scenarios = num_scenarios;
    input_preprocess_medium_sampled.yield_loss_rate = yield_loss_rate_medium;
    input_preprocess_medium_sampled.p_disrupt = p_medium;
    input_preprocess_medium_sampled.Monthly_Quantity = Monthly_Quantity_3scenarios;
    input_preprocess_medium_sampled.Demand_Probability = Demand_Probability_3scenarios;
    
    input_preprocess_medium_sampled.sample_mode = 2;
    input_preprocess_medium_sampled.disruption_sample_flag = 1; % We don't sample disruption (keep all combinations)
    input_preprocess_medium_sampled.demand_sample_flag = 1;     % For each disruption scenario, sample a fixed number of demand combos
    
    if nn < 20
        demand_samplesize_saa = 200;
        disruption_samplesize_saa = 100;
    elseif nn < 30
        demand_samplesize_saa = 300;
        disruption_samplesize_saa = 150;        
    elseif nn < 40
        demand_samplesize_saa = 400;
        disruption_samplesize_saa = 200; 
    else
        demand_samplesize_saa = 500;
        disruption_samplesize_saa = 250;         
    end
    
    input_preprocess_medium_sampled.demand_num_per_disruption = demand_samplesize_saa;
    input_preprocess_medium_sampled.disruption_samplesize = disruption_samplesize_saa;
    
    
    output_preprocess_medium_benders = Data_prep_for_MIP(input_preprocess_medium_sampled);
    
    disruption_demand_joint_prob_medium_benders = output_preprocess_medium_benders.disruption_demand_joint_prob;
    failure_data_medium_benders = output_preprocess_medium_benders.failure_data;
    demand_data_medium_benders = output_preprocess_medium_benders.demand_data;


    %% Setup Benders
    K_3DP_medium = capacity_percentage*sum(max(Monthly_Weight_3scenarios'));
    C_3DP_medium = (K_3DP_medium./speed_per_machine_month)'*cost_of_3dp_per_machine_month;
    input_medium_benders.K_3DP = K_3DP_medium;
    input_medium_benders.C_3DP = C_3DP_medium;

    input_medium_benders.Max_Steps = 1e3;
    input_medium_benders.tolerance = 1e-4;


    %% Use the BoE results for initialization (init_cheat == 1 <=> we know the optimal solution as baseline)
    init_cheat = 0;
    if init_cheat == 1
        x_tmp = zeros(num_suppliers,1);
        x_tmp(OUTPUT_BOE{1,nn}.A_t) = 1;
        input_medium_benders.x_init = x_tmp;
        input_medium_benders.q_init = OUTPUT_BOE{1,nn}.Q_FINAL{2,1};
    else
        input_medium_benders.x_init = logical(ones(num_suppliers,1));
        input_medium_benders.q_init = min(Monthly_Quantity_3scenarios')';
    end
        
    input_medium_benders.bigM1 = bigM_scale*max(sum(max(Monthly_Weight_3scenarios'))/min(weight) , max(Monthly_Quantity_3scenarios(:)));
    input_medium_benders.bigM2 = bigM_scale*max(sum(max(Monthly_Weight_3scenarios'))/min(weight) , max(Monthly_Quantity_3scenarios(:)));

    input_medium_benders.display_flag = 1;

    input_medium_benders.GRB_flag = 0;

    input_medium_benders.warmstart_flag = 0;
    
    input_medium_benders.recompute_flag = 0;

    if input_medium_benders.recompute_flag == 1
        
        input_medium_benders.recompute_sample_mode = 1;
        input_medium_benders.recompute_disruption_sample_flag = 0;
        input_medium_benders.recompute_demand_sample_flag = 1;

        input_medium_benders.recompute_disruption_samplesize = 500;
        input_medium_benders.recompute_demand_samplesize = 1000;

    elseif input_medium_benders.recompute_flag == 2
        
        input_medium_benders.recompute_sample_mode = 2;
        input_medium_benders.recompute_disruption_sample_flag = 1;
        input_medium_benders.recompute_demand_sample_flag = 1;

        input_medium_benders.recompute_disruption_samplesize_eval = 100;
        input_medium_benders.recompute_demand_samplesize_eval = 100; 
        input_medium_benders.recompute_disruption_samplesize_finaleval = 500;
        input_medium_benders.recompute_demand_samplesize_finaleval = 1000;

        input_medium_benders.recompute_sgd_Maxsteps = 1e6;

    end

    %% Try unregularized + sampled Benders
    input_medium_benders.regularize_flag = 0;
    
    input_medium_benders.D_scenarios = demand_data_medium_benders;
    input_medium_benders.s_scenarios = failure_data_medium_benders;
    input_medium_benders.prob_scenarios = disruption_demand_joint_prob_medium_benders;
    input_medium_benders.S = size(input_medium_benders.D_scenarios,2);
    
    output_medium_benders_unreg_sampled = U3DP_MIP_Benders(input_medium_benders);

    OUTPUT_MEDIUM_BENDERS_UNREG_SAMPLED{nn} = output_medium_benders_unreg_sampled;


end


%% Compute the optimality gap when Benders is stopped at BOE_total_time
Benders_gap_at_stopping_time = zeros(1, 55);
Benders_time_vs_n = zeros(1, 55);

for nn = 3:55
    Benders_gap_at_stopping_time(nn) = OUTPUT_MEDIUM_BENDERS_UNREG_SAMPLED{nn}.relative_best_gap * 100;
    Benders_time_vs_n(nn) = OUTPUT_MEDIUM_BENDERS_UNREG_SAMPLED{nn}.solving_time;
end

time_sofar = 0;

%% Recompute the MIO objective value using SGD for better accuracy
for nn = 3:55
    fprintf("Recomputing Case %d ", nn)
    
    startTime = clock;

    num_suppliers = nn;
    supplier_subset_idx = 1:nn; % Select first nn suppliers
    
    % Extract data for selected suppliers
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

    %% Compute costs for different backup strategies
    % No backup
    input_medium_no3dp = struct('n', num_suppliers, 'v', v, 'h', h, ...
        'p', p_medium, 'yield_loss_rate', yield_loss_rate_medium, ...
        'Demand_atoms', Monthly_Quantity_3scenarios, ...
        'Demand_prob', Demand_Probability_3scenarios, ...
        'Demand_mean', mean_demand_3scenarios, 'TM_flag', 0);
    output_medium_no3dp = Cost_No3DP_or_TM(input_medium_no3dp);
    
    % 3DP backup (unlimited capacity)
    input_medium_3DP_infK = input_medium_no3dp;
    input_medium_3DP_infK.v = c_3DP;
    output_medium_3DP_infK = Cost_No3DP_or_TM(input_medium_3DP_infK);
    
    % TM backup
    TM_retainer_ratio = 0.8;
    C_TM = TM_retainer_ratio * c_source .* mean_demand_3scenarios;
    input_medium_TM = input_medium_no3dp;
    input_medium_TM.TM_flag = 1;
    input_medium_TM.c_TM = c_TM; 
    input_medium_TM.C_TM = C_TM; 
    output_medium_TM = Cost_No3DP_or_TM(input_medium_TM);

    %% Prepare input for Benders
    bigM_scale = 1e2;
    input_medium = struct('n', num_suppliers, 'c_3DP', c_3DP, 'v', v, ...
        'h', h, 'weight', weight, 'mean_demand', mean_demand_3scenarios, ...
        'cost_of_3dp_per_machine_month', cost_of_3dp_per_machine_month, ...
        'speed_per_machine_month', speed_per_machine_month, ...
        'num_suppliers', num_suppliers, 'num_scenarios', num_scenarios, ...
        'Monthly_Quantity', Monthly_Quantity_3scenarios, ...
        'Monthly_Weight', Monthly_Weight_3scenarios, ...
        'Demand_Probability', Demand_Probability_3scenarios, ...
        'TM_cost', output_medium_TM.TM_cost, ...
        'nobackup_cost', output_medium_no3dp.opt_val, ...
        'bigM', bigM_scale * max(sum(max(Monthly_Weight_3scenarios')) / min(weight), ...
        max(Monthly_Quantity_3scenarios(:))), ...
        'q_ub', output_medium_no3dp.opt_q, 'q_lb', output_medium_3DP_infK.opt_q, ...
        'p', p_medium, 'yield_loss_rate', yield_loss_rate_medium);

    input_medium_benders = input_medium;

    % Compute 3DP capacity and cost
    K_3DP_medium = capacity_percentage * sum(max(Monthly_Weight_3scenarios'));
    C_3DP_medium = (K_3DP_medium ./ speed_per_machine_month)' * cost_of_3dp_per_machine_month;
    input_medium_benders.K_3DP = K_3DP_medium;
    input_medium_benders.C_3DP = C_3DP_medium;

    input_medium_benders.bigM1 = bigM_scale * max(sum(max(Monthly_Weight_3scenarios')) / min(weight), max(Monthly_Quantity_3scenarios(:)));
    input_medium_benders.bigM2 = input_medium_benders.bigM1;

    %% Set up recomputation
    input_medium_benders.x_best = OUTPUT_MEDIUM_BENDERS_UNREG_SAMPLED{nn}.x_best;
    input_medium_benders.q_best = OUTPUT_MEDIUM_BENDERS_UNREG_SAMPLED{nn}.q_best;
    input_medium_benders.recompute_sgd_Maxsteps = 5e5;
    input_medium_benders.auto_recompute = 1;

    output_medium_benders_recompute = recompute_for_MIP_Benders(input_medium_benders);
    OUTPUT_MEDIUM_BENDERS_RECOMPUTE{1, nn} = output_medium_benders_recompute;

    endTime = clock;
    time_sofar = time_sofar + etime(endTime, startTime);

    fprintf("Time so far: %3.2f seconds\n", time_sofar)
end


%% Compute refined optimality gap for Benders method
Benders_gap_refined = zeros(1, 55);

for nn = 3:55
    num_suppliers = nn;
    supplier_subset_idx = 1:nn; % Select first nn suppliers
    
    % Extract weight data for selected suppliers
    Monthly_Weight_3scenarios = Monthly_Weight_3scenarios_all(supplier_subset_idx, :);

    % Compute 3DP capacity and cost
    K_3DP_medium = capacity_percentage * sum(max(Monthly_Weight_3scenarios'));
    C_3DP_medium = (K_3DP_medium ./ speed_per_machine_month)' * cost_of_3dp_per_machine_month;

    % Compute refined gap for Benders
    Benders_gap_refined(nn) = ...
        (OUTPUT_MEDIUM_BENDERS_RECOMPUTE{1, nn}.TOTAL_COST_NONZERO(1, 1) - C_3DP_medium(1, 1) ...
        - OUTPUT_MEDIUM_BENDERS_UNREG_SAMPLED{1, nn}.LB(end)) ...
        / abs(OUTPUT_MEDIUM_BENDERS_UNREG_SAMPLED{1, nn}.LB(end)) * 100;
end

% Retain original stopping time gaps for the first 9 cases
Benders_gap_refined(1:9) = Benders_gap_at_stopping_time(1:9);

%% Compute the gap for the BOE solution
BOE_gap = zeros(1, 55);

for nn = 3:55
    num_suppliers = nn;
    supplier_subset_idx = 1:nn; % Select first nn suppliers

    % Extract weight data for selected suppliers
    Monthly_Weight_3scenarios = Monthly_Weight_3scenarios_all(supplier_subset_idx, :);

    % Compute 3DP capacity and cost
    K_3DP_medium = capacity_percentage * sum(max(Monthly_Weight_3scenarios'));
    C_3DP_medium = (K_3DP_medium ./ speed_per_machine_month)' * cost_of_3dp_per_machine_month;

    % Compute BOE gap
    BOE_gap(nn) = ...
        (OUTPUT_BOE{1, nn}.TOTAL_COST_NONZERO(1, 1) - C_3DP_medium(1, 1) ...
        - OUTPUT_MEDIUM_BENDERS_UNREG_SAMPLED{1, nn}.LB(end)) ...
        / abs(OUTPUT_MEDIUM_BENDERS_UNREG_SAMPLED{1, nn}.LB(end)) * 100;
end


%% Define directory paths
data_dir = 'Experiment_Data/Compare_MIP_Time_GRB_Benders_BoE';

% Create directories if they do not exist
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end

%% Save computed results
save(fullfile(data_dir, 'data.mat'));

%% Plot and compare optimality gaps
figure;
plot(3:55, BOE_gap(3:55), 'LineWidth', 5);
hold on;
plot(3:55, Benders_gap_refined(3:55), '-o', 'LineWidth', 1.5);
xlim([3, 55]);
grid on;

legend({'SuperMod. Approx.', 'SAA-MIO-Benders'}, 'FontSize', 20, 'Location', 'southeast');
ytickformat('percentage');

xlabel('$n$', 'Interpreter', 'latex', 'FontSize', 20);
ylabel('Optimality Gap', 'FontSize', 20);

% Save the plot
saveas(gcf, fullfile(data_dir, 'opt_gap_comparison.pdf'));
close(gcf);

%% Plot and compare computation times
figure;
plot(3:55, BOE_time_vs_n(3:55) / BOE_total_time * 100, 'LineWidth', 2);
hold on;
plot(3:55, min(BOE_total_time, Benders_time_vs_n(3:55)) / BOE_total_time * 100, 'LineWidth', 2);
xlim([3, 55]);
ylim([0, 120]);
grid on;

legend({'SuperMod. Approx.', 'SAA-MIO-Benders'}, 'FontSize', 15, 'Location', 'northeast');
ytickformat('percentage');

xlabel('$n$', 'Interpreter', 'latex', 'FontSize', 20);
ylabel('Time (% of Total Heuristic Time)', 'FontSize', 20);

% Save the plot
saveas(gcf, fullfile(data_dir, 'time_comparison.pdf'));
close(gcf);

%% Save data for analysis
% Optimality gap data
csvwrite(fullfile(data_dir, 'optimality_gap_data.csv'), ...
    [(3:55)', BOE_gap(3:55)', Benders_gap_refined(3:55)']);

% Computation time comparison data
csvwrite(fullfile(data_dir, 'time_comparison_data.csv'), ...
    [(3:55)', (BOE_time_vs_n(3:55) / BOE_total_time * 100)', ...
    (min(BOE_total_time, Benders_time_vs_n(3:55)) / BOE_total_time * 100)']);

