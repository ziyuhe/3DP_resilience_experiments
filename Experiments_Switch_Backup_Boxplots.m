% =========================================================================
% Script Name:     Experiments_Switch_Backup_Boxplots.m  
% Author:          Ziyu He  
% Date:            02/01/2025  
%  
% Description:  
%   This script evaluates the impact of introducing 3DP backup by analyzing:  
%     - The percentage of unprotected products, 3DP-protected products, and backup switches.  
%     - Cost savings and demand shortfall reductions provided by both the full and naive 3DP strategies.  
%  
%   The number of suppliers (n) is varied among 15, 30, and 45.  
%  
% Methodology:  
%   - For each num_suppliers value, multiple supplier subsets are sampled.  
%   - BoE local search is used to approximate the optimal 3DP backup selection (Full 3DP strategy).  
%   - Stochastic Gradient Descent (SGD) is used to evaluate the cost of the naive 3DP strategy.  
%   - For n > 40, BoE utilizes Sample Average Approximation (SAA) to ensure feasibility.  
%  
% Outputs:  
%   - Percentage of 3DP backup, unprotected products, and backup switches.  
%   - Cost savings and demand shortfalls under different problem sizes and strategies.  
%   - Intermediate data is saved, and comparative plots are generated.  
%  
% Notes:  
%   - For larger instances (n > 40), BoE solutions are approximated using Sample Average Approximation (SAA) to improve tractability.  
%   - "Dedicated Backup" refers to "TM" (Traditional Manufacturing).  
%   - 3DP capacity represents the total monthly output of printing materials.  
%   - Weight measurements are initially in grams but converted to kilograms for cost scaling.  
%  
% Adjustments:  
%   - "weight_all" is divided by 1000 to reflect per-unit product weight.  
%   - "speed_per_machine_month" is divided by 1000 to account for material consumption per printer per month.  
%   - "Monthly_Weight_3scenarios_all" is divided by 1000 to scale demand scenarios in weight.  
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










%% Experiment: Understanding the percentage of products switched
%  - Vary the number of suppliers (num_suppliers)
%  - For each num_suppliers, sample multiple subsets of suppliers
%  - For each sampled subset on a grid of K values, run:
%       - BoE local search to approximate the optimal 3DP backup selection (At the optimal K, determine the 3DP set)
%       - Use SGD to evaluatate the naive policy of only backing up unprotected products

Random_Suppliers_Num = 50;

% 3DP capacity as a percentage of max demand
capacity_3dp_percentage = [1e-2, 1e-1 * [1:9], 1:2:25, 30:5:50, 75, 100] / 100;

OUTPUT_MEDIUM_BOE = {};
A_t_BOE_SET = {};
X_BOE_SET = {};
TOTAL_COST_SET = {};
X_BOE_OPT_SET = {};
SWITCHED_BOE_SET = {};
SWITCHED_BOE_OPT_SET = {};

SUPPLIER_SUBSET_IDX = {};
TM_BACKUPSET = {};

Q_SP_NAIVE_POLICY_SET = {};
Ob_NAIVEPOLICY_SET = {};
TOTAL_COST_NAIVEPOLICY_SET = {};


NUM_SUPPLIERS_SET = [15,30,45];

for num_suppliers_case = 1 : length(NUM_SUPPLIERS_SET)

    nn = NUM_SUPPLIERS_SET(num_suppliers_case);


    for random_suppliers_num = 1 : Random_Suppliers_Num

        startTime = clock;
    
        display("---------------------------------------------------------------------\n")
        fprintf("WE ARE CURRENTLY WORKING ON: %d suppliers, case #%d \n", nn, random_suppliers_num)
        display("---------------------------------------------------------------------\n")
    
        
        %% Randomly sample a subset of suppliers
        num_suppliers = nn;
        supplier_subset_idx = false(num_suppliers_all, 1);
        supplier_subset_idx(randperm(num_suppliers_all, num_suppliers)) = true;
        
        SUPPLIER_SUBSET_IDX{num_suppliers_case, random_suppliers_num} = supplier_subset_idx;
        
        % Extract relevant data for the sampled supplier subset
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
        
       

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% GET THE SINGLE SYSTEM (TM ONLY)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
        TM_BACKUPSET{num_suppliers_case, random_suppliers_num} = TM_backup_set;
        COST_TMONLY{num_suppliers_case, random_suppliers_num} = sum(output_medium_TM.TM_cost(TM_backup_set))+sum(output_medium_no3dp.opt_val(logical(nobackup_set)));






        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% GET THE FULL STRATEGY (BY BOE)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        
        %% Sample some data for BoE Submod Max SAA
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
        
        
        output_preprocess_medium_sampled = Data_prep_for_MIP(input_preprocess_medium_sampled);
        
        disruption_demand_joint_prob_medium_sampled = output_preprocess_medium_sampled.disruption_demand_joint_prob;
        failure_data_medium_sampled = output_preprocess_medium_sampled.failure_data;
        demand_data_medium_sampled = output_preprocess_medium_sampled.demand_data;
        
        input_boe.disruption_demand_joint_prob = disruption_demand_joint_prob_medium_sampled;
        input_boe.failure_data = failure_data_medium_sampled;
        input_boe.demand_data = demand_data_medium_sampled;
        
        
        
        
        
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
            
            input_boe.recompute_flag = 1;
        
            input_boe.recompute_sample_mode = 1;
            input_boe.recompute_disruption_sample_flag = 0;
            input_boe.recompute_demand_sample_flag = 0;
            
            input_boe.recompute_disruption_samplesize_eval = 1000;
            input_boe.recompute_demand_samplesize_eval = 500; 
            input_boe.recompute_disruption_samplesize_finaleval = 1000;
            input_boe.recompute_demand_samplesize_finaleval = 500;
        
            input_boe.recompute_sgd_Maxsteps = 2e5;
            
            if k == 1
                input_boe.A_init = [];
            else
                input_boe.A_init = OUTPUT_MEDIUM_BOE{num_suppliers_case, random_suppliers_num, k-1}.A_t;
            end
    
            if nn <= 10
                output_boe = BoE_Approx_Max_Submod_Exact(input_boe);
            else
                output_boe = BoE_Approx_Max_Submod_SAA(input_boe);
            end
        
            OUTPUT_MEDIUM_BOE{num_suppliers_case, random_suppliers_num, k} = output_boe; 
        
            disp("TO NEXT ONE! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ")
        
        end
        
        %% Store results
        %% Post-Processing: Comparing BoE results to the "No 3DP" strategy
        %  - Retain the BoE-selected 3DP set only if it reduces the total cost compared to the no-3DP strategy.
        %  - Define key variables:
        %    * A_t_BOE: The 3DP set obtained before post-processing (compared against no 3DP selection).
        %    * X_BOE: The final 3DP set after post-processing, evaluated under different C3DP cost coefficients.
        %    * TOTAL_COST: The total system cost after post-processing.
        %
        %  Computation of TOTAL_COST (for a fixed K):
        %  1. Use "A_t_BOE" from the local search method to define the 3DP set.
        %  2. Solve the fixed 3DP set problem using SGD (Stochastic Gradient Descent).
        %  3. Evaluate U3DP using a larger sample size based on the SGD solution.
        %  4. Compare the resulting cost against the no-3DP case.
        %
        %  Note: Since TOTAL_COST is evaluated under different K values, it may be computed on different sample sets.

        X_BOE = {}; 
        SWITCHED_BOE = {};
        for i = 1 : length(speed_per_machine_month)  
            for j = 1 : length(cost_of_3dp_per_machine_month)
                X_BOE{i,j} = zeros(num_suppliers,length(capacity_3dp_percentage));
                SWITCHED_BOE{i,j} = zeros(num_suppliers,length(capacity_3dp_percentage));
            end
        end
        TOTAL_COST = {};
        A_t_BOE = [];
        for k = 1 : length(capacity_3dp_percentage)
        
            A_t_BOE(OUTPUT_MEDIUM_BOE{num_suppliers_case, random_suppliers_num,k}.A_t, k) = 1;
        
            for i = 1 : length(speed_per_machine_month)  
                for j = 1 : length(cost_of_3dp_per_machine_month)
        
                    TOTAL_COST{i,j}(k) = OUTPUT_MEDIUM_BOE{num_suppliers_case, random_suppliers_num,k}.TOTAL_COST(i,j);
                    X_BOE{i,j}(:, k) = OUTPUT_MEDIUM_BOE{num_suppliers_case, random_suppliers_num,k}.X_FINAL{i,j};
                    if ~isinf(sum(X_BOE{i,j}(:,k)))
                        SWITCHED_BOE{i,j}(:,k) = TM_backup_set - (1-X_BOE{i,j}(:,k));
                    end
        
                end
            end
        
        end
        
    
        A_t_BOE_SET{num_suppliers_case, random_suppliers_num} = A_t_BOE;
        X_BOE_SET{num_suppliers_case, random_suppliers_num} = X_BOE;
        SWITCHED_BOE_SET{num_suppliers_case, random_suppliers_num} = SWITCHED_BOE;
        TOTAL_COST_SET{num_suppliers_case, random_suppliers_num} = TOTAL_COST;
    
        %% "X_BOE_OPT" is the 3DP set at the optimal K
        %% "SWITCHED_SET_OPT" is the set of products whos backup swithced after intro of 3DP
        X_BOE_OPT = {}; 
        SWITCHED_BOE_OPT = {};
        for i = 1 : length(speed_per_machine_month)  
            for j = 1 : length(cost_of_3dp_per_machine_month)

                [~, iii] = min(TOTAL_COST{i,j});
                X_BOE_OPT{i,j} = X_BOE{i,j}(:,iii);
                SWITCHED_BOE_OPT{i,j} = SWITCHED_BOE{i,j}(:,iii);

            end
        end
        
        X_BOE_OPT_SET{num_suppliers_case, random_suppliers_num} = X_BOE_OPT;
        SWITCHED_BOE_OPT_SET{num_suppliers_case, random_suppliers_num} = SWITCHED_BOE_OPT;




        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% GET THE NAIVE STRATEGY
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %% Run naive policy for the benchmark case (varying C3DP)
        num_suppliers = sum(nobackup_set);
        supplier_subset_idx_naive = logical(nobackup_set);
        Monthly_Weight_3scenarios = Monthly_Weight_3scenarios(supplier_subset_idx_naive, :);
        Monthly_Quantity_3scenarios = Monthly_Quantity_3scenarios(supplier_subset_idx_naive, :);
        Demand_Probability_3scenarios = Demand_Probability_3scenarios(supplier_subset_idx_naive, :);
        mean_demand_3scenarios = mean_demand_3scenarios(supplier_subset_idx_naive);
               
        c_source = c_source(supplier_subset_idx_naive);        
        c_3DP = c_3DP(supplier_subset_idx_naive); 
        c_TM = c_TM(supplier_subset_idx_naive);   
        v = v(supplier_subset_idx_naive);         
        h = h(supplier_subset_idx_naive);  
        weight = weight(supplier_subset_idx_naive);
        
        input_medium_no3dp_naive.n = num_suppliers;
        input_medium_no3dp_naive.v = v;
        input_medium_no3dp_naive.h = h;
        input_medium_no3dp_naive.p = p_medium;
        input_medium_no3dp_naive.yield_loss_rate = yield_loss_rate_medium;
        input_medium_no3dp_naive.Demand_atoms = Monthly_Quantity_3scenarios;
        input_medium_no3dp_naive.Demand_prob = Demand_Probability_3scenarios;
        input_medium_no3dp_naive.Demand_mean = mean_demand_3scenarios;
        input_medium_no3dp_naive.TM_flag = 0;
        output_medium_no3dp_naive = Cost_No3DP_or_TM(input_medium_no3dp_naive);
        
        % All backed-up by 3DP (inf. capacity, we only care about the solution)
        input_medium_3DP_infK_naive = input_medium_no3dp_naive;
        input_medium_3DP_infK_naive.v = c_3DP;
        output_medium_3DP_infK_naive = Cost_No3DP_or_TM(input_medium_3DP_infK_naive);
        
        
        %% First sample a larger set just for objective evaluation during SGD

        if num_suppliers <= 12
            sample_mode = 2;
            disruption_sample_flag = 0;
            demand_sample_flag = 1;
        else
            sample_mode = 2;
            disruption_sample_flag = 1;
            demand_sample_flag = 1;
        end

        input_preprocess_medium.num_suppliers = num_suppliers;
        input_preprocess_medium.num_scenarios = num_scenarios;
        input_preprocess_medium.yield_loss_rate = yield_loss_rate_medium;
        input_preprocess_medium.p_disrupt = p_medium;
        input_preprocess_medium.Monthly_Quantity = Monthly_Quantity_3scenarios;
        input_preprocess_medium.Demand_Probability = Demand_Probability_3scenarios;
        
        input_preprocess_medium.sample_mode = sample_mode;
        input_preprocess_medium.disruption_sample_flag = disruption_sample_flag; % We don't sample disruption (keep all combinations)
        input_preprocess_medium.demand_sample_flag = demand_sample_flag;     % For each disruption scenario, sample a fixed number of demand combos
        
        Disruption_sample_size_eval = 2^12;
        Demand_sample_size_eval = 100;
        
        if input_preprocess_medium.sample_mode == 1
            if input_preprocess_medium.disruption_sample_flag == 1
                input_preprocess_medium.disruption_samplesize = 10;
            end
            if input_preprocess_medium.demand_sample_flag == 1
                input_preprocess_medium.demand_samplesize = Demand_sample_size_eval;
            end
        else
            if input_preprocess_medium.disruption_sample_flag == 1
                input_preprocess_medium.disruption_samplesize = Disruption_sample_size_eval;
            end
            input_preprocess_medium.demand_num_per_disruption = Demand_sample_size_eval;
        end
        
        output_preprocess_medium = Data_prep_for_MIP(input_preprocess_medium);
        
        disruption_demand_joint_prob_medium_SGD_inprocess_eval = output_preprocess_medium.disruption_demand_joint_prob;
        failure_data_medium_SGD_inprocess_eval = output_preprocess_medium.failure_data;
        demand_data_medium_SGD_inprocess_eval = output_preprocess_medium.demand_data;
        
        %% Second sample a even larger set just for final objective evaluation
        input_preprocess_medium.num_suppliers = num_suppliers;
        input_preprocess_medium.num_scenarios = num_scenarios;
        input_preprocess_medium.yield_loss_rate = yield_loss_rate_medium;
        input_preprocess_medium.p_disrupt = p_medium;
        input_preprocess_medium.Monthly_Quantity = Monthly_Quantity_3scenarios;
        input_preprocess_medium.Demand_Probability = Demand_Probability_3scenarios;
        
        input_preprocess_medium.sample_mode = sample_mode;
        input_preprocess_medium.disruption_sample_flag = disruption_sample_flag; % We don't sample disruption (keep all combinations)
        input_preprocess_medium.demand_sample_flag = demand_sample_flag;     % For each disruption scenario, sample a fixed number of demand combos
        
        Disruption_sample_size_eval = 2^12;
        Demand_sample_size_eval = 100;
        
        if input_preprocess_medium.sample_mode == 1
        
            if input_preprocess_medium.disruption_sample_flag == 1
                input_preprocess_medium.disruption_samplesize = 10;
            end
            if input_preprocess_medium.demand_sample_flag == 1
                input_preprocess_medium.demand_samplesize = Demand_sample_size_eval;
            end
        
        else
        
            if input_preprocess_medium.disruption_sample_flag == 1
                input_preprocess_medium.disruption_samplesize = Disruption_sample_size_eval;
            end
            input_preprocess_medium.demand_num_per_disruption = Demand_sample_size_eval;
        
        end
        
        output_preprocess_medium = Data_prep_for_MIP(input_preprocess_medium);
        
        disruption_demand_joint_prob_medium_SGD_finaleval = output_preprocess_medium.disruption_demand_joint_prob;
        failure_data_medium_SGD_finaleval = output_preprocess_medium.failure_data;
        demand_data_medium_SGD_finaleval = output_preprocess_medium.demand_data;
        
        %% Run SGD
        input_sgd.n = num_suppliers;
        input_sgd.c_3DP = c_3DP;
        input_sgd.v = v;
        input_sgd.h = h;
        input_sgd.weight = weight;
        input_sgd.mean_demand = mean_demand_3scenarios;
        input_sgd.q_ub = output_medium_no3dp_naive.opt_q;    % HERE WE ADD BOUNDS FOR 1ST STAGE DECISIONS
        input_sgd.q_lb = output_medium_3DP_infK_naive.opt_q; % HERE WE ADD BOUNDS FOR 1ST STAGE DECISIONS
        
        input_sgd.prob_scenarios = disruption_demand_joint_prob_medium_SGD_inprocess_eval;
        input_sgd.D_scenarios = demand_data_medium_SGD_inprocess_eval;
        input_sgd.s_scenarios = failure_data_medium_SGD_inprocess_eval;
        input_sgd.S = size(demand_data_medium_SGD_inprocess_eval,2);
        
        
        ObJ_BENCHMARK_NAIVEPOLICY = [];
        Q_SP_NAIVE_POLICY = [];
        
        for k = 1 : length(capacity_3dp_percentage)
            fprintf("----------------------------------------------\n")
            fprintf("(%d suppliers, case #%d) %3.2f Percent of Max Yield Shortfall \n", nn, random_suppliers_num, 100*capacity_3dp_percentage(k))
            fprintf("----------------------------------------------\n\n")
        
            capacity_percentage = capacity_3dp_percentage(k); 
            % Small disruption
            K_3DP_medium = capacity_percentage*sum(max(Monthly_Weight_3scenarios'));
            C_3DP_medium = (K_3DP_medium./speed_per_machine_month)'*cost_of_3dp_per_machine_month;
            input_sgd.K_3DP = K_3DP_medium;
            input_sgd.C_3DP = C_3DP_medium;
        
            input_sgd.yield_loss_rate = yield_loss_rate_medium;
            input_sgd.p_disrupt = p_medium;
            input_sgd.Monthly_Quantity = Monthly_Quantity_3scenarios;
            input_sgd.Demand_Probability = Demand_Probability_3scenarios;
            input_sgd.num_scenarios = num_scenarios;
        
            % Max SGD steps
            input_sgd.Max_Steps = 5e5;
        
            % Steps for evaluating obj val
            input_sgd.show_objeval = 0;
            input_sgd.objeval_steps = input_sgd.Max_Steps+1;
        
            % When we evaluate the obj val, do we use the averaged solution ("input.ave_flag = 1" <=> do average)
            % We take average between floor(t*ave_ratio)+1 to t
            input_sgd.ave_flag = 1;
            input_sgd.ave_ratio = 1/2;
            
            % Sample disruptions ("disrupt_sample_flag == 1" <=> only sample one disruption combo per step)
            input_sgd.disrupt_sample_flag = 1;
            input_sgd.disruption_prob = output_preprocess_medium.disruption_prob;
            input_sgd.failure_combinations = output_preprocess_medium.failure_combinations;
            
            % Take an the SAA solution as initilization
            input_sgd.q_init = unifrnd(input_sgd.q_lb, input_sgd.q_ub);
        
            % Step size mode ("stepsize_flag == 0": constant; "stepsize_flag == 0": 1/sqrt(t))
            input_sgd.stepsize_flag = 1; 
            M_subgrad_l2 = norm(max(max(max(max((v-c_3DP)./weight)*weight+c_3DP, v), h), c_3DP), 2); 
            D_to_optsol = norm(max(abs(input_sgd.q_init-input_sgd.q_ub),abs(input_sgd.q_init-input_sgd.q_lb)), 2);
            tt = 5e-1;
            if input_sgd.stepsize_flag == 0
                % constant stepsize
                input_sgd.stepsize_const = tt*D_to_optsol/(M_subgrad_l2*sqrt(input_sgd.Max_Steps));
            else
                % 1/sqrt(t) stepsize
                input_sgd.stepsize_const = tt*(D_to_optsol/M_subgrad_l2);
            end
        
            % If we have benchmark then we use it as stopping rule
            % Two stopping rules when we have benchmark:
            %   - when we haven improved more than "stop_threshold_multisteps" in the past "stop_interval" steps
            %   - when we have reached "stop_threshold_singlestep"
            input_sgd.benchmark_flag = 0;
            if input_sgd.benchmark_flag == 1
                input_sgd.benchmark_optval = OUTPUT_FIXED_SUPPSELECT{k}.opt_val;
                input_sgd.stop_interval = 5;
                input_sgd.stop_threshold_multisteps = 5e-7;
                input_sgd.stop_threshold_singlestep = 5e-5;
            else
                input_sgd.stop_interval = 5;
                input_sgd.stop_threshold_multisteps = 5e-6;
            end
            
            % If we sample one-at-a-time (input_sgd.sample_ahead == 0) or sample ahead (== 1)
            % This turns out to be crucial: 
            %   - sample everything ahead can safe significant time
            %   - but it might run into storage issues
            % So for large scale problems, we can sample a big batch every "input.sample_ahead_batchsize" steps
            input_sgd.sample_ahead = 1;
            if input_sgd.sample_ahead == 1
                input_sgd.sample_ahead_batchsize = 1e6;
            end
        
            input_sgd.display_interval = 1e5; % Output the progress every "display_interval" steps (NO OBJ EVAL!!!)
        
            output_sgd = U3DP_SGD_fixed_suppselect_fixed_K(input_sgd);
        
            Q_SP_NAIVE_POLICY(:,k) = output_sgd.q_ave_final;

            % Evaluate the objective value of the SGD solution
            input_objeval.q_eval = output_sgd.q_ave_final;
            input_objeval.n = num_suppliers;
            input_objeval.c_3DP = c_3DP;
            input_objeval.v = v;
            input_objeval.h = h;
            input_objeval.weight = weight;
            input_objeval.mean_demand = mean_demand_3scenarios;
            input_objeval.K_3DP = K_3DP_medium;
            input_objeval.prob_scenarios = disruption_demand_joint_prob_medium_SGD_finaleval;
            input_objeval.D_scenarios = demand_data_medium_SGD_finaleval;
            input_objeval.s_scenarios = failure_data_medium_SGD_finaleval;
            input_objeval.S = size(demand_data_medium_SGD_finaleval,2);
            input_objeval.display_flag = 1;
            input_objeval.display_interval = 1e5;
            obj_fullinfo_sgd = U3DP_objeval_fixed_suppselect(input_objeval);
            ObJ_NAIVEPOLICY(k) = obj_fullinfo_sgd;
        
        end
        
        
        TOTAL_COST_NAIVEPOLICY = {};
        for k = 1 : length(capacity_3dp_percentage)
        
            capacity_percentage = capacity_3dp_percentage(k);
            K_3DP_medium = capacity_percentage*sum(max(Monthly_Weight_3scenarios'));
        
            for i = 1 : length(speed_per_machine_month)  
                for j = 1 : length(cost_of_3dp_per_machine_month)
        
                    TOTAL_COST_NAIVEPOLICY{i,j}(k) = ObJ_NAIVEPOLICY(k) - sum(mean_demand_3scenarios.*v) ...
                                                        + K_3DP_medium/speed_per_machine_month(i)*cost_of_3dp_per_machine_month(j) ...
                                                        + sum(output_medium_TM.TM_cost(TM_backup_set));
                end
            end
        end


        Q_SP_NAIVE_POLICY_SET{num_suppliers_case, random_suppliers_num} = Q_SP_NAIVE_POLICY;
        ObJ_NAIVEPOLICY_SET{num_suppliers_case, random_suppliers_num} = ObJ_NAIVEPOLICY;
        TOTAL_COST_NAIVEPOLICY_SET{num_suppliers_case, random_suppliers_num} = TOTAL_COST_NAIVEPOLICY;

        
        endTime = clock;

        timesofar = timesofar + etime(endTime,startTime);

        fprintf("****** TIME: %3.2f ***** \n\n", timesofar)

    end

end


%% Summarize the ratio of 3DP backup, the ratio of dedicated backup that are switched
Num_3DP_backup = {};
Num_switched_backup = {};
Num_bare = {};
Ratio_3DP_backup = {};
Ratio_switched_backup = {};
Ratio_bare = {};

for num_suppliers_case = 1 : length(NUM_SUPPLIERS_SET)

    nn = NUM_SUPPLIERS_SET(num_suppliers_case); 

    for random_suppliers_num = 1 : Random_Suppliers_Num

        Num_bare{num_suppliers_case}(random_suppliers_num) = nn - sum(TM_BACKUPSET{num_suppliers_case, random_suppliers_num});
        Ratio_bare{num_suppliers_case}(random_suppliers_num) = (nn - sum(TM_BACKUPSET{num_suppliers_case, random_suppliers_num}))/nn;        
    
        for i = 1 : length(speed_per_machine_month)  
            for j = 1 : length(cost_of_3dp_per_machine_month)
                
                Num_3DP_backup{num_suppliers_case, i, j}(random_suppliers_num) = sum(X_BOE_OPT_SET{num_suppliers_case, random_suppliers_num}{i,j});
                Ratio_3DP_backup{num_suppliers_case, i, j}(random_suppliers_num) = Num_3DP_backup{num_suppliers_case, i, j}(random_suppliers_num)/nn;


                Num_switched_backup{num_suppliers_case, i, j}(random_suppliers_num) = sum(SWITCHED_BOE_OPT_SET{num_suppliers_case, random_suppliers_num}{i,j});
                Ratio_switched_backup{num_suppliers_case, i, j}(random_suppliers_num) = ...
                    Num_switched_backup{num_suppliers_case, i, j}(random_suppliers_num) / sum(TM_BACKUPSET{num_suppliers_case, random_suppliers_num});
                
    
            end
        end

        
    end
end

%% Cost savings
Full_Cost_Savings = {};
Naive_Cost_Savings = {};

for num_suppliers_case = 1 : length(NUM_SUPPLIERS_SET)
    for random_suppliers_num = 1 : Random_Suppliers_Num     
    
        for i = 1 : length(speed_per_machine_month)  
            for j = 1 : length(cost_of_3dp_per_machine_month)

                Full_Cost_Savings{num_suppliers_case,i,j}(random_suppliers_num) = 100*max(COST_TMONLY{num_suppliers_case, random_suppliers_num} - TOTAL_COST_SET{num_suppliers_case, random_suppliers_num}{i,j})/abs(COST_TMONLY{num_suppliers_case, random_suppliers_num});
                Naive_Cost_Savings{num_suppliers_case,i,j}(random_suppliers_num) = 100*max(COST_TMONLY{num_suppliers_case, random_suppliers_num} - TOTAL_COST_NAIVEPOLICY_SET{num_suppliers_case, random_suppliers_num}{i,j})/abs(COST_TMONLY{num_suppliers_case, random_suppliers_num}) ;


            end
        end

    end
end

directory = "Experiment_Data/Switch_Backup_vs_n";
if ~exist(directory, 'dir')
    mkdir(directory);
end
save(fullfile(directory, "data_n_less_than_55.mat"));









%% BOX PLOTS
%  - Generate sample plots using MATLAB.
%  - The final plots in the paper are created using Python from CSV files saved here.

NUM_SUPPLIERS_SET = [15, 30, 45];

%% Plot the percentage of "TM Backed-up" vs. "Number of Suppliers" under three cases:
%     1. SINGLE system (TM-only, no 3DP)
%     2. Naive 3DP policy
%     3. Full 3DP policy
for i = 1:length(speed_per_machine_month)  
    for j = 1:length(cost_of_3dp_per_machine_month)

        figure;
        hold on;

        for num_suppliers_case = 1:length(NUM_SUPPLIERS_SET)       
            % SINGLE system
            boxplot(100 - Ratio_bare{num_suppliers_case}' * 100, ...
                'Positions', NUM_SUPPLIERS_SET(num_suppliers_case) - 3, 'Widths', 2);
            % Naive 3DP policy
            boxplot(100 - Ratio_bare{num_suppliers_case}' * 100, ...
                'Positions', NUM_SUPPLIERS_SET(num_suppliers_case), 'Widths', 2);
            % Full 3DP policy
            boxplot(100 - Ratio_3DP_backup{num_suppliers_case, i, j}' * 100, ...
                'Positions', NUM_SUPPLIERS_SET(num_suppliers_case) + 3, 'Widths', 2);
        end

        % Adjust x-axis properties
        xlim([min(NUM_SUPPLIERS_SET(2:end)) - 10, max(NUM_SUPPLIERS_SET) + 10]); 
        xticks(NUM_SUPPLIERS_SET(2:end)); 
        xticklabels(string(NUM_SUPPLIERS_SET(2:end))); 
        ylabel('Percentage of Bare Products');
        grid on;
        
        % Save figure
        fileName = strcat('Experiment_Data/Switch_Backup_vs_n/Ratio_TMbackup_boxplots', num2str(i), num2str(j), '.pdf');
        saveas(gcf, fileName);
        
        hold off;
        close(gcf);
    end
end

%% Save "TM Backed-up" Data to CSV for External Processing
for num_suppliers_case = 1:length(NUM_SUPPLIERS_SET)
    % Extract data for each strategy
    SINGLE_data = 100 - Ratio_bare{num_suppliers_case}' * 100;
    Naive_data = 100 - Ratio_bare{num_suppliers_case}' * 100;
    Full_data = 100 - Ratio_3DP_backup{num_suppliers_case, i, j}' * 100;

    % Create a table and add the corresponding supplier count
    data_table = table(SINGLE_data, Naive_data, Full_data, ...
        'VariableNames', {'SINGLE', 'Naive', 'Full'});
    data_table.NumSuppliers = repmat(NUM_SUPPLIERS_SET(num_suppliers_case), height(data_table), 1);

    % Save table as a CSV file
    file_name = strcat('Experiment_Data/Switch_Backup_vs_n/Data_NumSuppliers_TMbackup_', num2str(NUM_SUPPLIERS_SET(num_suppliers_case)), '.csv');
    writetable(data_table, file_name);
end

%% Plot Cost Savings Relative to the SINGLE System (No 3DP)
%     1. SINGLE system (Baseline, 0 savings)
%     2. Naive 3DP policy
%     3. Full 3DP policy
for i = 1:length(speed_per_machine_month)  
    for j = 1:length(cost_of_3dp_per_machine_month)

        figure;
        hold on;

        for num_suppliers_case = 1:length(NUM_SUPPLIERS_SET)       
            % SINGLE system (0 cost savings)
            boxplot(0, 'Positions', NUM_SUPPLIERS_SET(num_suppliers_case) - 3, 'Widths', 2);
            % Naive 3DP policy
            boxplot(Naive_Cost_Savings{num_suppliers_case, i, j}, ...
                'Positions', NUM_SUPPLIERS_SET(num_suppliers_case), 'Widths', 2);
            % Full 3DP policy
            boxplot(Full_Cost_Savings{num_suppliers_case, i, j}, ...
                'Positions', NUM_SUPPLIERS_SET(num_suppliers_case) + 3, 'Widths', 2);
        end

        % Adjust x-axis properties
        xlim([min(NUM_SUPPLIERS_SET(2:end)) - 10, max(NUM_SUPPLIERS_SET) + 10]); 
        xticks(NUM_SUPPLIERS_SET(2:end)); 
        xticklabels(string(NUM_SUPPLIERS_SET(2:end))); 
        ylabel('Cost Savings (%)');
        grid on;
        
        % Save figure
        fileName = strcat('Experiment_Data/Switch_Backup_vs_n/Cost_savings_boxplots', num2str(i), num2str(j), '.pdf');
        saveas(gcf, fileName);
        
        hold off;
        close(gcf);
    end
end

%% Save Cost Savings Data to CSV for External Processing
for num_suppliers_case = 2:length(NUM_SUPPLIERS_SET)
    % Extract data for each strategy
    SINGLE_data = 0;  % Baseline cost savings for SINGLE system
    Naive_data = Naive_Cost_Savings{num_suppliers_case, i, j};
    Full_data = Full_Cost_Savings{num_suppliers_case, i, j};

    % Create a table and add the corresponding supplier count
    data_table = table(repmat(SINGLE_data, numel(Naive_data), 1), Naive_data', Full_data', ...
        'VariableNames', {'SINGLE', 'Naive', 'Full'});
    data_table.NumSuppliers = repmat(NUM_SUPPLIERS_SET(num_suppliers_case), height(data_table), 1);

    % Save table as a CSV file
    file_name = strcat('Experiment_Data/Switch_Backup_vs_n/Data_NumSuppliers_costsavings_', num2str(NUM_SUPPLIERS_SET(num_suppliers_case)), '.csv');
    writetable(data_table, file_name);
end










%% Demand Shortfall Calculation for Baseline Case
%  - Evaluate demand shortfall under three different strategies:
%    1. No 3DP (TM-only)
%    2. Naive 3DP policy
%    3. Full 3DP policy
%  - Use the first case of printer cost and speed to determine C3DP.
%  - For each scenario ("num_suppliers_case"), conduct multiple runs indexed by "random_suppliers_num".
%  - Compute both absolute and relative demand shortfall.

i = 1; j = 1;

% Initialize storage for demand shortfall calculations
TMONLY_Demand_shortfall_mean = {};
Naive_Demand_shortfall_mean = {};
Full_Demand_shortfall_mean = {};

TMONLY_Demand_shortfall_mean_relative = {};
Naive_Demand_shortfall_mean_relative = {};
Full_Demand_shortfall_mean_relative = {};

for num_suppliers_case = 1:length(NUM_SUPPLIERS_SET)
    for random_suppliers_num = 1:Random_Suppliers_Num

        fprintf("Processing case %d, iteration %d \n\n", num_suppliers_case, random_suppliers_num)

        % Extract sampled supplier indices for this iteration
        supplier_subset_idx = SUPPLIER_SUBSET_IDX{num_suppliers_case, random_suppliers_num};

        % Extract relevant data for the selected suppliers
        Monthly_Weight_3scenarios_tmp = Monthly_Weight_3scenarios_all(supplier_subset_idx, :);
        Monthly_Quantity_3scenarios_tmp = Monthly_Quantity_3scenarios_all(supplier_subset_idx, :);
        Demand_Probability_3scenarios_tmp = Demand_Probability_3scenarios_all(supplier_subset_idx, :);
        mean_demand_3scenarios_tmp = mean_demand_3scenarios_all(supplier_subset_idx);
        c_source_tmp = c_source_all(supplier_subset_idx);
        c_3DP_tmp = c_3DP_all(supplier_subset_idx);
        c_TM_tmp = c_TM_all(supplier_subset_idx);
        v_tmp = v_all(supplier_subset_idx);
        h_tmp = h_all(supplier_subset_idx);
        weight_tmp = weight_all(supplier_subset_idx);

        % Identify products without TM backup
        nobackup_set = logical(1 - TM_BACKUPSET{num_suppliers_case, random_suppliers_num});
        num_suppliers = sum(nobackup_set);

        % Filter data for products without TM backup
        Monthly_Weight_3scenarios = Monthly_Weight_3scenarios_tmp(nobackup_set, :);
        Monthly_Quantity_3scenarios = Monthly_Quantity_3scenarios_tmp(nobackup_set, :);
        Demand_Probability_3scenarios = Demand_Probability_3scenarios_tmp(nobackup_set, :);
        mean_demand_3scenarios = mean_demand_3scenarios_tmp(nobackup_set);
        c_source = c_source_tmp(nobackup_set);
        c_3DP = c_3DP_tmp(nobackup_set);
        c_TM = c_TM_tmp(nobackup_set);
        v = v_tmp(nobackup_set);
        h = h_tmp(nobackup_set);
        weight = weight_tmp(nobackup_set);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Compute Demand Shortfall for TM-Only System (No 3DP)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Compute costs when no 3DP is available
        input_medium_no3dp = struct( ...
            'n', num_suppliers, 'v', v, 'h', h, 'p', p_medium, ...
            'yield_loss_rate', yield_loss_rate_medium, 'Demand_atoms', Monthly_Quantity_3scenarios, ...
            'Demand_prob', Demand_Probability_3scenarios, 'Demand_mean', mean_demand_3scenarios, 'TM_flag', 0 ...
        );
        output_medium_no3dp = Cost_No3DP_or_TM(input_medium_no3dp);

        % Generate demand-disruption samples for evaluation
        input_preprocess_medium_sampled = struct( ...
            'num_suppliers', num_suppliers, 'num_scenarios', num_scenarios, ...
            'yield_loss_rate', yield_loss_rate_medium, 'p_disrupt', p_medium, ...
            'Monthly_Quantity', Monthly_Quantity_3scenarios, 'Demand_Probability', Demand_Probability_3scenarios, ...
            'sample_mode', 2, 'disruption_sample_flag', 1, 'demand_sample_flag', 1, ...
            'demand_num_per_disruption', 100, 'disruption_samplesize', 1000 ...
        );
        output_preprocess_medium_sampled = Data_prep_for_MIP(input_preprocess_medium_sampled);
        disruption_demand_joint_prob_medium_sampled = output_preprocess_medium_sampled.disruption_demand_joint_prob;
        failure_data_medium_sampled = output_preprocess_medium_sampled.failure_data;
        demand_data_medium_sampled = output_preprocess_medium_sampled.demand_data;

        % Compute demand shortfall across sampled scenarios
        q_SP = output_medium_no3dp.opt_q;
        demand_shortfalls = arrayfun(@(ss) sum(max(demand_data_medium_sampled(:, ss) - q_SP .* failure_data_medium_sampled(:, ss), 0)), ...
                                     1:length(disruption_demand_joint_prob_medium_sampled));

        % Store results
        TMONLY_Demand_shortfall_mean{num_suppliers_case, i, j}(random_suppliers_num) = mean(demand_shortfalls);
        TMONLY_Demand_shortfall_mean_relative{num_suppliers_case, i, j}(random_suppliers_num) = ...
            100 * mean(demand_shortfalls) / sum(max(Monthly_Quantity_3scenarios_tmp, [], 2));

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Compute Demand Shortfall for Naive 3DP Policy
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Retrieve optimal backup assignment and ordering quantity
        [~, kkk] = min(TOTAL_COST_NAIVEPOLICY_SET{num_suppliers_case, random_suppliers_num}{i, j});
        q_SP = Q_SP_NAIVE_POLICY_SET{num_suppliers_case, random_suppliers_num}(:, kkk);

        % Compute demand shortfall under Naive 3DP strategy
        demand_shortfalls = arrayfun(@(ss) ...
            sum(max(demand_data_medium_sampled(:, ss) - q_SP .* failure_data_medium_sampled(:, ss) - ...
                     V3DP_b2b(struct('D_bar', demand_data_medium_sampled(:, ss) - q_SP .* failure_data_medium_sampled(:, ss), ...
                                     'K_3DP', capacity_3dp_percentage(kkk) * sum(max(Monthly_Weight_3scenarios, [], 2)))).q_3DP, ...
                   0)), ...
            1:length(disruption_demand_joint_prob_medium_sampled));

        % Store results
        Naive_Demand_shortfall_mean{num_suppliers_case, i, j}(random_suppliers_num) = mean(demand_shortfalls);
        Naive_Demand_shortfall_mean_relative{num_suppliers_case, i, j}(random_suppliers_num) = ...
            100 * mean(demand_shortfalls) / sum(max(Monthly_Quantity_3scenarios_tmp, [], 2));

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Compute Demand Shortfall for Full 3DP Strategy
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Retrieve optimal 3DP assignment
        [~, kkk] = min(TOTAL_COST_SET{num_suppliers_case, random_suppliers_num}{i, j});
        x_3DP = OUTPUT_MEDIUM_BOE{num_suppliers_case, random_suppliers_num, kkk}.X_FINAL{i, j};
        q_SP = OUTPUT_MEDIUM_BOE{num_suppliers_case, random_suppliers_num, kkk}.Q_FINAL{i, j}(x_3DP);

        % Compute demand shortfall under Full 3DP strategy
        demand_shortfalls = arrayfun(@(ss) ...
            sum(max(demand_data_medium_sampled(:, ss) - q_SP .* failure_data_medium_sampled(:, ss) - ...
                     V3DP_b2b(struct('D_bar', demand_data_medium_sampled(:, ss) - q_SP .* failure_data_medium_sampled(:, ss), ...
                                     'K_3DP', capacity_3dp_percentage(kkk) * sum(max(Monthly_Weight_3scenarios_tmp, [], 2)))).q_3DP, ...
                   0)), ...
            1:length(disruption_demand_joint_prob_medium_sampled));

        % Store results
        Full_Demand_shortfall_mean{num_suppliers_case, i, j}(random_suppliers_num) = mean(demand_shortfalls);
        Full_Demand_shortfall_mean_relative{num_suppliers_case, i, j}(random_suppliers_num) = ...
            100 * mean(demand_shortfalls) / sum(max(Monthly_Quantity_3scenarios_tmp, [], 2));

    end
end

%% Plot the mean demand shortfall (for different "num_suppliers_case" under the three strategies)
figure;
hold on;

for num_suppliers_case = 1 : length(NUM_SUPPLIERS_SET)       
    % SINGLE 
    boxplot(TMONLY_Demand_shortfall_mean_relative{num_suppliers_case,i,j} , 'Positions', NUM_SUPPLIERS_SET(num_suppliers_case)-3, 'Widths', 2) 
    % Naive
    boxplot(Naive_Demand_shortfall_mean_relative{num_suppliers_case,i,j}, 'Positions', NUM_SUPPLIERS_SET(num_suppliers_case), 'Widths', 2) 
    % Full
    boxplot(Full_Demand_shortfall_mean_relative{num_suppliers_case,i,j}, 'Positions', NUM_SUPPLIERS_SET(num_suppliers_case)+3, 'Widths', 2) 
end
xlim([min(NUM_SUPPLIERS_SET(2:end)) - 10, max(NUM_SUPPLIERS_SET) + 10]); % Adjust x-axis limits
xticks(NUM_SUPPLIERS_SET(2:end)); % Set x-axis ticks at specified positions
xticklabels(string(NUM_SUPPLIERS_SET(2:end))); % Display the actual positions as tick labels
% title('Boxplots for Each Row');
grid on;

fileName = strcat('Experiment_Data/Switch_Backup_vs_n/Demand_shortfall_boxplots', num2str(i), num2str(j), '.pdf'); % Specify the file name
saveas(gcf, fileName); % Save current figure as a PDF

hold off;

close(gcf);


% Loop through and save data in CSV format
for num_suppliers_case = 1 : length(NUM_SUPPLIERS_SET)
    % Ensure the data is a column vector
    SINGLE_data = TMONLY_Demand_shortfall_mean_relative{num_suppliers_case, i, j}(:); % Convert to column vector
    Naive_data = Naive_Demand_shortfall_mean_relative{num_suppliers_case, i, j}(:); % Convert to column vector
    Full_data = Full_Demand_shortfall_mean_relative{num_suppliers_case, i, j}(:);   % Convert to column vector
    
    % Combine data into a table
    data_table = table(SINGLE_data, Naive_data, Full_data, ...
        'VariableNames', {'SINGLE', 'Naive', 'Full'});
    
    % Add NumSuppliers column (repeated for each observation)
    data_table.NumSuppliers = repmat(NUM_SUPPLIERS_SET(num_suppliers_case), height(data_table), 1);
    
    % Save to CSV
    file_name = strcat('Experiment_Data/Switch_Backup_vs_n/Data_NumSuppliers_demandshortfall_', num2str(NUM_SUPPLIERS_SET(num_suppliers_case)), '.csv');
    writetable(data_table, file_name);
end




save("Experiment_Data/Switch_Backup_vs_n/data_n_less_than_55.mat")



