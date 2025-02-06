
% =========================================================================
% Script Name:     Experiment_Switch_Backup_DT.m 
% Author:          Ziyu He  
% Date:            02/01/2025  
%  
% Description:  
%   This script investigates the factors driving product switches to 3DP backup.  
%   Specifically, we analyze:  
%     - The probability of switching based on supplier characteristics.  
%     - The role of cost structures, demand levels, and disruption risks.  
%  
% Methodology:  
%   - We sample multiple supplier subsets, each containing a fixed number of suppliers (n = 10).  
%   - For each subset:  
%       - BoE local search is applied over a grid of K values.  
%       - The optimal K value determines the 3DP backup set.  
%   - To introduce additional heterogeneity, the following parameters are varied:  
%       - **Marginal failure rate** (`p`): Uniformly sampled from (0, 0.5).  
%       - **Yield loss rate** (`yield_loss_rate`): Uniformly sampled from (0, 1).  
%       - **Cost ratio of TM to primary sourcing** (`c_TM / c_source`): Uniformly sampled from (0, 1).  
%       - **TM fixed cost ratio relative to mean sourcing cost** (`TM_retainer_ratio`): Uniformly sampled from (0.5, 1).  
%  
% Data Structure:  
%   - Each supplier contributes a data point:  
%       - **Binary response**: Switched to 3DP or not.  
%       - **Feature vector**: Capturing key supplier attributes.  
%  
% Feature Engineering:  
%   - **Raw Features**:  
%       - Cost Parameters: `v`, `h`, `weight`, `c_TM`, `c_3DP`.  
%       - Demand Statistics: Mean, max, min demand levels.  
%       - Disruption Characteristics: Failure probability, yield loss rate.  
%       - 3DP-Specific: `c_cap` (depreciation cost per output).  
%   - **Synthetic Features**:  
%       - **Relative Expensiveness of 3DP**: (v-c3DP/weight) and c_cap/(v-c3DP/weight) 
%       - **Service Level (Profitability)**: v/(v+h), c_TM/(c_TM+h),  c_3DP/(c_3DP+h)
%       - **Max Demand Shortfall to Handle**: [ D - q*s ]^+,   where q stands for the opt. primary order when K=0
%       - **The ratio of TM fixed cost to the mean sourcing costs
%  
% Outputs:  
%   - A dataset containing product-switching probabilities and corresponding feature vectors.  
%   - Comparative analysis of different cost structures and demand-disruption scenarios.  
%   - Plots illustrating switching trends under varying parameter conditions.  
%  
% Notes:  
%   - "Dedicated Backup" refers to Traditional Manufacturing (TM).  
%   - 3DP capacity represents the total monthly output of printing materials.  
%   - Weight values are converted from grams to kilograms for cost scaling.  
%  
% Adjustments:  
%   - `weight_all` is divided by 1000 to reflect per-unit product weight.  
%   - `speed_per_machine_month` is divided by 1000 for material consumption scaling.  
%   - `Monthly_Weight_3scenarios_all` is divided by 1000 to normalize weight-based demand.  
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





%% Experiment: Understanding the Drivers of Product Switching  
%
%  - We aim to identify the factors influencing the decision to switch products to 3DP backup.  
%  - To achieve this, we sample multiple subsets of suppliers, each with a fixed size (n = 10).  
%  - For each sampled subset:  
%      - We run BoE local search across a grid of K values.  
%      - At the optimal K, we determine the 3DP backup set.  
%  - To introduce additional heterogeneity, we vary the following parameters for each subset (relative to baseline values):  
%      - **Marginal failure rate** (`p`): Uniformly sampled from (0, 0.5).  
%      - **Yield loss rate** (`yield_loss_rate`): Uniformly sampled from (0, 1).  
%      - **Cost ratio of TM to primary sourcing** (`c_TM / c_source`): Uniformly sampled from (0, 1).  
%      - **TM fixed cost ratio relative to mean sourcing cost** (`TM_retainer_ratio`): Uniformly sampled from (0.5, 1).  


Random_Suppliers_Num = 500;

% 3DP capacity as a percentage of max demand
capacity_3dp_percentage = [1e-2, 1e-1 * [1:9], 1:2:25, 30:5:50, 75, 100] / 100;

OUTPUT_MEDIUM_BOE = {};
A_t_BOE_SET = {};
X_BOE_SET = {};
TOTAL_COST_SET = {};
X_BOE_OPT_SET = {};


nn = 10;

P_MEDIUM_SET = [];
YIELD_LOSS_RATE_MEDIUM_SET = [];
SUPPLIER_SUBSET_IDX = {};
TM_BACKUPSET = {};
C_TM_SET = {};
TM_RETAINER_RATIO_SET = {}; 
NOBACKUP_SET = [];
Q_NOBACKUP_SET = [];


for random_suppliers_num = 1 : Random_Suppliers_Num
    
    startTime = clock;

    display("---------------------------------------------------------------------\n")
    fprintf("WE ARE CURRENTLY WORKING ON: %d suppliers, case #%d \n", nn, random_suppliers_num)
    display("---------------------------------------------------------------------\n")
    
    p_medium = unifrnd(0, 0.5);
    yield_loss_rate_medium = unifrnd(0, 1);

    P_MEDIUM_SET(random_suppliers_num) = p_medium;
    YIELD_LOSS_RATE_MEDIUM_SET(random_suppliers_num) = yield_loss_rate_medium;
    
    %% In what follows, we radomly sample a subset of products and treat it as all the products we have here
    num_suppliers = nn;
    supplier_subset_idx = false(num_suppliers_all, 1);
    supplier_subset_idx(randperm(num_suppliers_all, num_suppliers)) = true;

    SUPPLIER_SUBSET_IDX{random_suppliers_num} = supplier_subset_idx;
    
    Monthly_Weight_3scenarios = Monthly_Weight_3scenarios_all(supplier_subset_idx, :);
    Monthly_Quantity_3scenarios = Monthly_Quantity_3scenarios_all(supplier_subset_idx, :);
    Demand_Probability_3scenarios = Demand_Probability_3scenarios_all(supplier_subset_idx, :);
    mean_demand_3scenarios = mean_demand_3scenarios_all(supplier_subset_idx);
    
    c_source = c_source_all(supplier_subset_idx);
    
    c_3DP = c_3DP_all(supplier_subset_idx); 
    % c_TM = c_TM_all(supplier_subset_idx);   
    c_TM = c_source.*unifrnd(0,1,10,1); C_TM_SET{1,random_suppliers_num} = c_TM;
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
    % TM_retainer_ratio = 0.75;
    TM_retainer_ratio = unifrnd(0.5,1,nn,1);
    C_TM = TM_retainer_ratio.*c_source.*mean_demand_3scenarios; TM_RETAINER_RATIO_SET{1,random_suppliers_num} = TM_retainer_ratio;
    input_medium_TM = input_medium_no3dp;
    input_medium_TM.TM_flag = 1;
    input_medium_TM.c_TM = c_TM; 
    input_medium_TM.C_TM = C_TM; 
    output_medium_TM = Cost_No3DP_or_TM(input_medium_TM);



    %% The products that are originally backed-up by TM
    TM_backup_set = output_medium_TM.TM_cost < output_medium_no3dp.opt_val;
    nobackup_set = 1 - TM_backup_set;
    TM_BACKUPSET{random_suppliers_num} = TM_backup_set;

    NOBACKUP_SET(:,random_suppliers_num) = (output_medium_no3dp.opt_val < output_medium_TM.TM_cost);
    Q_NOBACKUP_SET(:,random_suppliers_num) = output_medium_no3dp.opt_q;    

    
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
            input_boe.A_init = OUTPUT_MEDIUM_BOE{random_suppliers_num, k-1}.A_t;
        end

        if nn <= 10
            output_boe = BoE_Approx_Max_Submod_Exact(input_boe);
        else
            output_boe = BoE_Approx_Max_Submod_SAA(input_boe);
        end
        
    
        OUTPUT_MEDIUM_BOE{random_suppliers_num,k} = output_boe; 
    
        disp("TO NEXT ONE! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ")
    
    end
    
    %% A_t_BOE:      the 3DP set we obtain before post-processing (comparing to having no 3DP selection)
    %% X_BOE:        the 3DP set we obatin after post-processing (under different C3DP coefficients)
    %% TOTAL_COST:   the total system cost after post-processing 
    %% NOTE: TOTAL_COST is computed by (K is fixed):
    %%      - Given "A_t_BOE" obtained from local search method, fix the 3DP set as "A_t_BOE"
    %%      - Run SGD on this fixed 3DP set problem
    %%      - Given solution of SGD, evaluate the U3DP with a larger sample size
    %%      - Compare to the case when no 3DP is selected
    %% Threfore, under different K, the "TOTAL_COST" could be computed on different samples
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
    
        A_t_BOE(OUTPUT_MEDIUM_BOE{random_suppliers_num,k}.A_t, k) = 1;
    
        for i = 1 : length(speed_per_machine_month)  
            for j = 1 : length(cost_of_3dp_per_machine_month)
    
                TOTAL_COST{i,j}(k) = OUTPUT_MEDIUM_BOE{random_suppliers_num,k}.TOTAL_COST(i,j);
                X_BOE{i,j}(:, k) = OUTPUT_MEDIUM_BOE{random_suppliers_num,k}.X_FINAL{i,j};
                if ~isinf(sum(X_BOE{i,j}(:,k)))
                    SWITCHED_BOE{i,j}(:,k) = TM_backup_set - (1-X_BOE{i,j}(:,k));
                end
    
            end
        end
    
    end
    

    A_t_BOE_SET{random_suppliers_num} = A_t_BOE;
    X_BOE_SET{random_suppliers_num} = X_BOE;
    SWITCHED_BOE_SET{random_suppliers_num} = SWITCHED_BOE;
    TOTAL_COST_SET{random_suppliers_num} = TOTAL_COST;

    %% "X_BOE_OPT" is the 3DP set at the optimal K
    %% "SWITCHED_SET_OPT" is the set of products whose backup switched after intro of 3DP
    X_BOE_OPT = {}; 
    SWITCHED_BOE_OPT = {};
    for i = 1 : length(speed_per_machine_month)  
        for j = 1 : length(cost_of_3dp_per_machine_month)

            [~, iii] = min(TOTAL_COST{i,j});
            X_BOE_OPT{i,j} = X_BOE{i,j}(:,iii);
            SWITCHED_BOE_OPT{i,j} = SWITCHED_BOE{i,j}(:,iii);

        end
    end
    
    X_BOE_OPT_SET{random_suppliers_num} = X_BOE_OPT;
    SWITCHED_BOE_OPT_SET{random_suppliers_num} = SWITCHED_BOE_OPT;
    
    endTime = clock;
    Tot_time = Tot_time + etime(endTime, startTime);

    fprintf("TIME: %3.2f\n\n", Tot_time)

end

Num_3DP_backup = {};
for random_suppliers_num = 1 : Random_Suppliers_Num

    for i = 1 : length(speed_per_machine_month)  
        for j = 1 : length(cost_of_3dp_per_machine_month)
            
            Num_3DP_backup{i, j}(random_suppliers_num) = sum(X_BOE_OPT_SET{random_suppliers_num}{i,j});

        end
    end

    
end






%% Feature Engineering: for each product, workout the followings
%% Vanilla features:
%%      - Cost-parameters: v, h, weight, cTM, c3DP
%%      - Demand: mean, max, min values
%%      - Disruption: failure probability, yield loss rate (only two atoms and one of them is 1)
%%      - (NOT SURE) C3DP related: c_cap (depreciation cost / output)
%% Synthesize features:
%%      - Relative Expensiveness of 3DP : c_cap/(v-c3DP)
%%      - Service Level (profitability) : (v-c)/(v+h)
%%      - Max Demand shortfall to deal with: [ D - q*s ]^+, where q stands for the opt. primary order when K=infty


RAW_FEATURES_SET = {};
SYNTH_FEATURES_SET = {};
RESPONSES_SET = {};

for i = 1 : length(speed_per_machine_month)  
    for j = 1 : length(cost_of_3dp_per_machine_month)

        RAW_FEATURES = [];
        SYNTH_FEATURES = [];
        RESPONSES = [];


        for random_suppliers_num = 1 : Random_Suppliers_Num
        
            display("---------------------------------------------------------------------\n")
            fprintf("WE ARE CURRENTLY WORKING ON: %d suppliers, case #%d \n", nn, random_suppliers_num)
            display("---------------------------------------------------------------------\n")
            
            %% Read the data for each sampled subset (of size nn)

            p_medium = P_MEDIUM_SET(random_suppliers_num);
            yield_loss_rate_medium = YIELD_LOSS_RATE_MEDIUM_SET(random_suppliers_num);
            
            supplier_subset_idx = SUPPLIER_SUBSET_IDX{random_suppliers_num};
            
            Monthly_Weight_3scenarios = Monthly_Weight_3scenarios_all(supplier_subset_idx, :);
            Monthly_Quantity_3scenarios = Monthly_Quantity_3scenarios_all(supplier_subset_idx, :);
            Demand_Probability_3scenarios = Demand_Probability_3scenarios_all(supplier_subset_idx, :);
            mean_demand_3scenarios = mean_demand_3scenarios_all(supplier_subset_idx);
            
            c_source = c_source_all(supplier_subset_idx);
            
            c_3DP = c_3DP_all(supplier_subset_idx); 
            c_TM = C_TM_SET{1,random_suppliers_num};
            v = v_all(supplier_subset_idx);         
            h = h_all(supplier_subset_idx);  
            
            weight = weight_all(supplier_subset_idx);
            
           
       
            %% Vanilla (Raw) features:
            %%      - Cost-parameters: v, h, c3DP, cTM, weight, (v-c_3DP)./weight
            %%      - Demand: mean
            %%      - Disruption: failure probability, yield loss rate (only two atoms and one of them is 1)
            raw_feature_block = [v, h, c_3DP, c_TM, weight, (v-c_3DP)./weight, mean_demand_3scenarios, p_medium*ones(nn,1), yield_loss_rate_medium*ones(nn,1)];        
            RAW_FEATURES = [RAW_FEATURES; raw_feature_block];


            %% Responses (1="Switched from TM to 3DP", 0="Not Swithced")
            response_block = SWITCHED_BOE_OPT_SET{random_suppliers_num}{i,j};
            RESPONSES = [RESPONSES; response_block];
        

            %% Synthesize features:
            %%      - Absolute and Relative Expensiveness of 3DP : (v-c3DP/weight) and c_cap/(v-c3DP/weight) 
            %%      - Service Level (profitability) : v/(v+h), c_TM/(c_TM+h),  c_3DP/(c_3DP+h)
            %%      - (Relative to mean deamnd) Mean Demand shortfall: [ D - q*s ]^+, where q stands for the opt. primary order when K=0
            %%      - The ratio of TM fixed cost to the mean sourcing costs
            
            mean_shortfall = [];
            for l =  1 : size(Monthly_Weight_3scenarios,1)
                mean_shortfall(l,:) = sum(sum( ([p_medium; 1-p_medium] * Demand_Probability_3scenarios(l,:)) ...
                    .* max(0, Monthly_Quantity_3scenarios(l,:) - Q_NOBACKUP_SET(l,random_suppliers_num)*[1-yield_loss_rate_medium; 1]) ));
                mean_shortfall(l,:) = mean_shortfall(l,:)/mean_demand_3scenarios(l,:);
            end
            
            c_cap = cost_of_3dp_per_machine_month(j)/speed_per_machine_month(i);
            synth_feature_block = [(v-c_3DP)./weight, c_cap./((v-c_3DP)./weight), (v./(v+h)), (c_TM./(c_TM+h)), (c_3DP./(c_3DP+h)), mean_shortfall, TM_RETAINER_RATIO_SET{1,random_suppliers_num}];
           
            SYNTH_FEATURES = [SYNTH_FEATURES; synth_feature_block];
           
            
        end
        

        RAW_FEATURES_SET{i,j} = RAW_FEATURES;
        RESPONSES_SET{i,j} = RESPONSES;
        SYNTH_FEATURES_SET{i,j} = SYNTH_FEATURES;


    end
end


for i = 1 : length(speed_per_machine_month)  
    for j = 1 : length(cost_of_3dp_per_machine_month)
        SYNTH_FEATURES_SET{i,j}(:,end)=SYNTH_FEATURES_SET{i,j}(:,end)*100;
    end
end


%% Save the data for R or Python
raw_featureNames = {'v', 'h', 'c3DP', 'cTM', 'weight', '3DP profit', 'demand mean', 'disrupt. prob.', 'yield loss'};
synth_featureNames = {'Abs 3DP profit', 'Relative 3DP profit', 'service level1', 'service level2', 'service level3', 'mean shortfall', 'DB Retainer Rate'};

base_dir = 'Experiment_Data/Decision_Tree/';

for i = 1:length(speed_per_machine_month)
    for j = 1:length(cost_of_3dp_per_machine_month)
        % Get the current datasets
        raw_features = RAW_FEATURES_SET{i,j};
        synth_features = SYNTH_FEATURES_SET{i,j};
        responses = RESPONSES_SET{i,j};
        
        % Create descriptive filename based on parameters
        filename_base = sprintf('scenario_speed_%d_cost_%d', i, j);
        
        % Save raw features
        raw_table = array2table(raw_features, 'VariableNames', raw_featureNames);
        writetable(raw_table, fullfile(base_dir, ...
            [filename_base '_raw_features.csv']));
        
        % Save synthetic features
        synth_table = array2table(synth_features, 'VariableNames', synth_featureNames);
        writetable(synth_table, fullfile(base_dir, ...
            [filename_base '_synth_feature.csv']));
        
        % Save responses
        resp_table = array2table(responses, 'VariableNames', {'response'});
        writetable(resp_table, fullfile(base_dir, ...
            [filename_base '_responses.csv']));
        
        % Save a metadata file with scenario parameters
        metadata = struct();
        metadata.speed = speed_per_machine_month(i);
        metadata.cost = cost_of_3dp_per_machine_month(j);
        metadata.raw_feature_names = raw_featureNames;
        metadata.synth_feature_names = synth_featureNames;
        save(fullfile(base_dir, ...
            [filename_base '_metadata.mat']), 'metadata');
        
        % Also save metadata in JSON format for easier reading in Python/R
        fid = fopen(fullfile(base_dir, ...
            [filename_base '_metadata.json']), 'w');
        fprintf(fid, jsonencode(metadata));
        fclose(fid);
    end
end




save("Experiment_Data/Decision_Tree/data.mat")











