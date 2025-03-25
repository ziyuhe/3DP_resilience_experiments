% =========================================================================
% Script Name:       Experiments_CostSavings_and_DemandShortfalls_Varying_c3DP.m
% Date:              02/01/2025
% Description:       
%   - This is part of a series of experiments prefixed with "Experiments_CostSavings_and_DemandShortfalls".
%   - Evaluates the impact of key hyperparameters on the performance of the 3DP resilience strategy.
%   - Specifically, here we analyze effect of varying 3DP variable costs (c_3DP) on resilience performance.
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
%   - **Focus**: Analyzing the impact of **c_3DP**, in the range of:
%       0.5x, 1x, 2x, 3x of baseline value
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
  

%% In this experiment, we focus on analyzing the impact of **c_3DP**,  
% where **c_3DP** varies as follows: 0.5x, 1x, 2x, 3x of baseline
% (NOTE: the 1x baseline case was already computed in "Experiments_CostSavings_and_DemandShortfalls_Varying_C3DP")


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VARYING c_3DP: COMPUTATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% NOW WE TRY DIFFERENT c_3DP
%%  - baseline is c_3DP = c_source and c_TM = 0.5 * c_source
%%  - we can try: c_3DP = r * c_source, where r = 0.5, 2, 3 
%%          - NOTE WE CAN'T LET c_3DP > 3 * c_source, in such case some products will never be 3DP backed-up)

c_3DP_rate_set = [0.5, 2, 3];

OUTPUT_MEDIUM_BOE_VARYING_c3DP = {};
COST_TMONLY_VARYING_c3DP = {};

for ll = 1:length(c_3DP_rate_set)

    num_suppliers = num_suppliers_all;
    supplier_subset_idx = false(num_suppliers_all, 1);
    
    Monthly_Weight_3scenarios = Monthly_Weight_3scenarios_all;
    Monthly_Quantity_3scenarios = Monthly_Quantity_3scenarios_all;
    Demand_Probability_3scenarios = Demand_Probability_3scenarios_all;
    mean_demand_3scenarios = mean_demand_3scenarios_all;
            
    c_source = c_source_all;
    
    v = v_all;         
    h = h_all;  
    
    weight = weight_all;

    c_3DP = c_3DP_rate_set(ll) * c_source; 
    c_TM = 0.5 * c_source;   
    
    %% Compute supplier costs under different backup strategies:
    % 1. No backup
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
            
    % 2. Backup with 3DP (infinite capacity)
    input_medium_3DP_infK = input_medium_no3dp;
    input_medium_3DP_infK.v = c_3DP;
    output_medium_3DP_infK = Cost_No3DP_or_TM(input_medium_3DP_infK);
    
    % 3. Backup with TM
    TM_retainer_ratio = 0.75;
    C_TM = TM_retainer_ratio * c_source .* mean_demand_3scenarios;
    input_medium_TM = input_medium_no3dp;
    input_medium_TM.TM_flag = 1;
    input_medium_TM.c_TM = c_TM; 
    input_medium_TM.C_TM = C_TM; 
    output_medium_TM = Cost_No3DP_or_TM(input_medium_TM);
    
    %% Identify products originally backed up by TM
    TM_backup_set = output_medium_TM.TM_cost < output_medium_no3dp.opt_val;
    nobackup_set = 1 - TM_backup_set;
    COST_TMONLY_VARYING_c3DP{ll} = sum(output_medium_TM.TM_cost(TM_backup_set)) + sum(output_medium_no3dp.opt_val(logical(nobackup_set)));
    
    %% Sample data for BoE Submodular Maximization
    input_preprocess_medium_sampled.num_suppliers = num_suppliers;
    input_preprocess_medium_sampled.num_scenarios = num_scenarios;
    input_preprocess_medium_sampled.yield_loss_rate = yield_loss_rate_medium;
    input_preprocess_medium_sampled.p_disrupt = p_medium;
    input_preprocess_medium_sampled.Monthly_Quantity = Monthly_Quantity_3scenarios;
    input_preprocess_medium_sampled.Demand_Probability = Demand_Probability_3scenarios;
    
    input_preprocess_medium_sampled.sample_mode = 2;
    input_preprocess_medium_sampled.disruption_sample_flag = 1; % No disruption sampling (keep all combinations)
    input_preprocess_medium_sampled.demand_sample_flag = 1;     % Sample a fixed number of demand combinations per disruption scenario
    
    % Set sample sizes based on supplier count
    if num_suppliers < 20
        demand_samplesize_saa = 200;
        disruption_samplesize_saa = 100;
    elseif num_suppliers < 30
        demand_samplesize_saa = 300;
        disruption_samplesize_saa = 150;        
    elseif num_suppliers < 40
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
    
    %% Prepare data for BoE Submodular Maximization
    Obj_const = - output_medium_no3dp.opt_val;
    U0_with_vmean = output_medium_no3dp.opt_val + v .* mean_demand_3scenarios;
    U0_no_vmean = output_medium_no3dp.opt_val;
    TM_Delta = output_medium_TM.TM_cost - output_medium_no3dp.opt_val;
    
    ratio_over_weight = (v - c_3DP) ./ weight;
    
    % Optimal ordering when no backup
    q0 = output_medium_no3dp.opt_q;
    
    % Compute probabilities of unfulfilled demand
    pi_p = [];
    pi_0 = [];
    
    for j = 1:num_suppliers
        tmp1 = max(0, Monthly_Quantity_3scenarios(j,:)' - q0(j) * [1 - yield_loss_rate_medium, 1]);
        tmp2 = Demand_Probability_3scenarios(j,:)' * [p_medium, 1 - p_medium];
    
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
    
    input_boe.q_ub = output_medium_no3dp.opt_q;  % Upper bound for 1st-stage decisions
    input_boe.q_lb = output_medium_3DP_infK.opt_q;  % Lower bound for 1st-stage decisions
    
    %% Iterate over different 3DP capacities
    for k = 1:length(capacity_3dp_percentage)
    
        fprintf("%3.2f Percent of Max Yield Shortfall \n\n", capacity_3dp_percentage(k) * 100);
    
        capacity_percentage = capacity_3dp_percentage(k);
        K_3DP_medium = capacity_percentage * sum(max(Monthly_Weight_3scenarios'));
        C_3DP_medium = (K_3DP_medium ./ speed_per_machine_month)' * cost_of_3dp_per_machine_month;
        
        input_boe.K_3DP = K_3DP_medium;
        input_boe.C_3DP = C_3DP_medium;
    
        input_boe.auto_recompute = 1;
        
        if num_suppliers <= 10
            output_boe = BoE_Approx_Max_Submod_Exact(input_boe);
        else
            output_boe = BoE_Approx_Max_Submod_SAA(input_boe);
        end
    
        OUTPUT_MEDIUM_BOE_VARYING_c3DP{ll, k} = output_boe; 
    
        disp("TO NEXT ONE! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ");
    
    end

end

save("Experiment_Results/Relative_Cost_Savings_Shortfalls_c3DP/Varying_c3DP.mat")


DDD = load("Experiment_Results/Relative_Cost_Savings_Shortfalls_c3DP/Benchmark.mat");

COST_TMONLY_BENCMARK = DDD.COST_TMONLY_BENCMARK;
OUTPUT_MEDIUM_BOE_BENCHMARK = DDD.OUTPUT_MEDIUM_BOE_BENCHMARK;

%% Summarize the relative cost-savings (PERCENTAGE OF BASELINE COST)
COST_SAVINGS_BENCHMARK = {};
for i = 1 : length(speed_per_machine_month)  
    for j = 1 : length(cost_of_3dp_per_machine_month)

        for k = 1 : length(capacity_3dp_percentage)
                
            COST_SAVINGS_BENCHMARK{i,j}(k) = (COST_TMONLY_BENCMARK - OUTPUT_MEDIUM_BOE_BENCHMARK{k}.TOTAL_COST_NONZERO(i,j)) / abs(COST_TMONLY_BENCMARK)*100;
            
        end

    end
end

COST_SAVINGS_VARYING_c_3DP = {};
for ll = 1 : length(c_3DP_rate_set)

    for i = 1 : length(speed_per_machine_month)  
        for j = 1 : length(cost_of_3dp_per_machine_month)
    
            for k = 1 : length(capacity_3dp_percentage)
                    
                COST_SAVINGS_VARYING_c_3DP{ll,i,j}(k) = (COST_TMONLY_VARYING_c3DP{ll} - OUTPUT_MEDIUM_BOE_VARYING_c3DP{ll, k}.TOTAL_COST_NONZERO(i,j)) / abs(COST_TMONLY_VARYING_c3DP{ll})*100;
                
            end
    
        end
    end

end








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VARYING c_3DP: PLOT COST-SAVINGS VS. K
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% NOTE: These plots are for initial verification. The final plots in the paper are generated in Python using the data saved here.

%% Plot Cost Savings: 
%%      - For each combo of "speed_per_machine_month" and "cost_of_3dp_per_machine_month"
%%      - Plot "cost-savings" vs. "K" under different c_3DP
for i = 1 : length(speed_per_machine_month)  
    for j = 1 : length(cost_of_3dp_per_machine_month)

        plot([0,capacity_3dp_percentage]*100, [0,COST_SAVINGS_BENCHMARK{i,j}], '-o','LineWidth', 1)
        hold on
        
        for ll = 1 : length(c_3DP_rate_set)
        
            plot([0,capacity_3dp_percentage]*100, [0,COST_SAVINGS_VARYING_c_3DP{ll,i,j}], '-o','LineWidth', 1)
            hold on
        
        end
        
        ylimit = max([COST_SAVINGS_VARYING_c_3DP{1,i,j}, COST_SAVINGS_VARYING_c_3DP{2,i,j}, COST_SAVINGS_VARYING_c_3DP{3,i,j}, COST_SAVINGS_BENCHMARK{i,j}]);
        ylim([-1.1*ylimit,  1.1*ylimit])
        
        xlim([0,100])
        
        lgd = legend({'1x Baseline', '0.5x Baseline', '2x Baseline', '3x Baseline'}, 'FontSize', 12, 'location', 'northeast');
        title(lgd, '3DP variable cost $c^{\mathsf{3DP}}$', 'interpreter', 'latex');
        
        filename = strcat('Experiment_Results/Relative_Cost_Savings_Shortfalls_c3DP/Varying_c3DP(CostSavings)/Varying_c3DP_Speed', num2str(i), '_PrinterCost', num2str(j) ,'.pdf');
        
        ax = gca;
        ax.XTickLabel = strcat(ax.XTickLabel, '%');
        ax.YTickLabel = strcat(ax.YTickLabel, '%');
        
        saveas(gcf,filename)  % as MATLAB figure
        close(gcf)

    end
end

%% Plot i=1, j=1 case

iiii = [1,11:length(capacity_3dp_percentage)];

i = 1; j = 1;

plot([0,capacity_3dp_percentage(iiii)]*100, [0,COST_SAVINGS_BENCHMARK{i,j}(iiii)], '-o','LineWidth', 1)
hold on

for ll = 1 : length(c_3DP_rate_set)

    plot([0,capacity_3dp_percentage(iiii)]*100, [0,COST_SAVINGS_VARYING_c_3DP{ll,i,j}(iiii)], '-o','LineWidth', 1)
    hold on

end

yline(0, '--', 'LineWidth', 1.5, 'Color', [0.5, 0.5, 0.5]);  % Horizontal grey line

x_fill = [0, 100, 100, 0];  % X-coordinates for the shaded area
y_fill = [0, 0, -1e6, -1e6];  % Arbitrary large negative value for shading
fill(x_fill, y_fill, [0.8, 0.8, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);  % Light grey fill

ylimit = max([COST_SAVINGS_VARYING_c_3DP{1,i,j}, COST_SAVINGS_VARYING_c_3DP{2,i,j}, COST_SAVINGS_VARYING_c_3DP{3,i,j}, COST_SAVINGS_BENCHMARK{i,j}]);
ylim([-1.1*ylimit,  1.1*ylimit])

xlim([0,30])

lgd = legend({'1x Baseline', '0.5x Baseline', '2x Baseline', '3x Baseline'}, 'FontSize', 12, 'location', 'northeast');
title(lgd, '3DP variable cost $c^{\mathsf{3DP}}$', 'interpreter', 'latex');

filename = strcat('Experiment_Results/Relative_Cost_Savings_Shortfalls_c3DP/Varying_c3DP(CostSavings)/Varying_c3DP_Speed', num2str(i), '_PrinterCost', num2str(j) ,'(PRETTIER).pdf');

ax = gca;
ax.XTickLabel = strcat(ax.XTickLabel, '%');
ax.YTickLabel = strcat(ax.YTickLabel, '%');

saveas(gcf,filename)  % as MATLAB figure
close(gcf)



%% Save the data for CSV and PYTHON
% Prepare the data
% Prepare the X-axis data (capacity percentages)
iiii = [1, 11:length(capacity_3dp_percentage)];
capacity_selected = capacity_3dp_percentage(iiii) * 100; % Scale to percentage

% Initialize data matrix
num_rows = length(capacity_selected) + 1; % Include the leading zero row
num_curves = length(c_3DP_rate_set);  % Baseline + varying curves
data_matrix = zeros(num_rows, num_curves); % Extra column for capacity

% Add capacity (X-axis) as the first column
data_matrix(:, 1) = [0; capacity_selected(:)]; % Leading zero for baseline

% Add Baseline (COST_SAVINGS_BENCHMARK) as the second column
data_matrix(:, 2) = [0; COST_SAVINGS_VARYING_c_3DP{1, i, j}(iiii)'];
data_matrix(:, 3) = [0; COST_SAVINGS_BENCHMARK{i, j}(iiii)'];

% Add varying cost curves (COST_SAVINGS_VARYING_c_3DP) as subsequent columns
col_idx = 4;  % Start at column 3
for ll = 2:length(c_3DP_rate_set)
    data_matrix(:, col_idx) = [0; COST_SAVINGS_VARYING_c_3DP{ll, i, j}(iiii)'];
    col_idx = col_idx + 1;
end

% Define column headers
headers = ['Capacity_3DP_Percentage', 'Baseline', ...
           strcat('Varying_Curve_', string(1:length(c_3DP_rate_set)))];
headers=headers([1,3,2,4,5]);

% Write to CSV file
csv_filename = 'Experiment_Results/Relative_Cost_Savings_Shortfalls_c3DP/varying_c3DP_for_python_costsavings.csv';
fid = fopen(csv_filename, 'w');
fprintf(fid, '%s,', headers{1:end-1});  % Write column headers
fprintf(fid, '%s\n', headers{end});
fclose(fid);

% Append the data matrix to the file
dlmwrite(csv_filename, data_matrix, '-append', 'precision', '%.6f');

fprintf('Data saved to %s\n', csv_filename);








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VARYING c3DP: PLOT DEMAND SHORTFALLS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SINGLE system: No 3DP
%  - Identify the set of products without backup (A0)
%  - Sample pairs of demand and disruptions
%  - For each sample (D, s), compute [D_j - q_j * s_j]^+ and sum over j in A0
%  - Store the demand shortfall distribution as a vector

%% Retrieve demand shortfall results for:
%  - No 3DP (independent of c_3DP or C3DP)
%  - Benchmark c_3DP case (all C3DP)
DDD = load("Experiment_Results/Relative_Cost_Savings_Shortfalls_c3DP/Benchmark.mat");
Relative_Demand_shortfall_no3DP_Benchmark = DDD.Relative_Demand_shortfall_no3DP_Benchmark;
Relative_Demand_shortfall_varying_C3DP = DDD.Relative_Demand_shortfall_varying_C3DP;

%% DUO system: With 3DP backup
%  - Iterate over a grid of 3DP capacities (K)
%  - Identify the set of products backed up by 3DP (A)
%  - Sample pairs of demand and disruptions
%  - For each sample (D, s), compute the optimal 3DP decision q3DP(D, s) and [D_j - q_j * s_j - q_j^3DP(D, s)]^+ and sum over j in A
%  - Store the demand shortfall distribution as a vector

%% Important notes:
%  - For a given K, the demand shortfall remains the same across all c_cap values (q_SP is unchanged)
%  - Different c_cap values lead to different optimal K values, affecting the demand shortfall

%% Compute demand shortfall distributions for each c_3DP case under a fixed c_cap:
%  - For each c_3DP (under fixed c_cap), find the optimal K (max cost savings)
%  - Retrieve x and q_SP at the optimal K and compute demand shortfall
%  - The cost and q_SP values are derived from SGD

Demand_shortfall_varying_c3DP = {};

for i = 1:length(speed_per_machine_month)
    for j = 1:length(cost_of_3dp_per_machine_month)
        
        fprintf("Processing case %d, %d\n", i, j)

        for ll = 1:length(c_3DP_rate_set)

            %% Retrieve the optimal x and q_SP
            [~, kkk] = max(COST_SAVINGS_VARYING_c_3DP{ll, i, j});
            x_3DP = logical(OUTPUT_MEDIUM_BOE_VARYING_c3DP{ll, kkk}.X_FINAL{i, j});
            q_SP = OUTPUT_MEDIUM_BOE_VARYING_c3DP{ll, kkk}.Q_FINAL{i, j}(x_3DP);

            %% Data preparation
            num_suppliers = sum(x_3DP);
            supplier_subset_idx = x_3DP;
            Monthly_Weight_3scenarios = Monthly_Weight_3scenarios_all(supplier_subset_idx, :);
            Monthly_Quantity_3scenarios = Monthly_Quantity_3scenarios_all(supplier_subset_idx, :);
            Demand_Probability_3scenarios = Demand_Probability_3scenarios_all(supplier_subset_idx, :);
            mean_demand_3scenarios = mean_demand_3scenarios_all(supplier_subset_idx);
                   
            c_source = c_source_all(supplier_subset_idx);        
            c_3DP = c_3DP_all(supplier_subset_idx) * c_3DP_rate_set(ll); 
            c_TM = c_TM_all(supplier_subset_idx);   
            v = v_all(supplier_subset_idx);         
            h = h_all(supplier_subset_idx);  
            weight = weight_all(supplier_subset_idx);
        
            capacity_percentage = capacity_3dp_percentage(kkk);
            K_3DP_medium = capacity_percentage * sum(max(Monthly_Weight_3scenarios'));

            %% Sample (D, s) pairs
            input_preprocess_medium_sampled.num_suppliers = num_suppliers;
            input_preprocess_medium_sampled.num_scenarios = num_scenarios;
            input_preprocess_medium_sampled.yield_loss_rate = yield_loss_rate_medium;
            input_preprocess_medium_sampled.p_disrupt = p_medium;
            input_preprocess_medium_sampled.Monthly_Quantity = Monthly_Quantity_3scenarios;
            input_preprocess_medium_sampled.Demand_Probability = Demand_Probability_3scenarios;
            
            input_preprocess_medium_sampled.sample_mode = 2;
            input_preprocess_medium_sampled.disruption_sample_flag = 1; % No disruption sampling (keep all combinations)
            input_preprocess_medium_sampled.demand_sample_flag = 1;     % Sample a fixed number of demand values per disruption scenario
            
            disruption_samplesize_saa = 1000;    
            demand_samplesize_saa = 100;            
            
            input_preprocess_medium_sampled.demand_num_per_disruption = demand_samplesize_saa;
            input_preprocess_medium_sampled.disruption_samplesize = disruption_samplesize_saa;
            
            output_preprocess_medium_sampled = Data_prep_for_MIP(input_preprocess_medium_sampled);
            
            disruption_demand_joint_prob_medium_sampled = output_preprocess_medium_sampled.disruption_demand_joint_prob;
            failure_data_medium_sampled = output_preprocess_medium_sampled.failure_data;
            demand_data_medium_sampled = output_preprocess_medium_sampled.demand_data;
            
            %% Compute demand shortfall for each (D, s) pair
            input_b2b.x = logical(ones(sum(x_3DP), 1));
            input_b2b.c_3DP = c_3DP;
            input_b2b.v = v;
            input_b2b.h = h;
            input_b2b.weight = weight;
            input_b2b.K_3DP = K_3DP_medium;
        
            for ss = 1:length(disruption_demand_joint_prob_medium_sampled)
                
                D_sample = demand_data_medium_sampled(:, ss); 
                s_sample = failure_data_medium_sampled(:, ss);   
                D_bar = D_sample - q_SP .* s_sample;
                input_b2b.D_bar = D_bar;
        
                output_b2b = V3DP_b2b(input_b2b);
                q_3DP = output_b2b.q_3DP;
                
                Demand_shortfall_varying_c3DP{i, j, ll}(ss) = sum(max(D_bar - q_3DP, 0));
        
            end

        end

    end
end



%% Get the relative demand shortfall (relative to total max demand)
Relative_Demand_shortfall_varying_c3DP = {};
for i = 1 : length(speed_per_machine_month)  
    for j = 1 : length(cost_of_3dp_per_machine_month)

        Relative_Demand_shortfall_varying_c3DP{i,j,1} = Demand_shotfall_varying_c3DP{i,j,1}/sum(max(Monthly_Quantity_3scenarios_all'))*100;
        Relative_Demand_shortfall_varying_c3DP{i,j,2} = Relative_Demand_shortfall_varying_C3DP{i,j};

        for ll = 2 : length(c_3DP_rate_set)
            Relative_Demand_shortfall_varying_c3DP{i,j,ll+1} = Demand_shotfall_varying_c3DP{i,j,ll}/sum(max(Monthly_Quantity_3scenarios_all'))*100;
        end

    end
end



%% Now let's draw the box plots of the distributions
%% For each combo of C3DP, draw four boxes:
%%      - 1x Baseline, 2x Baselien, 3x Baseline, No 3DP

tick_labels = {'0.5x', '1x', '2x', '3x', 'No 3DP'};

Mean_Relative_Demand_shortfall_varying_c3DP = {};

for i = 1 : length(speed_per_machine_month)  
    for j = 1:length(cost_of_3dp_per_machine_month)

        for ll = 1 : length(c_3DP_rate_set)+1
            boxplot(Relative_Demand_shortfall_varying_c3DP{i,j,ll}, 'Positions', ll, 'Widths', 0.5);
            hold on
        end
        boxplot(Relative_Demand_shortfall_no3DP_Benchmark, 'Positions', length(c_3DP_rate_set)+2 , 'Widths', 0.5);

        xticks(1:length(tick_labels)); 
        xticklabels(tick_labels);

        xlabel('$c^{\mathsf{cap}}$ (Multiple of Baseline)', 'FontSize', 12, 'interpreter', 'latex');
        ylabel('Shortfall (% of Max Demand)', 'FontSize', 12);
        ytickformat('percentage');
        grid on;

        % Plot the mean shortdemand
        for ll = 1 : length(c_3DP_rate_set)+1
            Mean_Relative_Demand_shortfall_varying_c3DP{i,j}(ll) = mean(Relative_Demand_shortfall_varying_c3DP{i,j,ll});
        end
        hold on
        plot([1:length(c_3DP_rate_set)+2], [Mean_Relative_Demand_shortfall_varying_c3DP{i,j}, mean(Relative_Demand_shortfall_no3DP_Benchmark) ], ...
            '-o', 'LineWidth',2)
    
        filename = strcat('Experiment_Results/Relative_Cost_Savings_Shortfalls_c3DP/Varying_c3DP(Shortfalls)/boxplots_3dp_varying_c3DP_C3DPcase_', num2str(i), '_', num2str(j), '.pdf');
        saveas(gcf, filename);
        close(gcf)


    end
end



%% Save the data for python
% Initialize variables
i = 1; j = 1;

boxplot_data = [];
mean_data = [];
tick_labels = {'0.5', '1x', '2x', '3x', 'No 3DP'};

% Collect data for boxplots and means
for ll = 1:length(c_3DP_rate_set)+1
    current_data = Relative_Demand_shortfall_varying_c3DP{i, j, ll};
    boxplot_data = [boxplot_data; [repmat(ll, length(current_data), 1), current_data(:)]];
    mean_data = [mean_data; ll, mean(current_data)];
end

% Add data for 'No 3DP'
no_3dp_data = Relative_Demand_shortfall_no3DP_Benchmark;
boxplot_data = [boxplot_data; [repmat(length(c_3DP_rate_set) + 2, length(no_3dp_data), 1), no_3dp_data(:)]];
mean_data = [mean_data; length(c_3DP_rate_set) + 2, mean(no_3dp_data)];

% Save mean data
mean_table = array2table(mean_data, 'VariableNames', {'Position', 'MeanShortfall'});
writetable(mean_table, 'Experiment_Results/Relative_Cost_Savings_Shortfalls_c3DP/varying_c3DP_for_python_shortfalls1.csv');

% Save boxplot data
boxplot_table = array2table(boxplot_data, 'VariableNames', {'Position', 'Shortfall'});
writetable(boxplot_table, 'Experiment_Results/Relative_Cost_Savings_Shortfalls_c3DP/varying_c3DP_for_python_shortfalls2.csv');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



