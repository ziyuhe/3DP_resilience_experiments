% =========================================================================
% Script Name:       Experiments_CostSavings_and_DemandShortfalls_Varying_3DPFixedCost.m
% Author:            Ziyu He
% Date:              02/01/2025
% Description:       
%   - This is part of a series of experiments prefixed with "Experiments_CostSavings_and_DemandShortfalls".
%   - Evaluates the impact of key hyperparameters on the performance of the 3DP resilience strategy.
%   - Specifically, here we analyze effect of varying 3DP reservation costs (c_cap) on resilience performance.
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
%   - **Focus**: Analyzing the impact of **c_cap**, defined as:
%       c_cap = cost_of_3dp_per_machine_month(i) / speed_per_machine_month(j).
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
  

%% In this experiment, we focus on analyzing the impact of **c_cap**,  
% where **c_cap** varies as follows:  
%   c_cap = cost_of_3dp_per_machine_month(i) / speed_per_machine_month(j).  



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMPUTE THE OPTIMAL 3DP POLICY AND COSTS UNDER DIFFERENT c_cap
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%% Preprocessing: Compute costs under different backup strategies
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
        
% 2. All backed-up by 3DP (infinite capacity)
input_medium_3DP_infK = input_medium_no3dp;
input_medium_3DP_infK.v = c_3DP;
output_medium_3DP_infK = Cost_No3DP_or_TM(input_medium_3DP_infK);

% 3. All backed-up by TM
TM_retainer_ratio = 0.75;
C_TM = TM_retainer_ratio * c_source .* mean_demand_3scenarios;
input_medium_TM = input_medium_no3dp;
input_medium_TM.TM_flag = 1;
input_medium_TM.c_TM = c_TM; 
input_medium_TM.C_TM = C_TM; 
output_medium_TM = Cost_No3DP_or_TM(input_medium_TM);

% Identify products backed up by TM
TM_backup_set = output_medium_TM.TM_cost < output_medium_no3dp.opt_val;
nobackup_set = 1 - TM_backup_set;
COST_TMONLY_BENCMARK = sum(output_medium_TM.TM_cost(TM_backup_set)) + sum(output_medium_no3dp.opt_val(logical(nobackup_set)));

%% Sampling data for BoE Submod Max SAA
input_preprocess_medium_sampled.num_suppliers = num_suppliers;
input_preprocess_medium_sampled.num_scenarios = num_scenarios;
input_preprocess_medium_sampled.yield_loss_rate = yield_loss_rate_medium;
input_preprocess_medium_sampled.p_disrupt = p_medium;
input_preprocess_medium_sampled.Monthly_Quantity = Monthly_Quantity_3scenarios;
input_preprocess_medium_sampled.Demand_Probability = Demand_Probability_3scenarios;

input_preprocess_medium_sampled.sample_mode = 2;
input_preprocess_medium_sampled.disruption_sample_flag = 1; % Keep all combinations
input_preprocess_medium_sampled.demand_sample_flag = 1;     % Sample fixed number of demand scenarios per disruption

% Set sample sizes based on number of suppliers
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

%% Prepare data for BoE Submod Max
Obj_const = - output_medium_no3dp.opt_val;
U0_with_vmean = output_medium_no3dp.opt_val + v .* mean_demand_3scenarios;
U0_no_vmean = output_medium_no3dp.opt_val;
TM_Delta = output_medium_TM.TM_cost - output_medium_no3dp.opt_val;

ratio_over_weight = (v - c_3DP) ./ weight;

% Optimal ordering when no backup
q0 = output_medium_no3dp.opt_q;

% Compute probabilities of unmet demand
pi_p = [];
pi_0 = [];

for j = 1:num_suppliers
    tmp1 = max(0, Monthly_Quantity_3scenarios(j, :)' - q0(j) * [1 - yield_loss_rate_medium, 1]);
    tmp2 = Demand_Probability_3scenarios(j, :)' * [p_medium, 1 - p_medium];

    pi_p(j, :) = sum(sum(tmp2(tmp1 > 1e-5)));
    pi_0(j, :) = sum(sum(tmp2(tmp1 <= 1e-5)));
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

input_boe.q_ub = output_medium_no3dp.opt_q;    
input_boe.q_lb = output_medium_3DP_infK.opt_q; 

%% Benchmark case: Compute cost under different 3DP capacities
OUTPUT_MEDIUM_BOE_BENCHMARK = {};

capacity_3dp_percentage = [1e-2, 1e-1 * [1:9], 1:2:25, 30:5:50, 75, 100] / 100;

for k = 1:length(capacity_3dp_percentage)
    fprintf("%3.2f Percent of Max Yield Shortfall \n\n", capacity_3dp_percentage(k) * 100)

    capacity_percentage = capacity_3dp_percentage(k);
    K_3DP_medium = capacity_percentage * sum(max(Monthly_Weight_3scenarios'));
    C_3DP_medium = (K_3DP_medium ./ speed_per_machine_month)' * cost_of_3dp_per_machine_month;
    
    input_boe.K_3DP = K_3DP_medium;
    input_boe.C_3DP = C_3DP_medium;

    input_boe.GRB_display = 0;
    input_boe.auto_recompute = 1;
    
    if k == 1
        input_boe.A_init = [];
    else
        input_boe.A_init = OUTPUT_MEDIUM_BOE_BENCHMARK{k-1}.A_t;
    end

    if num_suppliers <= 10
        output_boe = BoE_Approx_Max_Submod_Exact(input_boe);
    else
        output_boe = BoE_Approx_Max_Submod_SAA(input_boe);
    end
    
    OUTPUT_MEDIUM_BOE_BENCHMARK{k} = output_boe;
end

save("Experiment_Data/Relative_Cost_Savings_Shortfall_Varying_3DPFixedCost/Benchmark.mat")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%











%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VARYING c_cap: PLOT COST-SAVINGS VS. K
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% NOTE: These plots are for initial verification. The final plots in the paper are generated in Python using the data saved here.

% Summarize relative cost-savings (percentage of benchmark cost)
COST_SAVINGS_BENCHMARK = [];
for i = 1:length(speed_per_machine_month)
    for j = 1:length(cost_of_3dp_per_machine_month)
        for k = 1:length(capacity_3dp_percentage)
            COST_SAVINGS_BENCHMARK{i,j}(k) = (COST_TMONLY_BENCMARK - OUTPUT_MEDIUM_BOE_BENCHMARK{k}.TOTAL_COST_NONZERO(i,j)) / abs(COST_TMONLY_BENCMARK) * 100;
        end
    end
end

% Plot relative cost-savings vs. K under different printer prices (fixed printing speed)
cost_of_3dp_per_machine_month_subset = [1,3,7,9];
for i = 1:length(speed_per_machine_month)
    for jj = 1:length(cost_of_3dp_per_machine_month_subset)
        j = cost_of_3dp_per_machine_month_subset(jj);
        plot([0, capacity_3dp_percentage] * 100, [0, COST_SAVINGS_BENCHMARK{i,j}], '-o', 'LineWidth', 1);
        hold on;
    end
    
    ylimit = max(COST_SAVINGS_BENCHMARK{i,1});
    ylim([-1.1 * ylimit, 1.1 * ylimit]);
    xlim([0, 100]);

    lgd = legend({'1x Baseline', '3x Baseline', '5x Baseline', '7x Baseline'}, 'FontSize', 12);
    title(lgd, 'Cost per unit 3DP capacity');

    ax = gca;
    ax.XTickLabel = strcat(ax.XTickLabel, '%');
    ax.YTickLabel = strcat(ax.YTickLabel, '%');

    filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfall_Varying_3DPFixedCost/Benchmark_Varying_C3DP_Coeffs/Varying_C3DP_Coeff_Fixed_Speed', num2str(i), '.pdf');
    
    saveas(gcf, filename);
    close(gcf);
end

% Plot relative cost-savings vs. K under different printing speeds (fixed printer price)
for jj = 1:length(cost_of_3dp_per_machine_month_subset)
    j = cost_of_3dp_per_machine_month_subset(jj);
    
    for i = 1:length(speed_per_machine_month)
        plot([0, capacity_3dp_percentage] * 100, [0, COST_SAVINGS_BENCHMARK{i,j}], '-o', 'LineWidth', 1);
        hold on;
    end
    
    ylimit = max(COST_SAVINGS_BENCHMARK{3,j});
    ylim([-1.1 * ylimit, 1.1 * ylimit]);
    xlim([0, 100]);

    lgd = legend({'1x Baseline', '0.4x Baseline', '0.2x Baseline'}, 'FontSize', 12);
    title(lgd, 'Cost per unit 3DP capacity');

    ax = gca;
    ax.XTickLabel = strcat(ax.XTickLabel, '%');
    ax.YTickLabel = strcat(ax.YTickLabel, '%');

    filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfall_Varying_3DPFixedCost/Benchmark_Varying_C3DP_Coeffs/Varying_C3DP_Coeff_Fixed_Printer_Cost', num2str(j), '.pdf');
    
    saveas(gcf, filename);
    close(gcf);
end

% Combine all results
% Fix "cost_of_3dp_per_machine_month" as case 1: plot "speed_per_machine_month" cases 2, 3
% Fix "speed_per_machine_month" as case 1: plot "cost_of_3dp_per_machine_month" cases 1, 3, 7, 9

figure;
hold on;

iiii = [1, 11:length(capacity_3dp_percentage)];

for i = length(speed_per_machine_month):-1:1
    plot([0, capacity_3dp_percentage(iiii)] * 100, [0, COST_SAVINGS_BENCHMARK{i,1}(iiii)], '-o', 'LineWidth', 1);
    hold on;
end

cost_of_3dp_per_machine_month_subset = [1,3,7,9];

for jj = 2:length(cost_of_3dp_per_machine_month_subset)
    j = cost_of_3dp_per_machine_month_subset(jj);
    plot([0, capacity_3dp_percentage(iiii)] * 100, [0, COST_SAVINGS_BENCHMARK{1,j}(iiii)], '-o', 'LineWidth', 1);
    hold on;
end

% Add horizontal reference line
yline(0, '--', 'LineWidth', 1.5, 'Color', [0.5, 0.5, 0.5]);

% Add shaded area
x_fill = [0, 100, 100, 0];
y_fill = [0, 0, -1e6, -1e6];
fill(x_fill, y_fill, [0.8, 0.8, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);

ylimit = max(COST_SAVINGS_BENCHMARK{3,1});
ylim([-1.1 * ylimit, 1.1 * ylimit]);
xlim([0, 50]);
grid on;

lgd = legend({'0.2x Baseline', '0.4x Baseline', '1x Baseline', '3x Baseline', '5x Baseline', '7x Baseline'}, 'FontSize', 12, 'Location', 'southeast');
title(lgd, 'Cost per unit 3DP capacity');

filename = 'Experiment_Data/Relative_Cost_Savings_Shortfall_Varying_3DPFixedCost/Benchmark_Varying_C3DP_Coeffs/Varying_C3DP_Coeff_All_in_One.pdf';

ax = gca;
ax.XTickLabel = strcat(ax.XTickLabel, '%');
ax.YTickLabel = strcat(ax.YTickLabel, '%');

saveas(gcf, filename);
close(gcf);



%% Save data to CSV for python

% Prepare the data
iiii = [1, 11:length(capacity_3dp_percentage)];
capacity_selected = capacity_3dp_percentage(iiii) * 100; % X-axis data

% Initialize an empty matrix to collect data
num_rows = length(capacity_selected)+1;
num_curves = length(speed_per_machine_month) + length(cost_of_3dp_per_machine_month_subset) - 1;
data_matrix = zeros(num_rows, num_curves + 1);  % Extra column for capacity

% Add capacity as the first column
data_matrix(:, 1) = [0, capacity_selected];

% Add COST_SAVINGS_BENCHMARK for each curve
col_idx = 2;  % Start adding data from column 2
for i = length(speed_per_machine_month):-1:1
    data_matrix(:, col_idx) = [0; COST_SAVINGS_BENCHMARK{i, 1}(iiii)'];
    col_idx = col_idx + 1;
end

cost_of_3dp_per_machine_month_subset = [1, 3, 7, 9];
for jj = 2:length(cost_of_3dp_per_machine_month_subset)
    j = cost_of_3dp_per_machine_month_subset(jj);
    data_matrix(:, col_idx) = [0; COST_SAVINGS_BENCHMARK{1, j}(iiii)'];
    col_idx = col_idx + 1;
end

% Save to CSV
file_name = 'Experiment_Data/Relative_Cost_Savings_Shortfall_Varying_3DPFixedCost/benchmark_for_python.csv';
headers = ['Capacity_3DP_Percentage', ...
           strcat('Curve_', string(1:num_curves))];  % Generate column headers
fid = fopen(file_name, 'w');
fprintf(fid, '%s,', headers{1:end-1});
fprintf(fid, '%s\n', headers{end});

% Write the data
fclose(fid);
dlmwrite(file_name, data_matrix, '-append', 'precision', '%.4f');

fprintf('Data saved to %s\n', file_name);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
















%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VARYING c_cap: PLOT DEMAND SHORTFALLS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SYSTEM DEFINITIONS:
% - "SINGLE" system: No 3DP backup.
% - "DUO" system: Includes 3DP backup.

% NOTE: These plots are for initial verification. The final plots in the paper are generated in Python using the data saved here.

%% SINGLE SYSTEM: No 3DP
% - Identify products without backup (A0).
% - Sample pairs of demand and disruptions.
% - For each sample (D, s), compute the demand shortfall [D_j - q_j*s_j]^+ and sum over j in A0.
% - Save the distribution as a vector.

num_suppliers = sum(nobackup_set);
supplier_subset_idx = logical(nobackup_set);
Monthly_Weight_3scenarios = Monthly_Weight_3scenarios_all(supplier_subset_idx, :);
Monthly_Quantity_3scenarios = Monthly_Quantity_3scenarios_all(supplier_subset_idx, :);
Demand_Probability_3scenarios = Demand_Probability_3scenarios_all(supplier_subset_idx, :);
mean_demand_3scenarios = mean_demand_3scenarios_all(supplier_subset_idx);

% Generate demand and disruption samples
input_preprocess_medium_sampled.num_suppliers = num_suppliers;
input_preprocess_medium_sampled.num_scenarios = num_scenarios;
input_preprocess_medium_sampled.yield_loss_rate = yield_loss_rate_medium;
input_preprocess_medium_sampled.p_disrupt = p_medium;
input_preprocess_medium_sampled.Monthly_Quantity = Monthly_Quantity_3scenarios;
input_preprocess_medium_sampled.Demand_Probability = Demand_Probability_3scenarios;

input_preprocess_medium_sampled.sample_mode = 2;
input_preprocess_medium_sampled.disruption_sample_flag = 1; % No sampling of disruptions (keep all combinations)
input_preprocess_medium_sampled.demand_sample_flag = 1;     % For each disruption scenario, sample a fixed number of demand values

disruption_samplesize_saa = 1000;    
demand_samplesize_saa = 100;

input_preprocess_medium_sampled.demand_num_per_disruption = demand_samplesize_saa;
input_preprocess_medium_sampled.disruption_samplesize = disruption_samplesize_saa;

output_preprocess_medium_sampled = Data_prep_for_MIP(input_preprocess_medium_sampled);

disruption_demand_joint_prob_medium_sampled = output_preprocess_medium_sampled.disruption_demand_joint_prob;
failure_data_medium_sampled = output_preprocess_medium_sampled.failure_data;
demand_data_medium_sampled = output_preprocess_medium_sampled.demand_data;

% Compute demand shortfall distribution for the SINGLE system (without 3DP)
opt_q_nobackup = output_medium_no3dp.opt_q(logical(nobackup_set));
Demand_shortfall_no3DP_Benchmark = [];

for ss = 1:length(disruption_demand_joint_prob_medium_sampled)
    D_sample = demand_data_medium_sampled(:, ss); 
    s_sample = failure_data_medium_sampled(:, ss);
    
    Demand_shortfall_no3DP_Benchmark(ss, 1) = sum(max(D_sample - opt_q_nobackup .* s_sample, 0));
end

%% DUO SYSTEM: With 3DP
% - For each capacity K on a predefined grid:
%   - Identify the set of products backed up by 3DP (A).
%   - Sample pairs of demand and disruptions.
%   - For each sample (D, s), compute the optimal 3DP decision q_3DP(D,s) and the total demand shortfall [D_j - q_j*s_j - q_j^3DP(D,s)]^+.
%   - Save the distribution as a vector.

% NOTES:
% - For the same K, demand shortfall remains the same for all c_cap values (q_SP is identical).
% - Different c_cap values yield different optimal K values, affecting demand shortfall.

% Compute demand shortfall distribution for each c_cap:
% - Identify the optimal K (max cost savings) for each c_cap.
% - Retrieve the corresponding x and q_SP at the optimal K.
% - Compute demand shortfall.

Demand_shortfall_varying_C3DP = {};

for i = 1:length(speed_per_machine_month)
    for j = 1:length(cost_of_3dp_per_machine_month)
        
        fprintf("Processing case %d, %d\n", i, j);

        % Retrieve the optimal x and q_SP
        [~, kkk] = min(TOTAL_COST_BENCHMARK{i, j});
        x_3DP = logical(OUTPUT_MEDIUM_BOE_BENCHMARK{1, kkk}.X_FINAL{i, j});
        q_SP = OUTPUT_MEDIUM_BOE_BENCHMARK{1, kkk}.Q_FINAL{i, j}(x_3DP);

        % Prepare demand and disruption data for the subset of products backed by 3DP
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
        K_3DP_medium = capacity_percentage * sum(max(Monthly_Weight_3scenarios'));

        % Generate demand and disruption samples
        input_preprocess_medium_sampled.num_suppliers = num_suppliers;
        input_preprocess_medium_sampled.num_scenarios = num_scenarios;
        input_preprocess_medium_sampled.yield_loss_rate = yield_loss_rate_medium;
        input_preprocess_medium_sampled.p_disrupt = p_medium;
        input_preprocess_medium_sampled.Monthly_Quantity = Monthly_Quantity_3scenarios;
        input_preprocess_medium_sampled.Demand_Probability = Demand_Probability_3scenarios;
        
        input_preprocess_medium_sampled.sample_mode = 2;
        input_preprocess_medium_sampled.disruption_sample_flag = 1; % No sampling of disruptions (keep all combinations)
        input_preprocess_medium_sampled.demand_sample_flag = 1;     % Sample a fixed number of demand values per disruption scenario
        
        disruption_samplesize_saa = 1000;    
        demand_samplesize_saa = 100;            
        
        input_preprocess_medium_sampled.demand_num_per_disruption = demand_samplesize_saa;
        input_preprocess_medium_sampled.disruption_samplesize = disruption_samplesize_saa;
        
        output_preprocess_medium_sampled = Data_prep_for_MIP(input_preprocess_medium_sampled);
        
        disruption_demand_joint_prob_medium_sampled = output_preprocess_medium_sampled.disruption_demand_joint_prob;
        failure_data_medium_sampled = output_preprocess_medium_sampled.failure_data;
        demand_data_medium_sampled = output_preprocess_medium_sampled.demand_data;
        
        % Compute demand shortfall for each (D,s) pair
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
            
            Demand_shortfall_varying_C3DP{i, j}(ss) = sum(max(D_bar - q_3DP, 0));
        end
    end
end





%% COMPUTING RELATIVE DEMAND SHORTFALL ANALYSIS

% Compute relative demand shortfall (percentage of total max demand)
Relative_Demand_shortfall_varying_C3DP = {};
for i = 1:length(speed_per_machine_month)  
    for j = 1:length(cost_of_3dp_per_machine_month)
        Relative_Demand_shortfall_varying_C3DP{i,j} = Demand_shotfall_varying_C3DP{i,j} / sum(max(Monthly_Quantity_3scenarios_all')) * 100;
    end
end
Relative_Demand_shortfall_no3DP_Benchmark = Demand_shortfall_no3DP_Benchmark / sum(max(Monthly_Quantity_3scenarios_all')) * 100;


% BOX PLOTS OF DEMAND SHORTFALL DISTRIBUTIONS
% Compare different c_cap values side by side
bbb = cost_of_3dp_per_machine_month(1) / speed_per_machine_month(1);

for i = 1:length(speed_per_machine_month)  

    % Initialize tick labels
    tick_labels = cell(1, length(cost_of_3dp_per_machine_month) + 1);
    
    % Generate boxplots for varying c_cap
    for j = 1:length(cost_of_3dp_per_machine_month)
        multiples(i,j) = (cost_of_3dp_per_machine_month(j) / speed_per_machine_month(i)) / bbb;
        boxplot(Relative_Demand_shortfall_varying_C3DP{i,j}, 'Positions', j, 'Widths', 0.5);
        tick_labels{j} = strcat(num2str(multiples(i,j), '%.2f'), 'x');  
        hold on;
    end

    % Add the reference boxplot for 'No 3DP'
    boxplot(Relative_Demand_shortfall_no3DP_Benchmark, 'Positions', length(cost_of_3dp_per_machine_month) + 1 , 'Widths', 0.5);
    tick_labels{end} = 'No 3DP';
    
    % Set axis labels and formatting
    xticks(1:length(tick_labels)); 
    xticklabels(tick_labels);
    xlabel('$c^{\mathsf{cap}}$ (Multiple of Baseline)', 'FontSize', 12, 'Interpreter', 'latex');
    ylabel('Shortfall (% of Max Demand)', 'FontSize', 12);
    ytickformat('percentage');
    grid on;
    
    % Plot the mean demand shortfall
    for j = 1:length(cost_of_3dp_per_machine_month)
        Mean_Relative_Demand_shortfall_varying_C3DP(i,j) = mean(Relative_Demand_shortfall_varying_C3DP{i,j});
    end
    plot([1:length(cost_of_3dp_per_machine_month) + 1], [Mean_Relative_Demand_shortfall_varying_C3DP(i,:), mean(Relative_Demand_shortfall_no3DP_Benchmark)], ...
        '-o', 'LineWidth', 2);

    filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfall_Varying_3DPFixedCost/Benchmark_Varying_C3DP_Coeffs(Shortfalls)/boxplots_3dp_varying_C3DP_fixed_speed', num2str(i), '.pdf');
    saveas(gcf, filename);
    close(gcf)

end

%% Define the baseline c_cap using speed case 1 and cost per machine case 1
%% Generate box plots for demand shortfall distributions and mean demand shortfall for the following cases:
%%      - 0.2x, 0.4x, 1x, 3x, 5x, No 3DP

C3DP_subset = [3,1; 2,1; 1,1; 1,3; 1,7];

tick_labels = {'0.2x', '0.4x', '1x', '3x', '5x', 'No 3DP'};

Mean_Relative_Demand_shortfall_varying_C3DP_subset = [];
for ll = 1:length(C3DP_subset)
    ttt = C3DP_subset(ll,:);
    i = ttt(1);
    j = ttt(2);
    Mean_Relative_Demand_shortfall_varying_C3DP_subset(ll) = Mean_Relative_Demand_shortfall_varying_C3DP(i,j);
end
plot([1:length(C3DP_subset)+1], [Mean_Relative_Demand_shortfall_varying_C3DP_subset, mean(Relative_Demand_shortfall_no3DP_Benchmark)], ...
    '-o', 'LineWidth', 2)
hold on

for ll = 1:length(C3DP_subset)
    ttt = C3DP_subset(ll,:);
    i = ttt(1);
    j = ttt(2);
    
    boxplot(Relative_Demand_shortfall_varying_C3DP{i,j}, 'Positions', ll, 'Widths', 0.5);
    hold on
end

% Add boxplot for 'No 3DP'
boxplot(Relative_Demand_shortfall_no3DP_Benchmark, 'Positions', length(C3DP_subset)+1 , 'Widths', 0.5);

xticks(1:length(tick_labels)); 
xticklabels(tick_labels);

xlabel('$c^{\mathsf{cap}}$ (Multiple of Baseline)', 'FontSize', 12, 'interpreter', 'latex');
ylabel('Shortfall (% of Max Demand)', 'FontSize', 12);
ytickformat('percentage');
grid on;
    
legend({'Mean Demand Shortfall', 'Distribution of Demand Shortfall'})

filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfall_Varying_3DPFixedCost/Benchmark_Varying_C3DP_Coeffs(Shortfalls)/boxplots_3dp_varying_C3DP_all_in_one.pdf');
saveas(gcf, filename);
close(gcf)

%% Save data for Python analysis
% Initialize arrays for means and boxplot data
Mean_Shortfall_python = [];  
boxplot_data = [];

% Collect mean data for each subset
for ll = 1:length(C3DP_subset)
    ttt = C3DP_subset(ll,:);
    i = ttt(1);
    j = ttt(2);
    Mean_Shortfall_python(ll) = Mean_Relative_Demand_shortfall_varying_C3DP(i,j);
    
    % Collect boxplot data for this subset
    current_data = Relative_Demand_shortfall_varying_C3DP{i, j};
    boxplot_data = [boxplot_data; [repmat(ll, length(current_data), 1), current_data(:)]];
end

% Add 'No 3DP' data
Mean_Shortfall_python = [Mean_Shortfall_python, mean(Relative_Demand_shortfall_no3DP_Benchmark)];
no_3dp_data = Relative_Demand_shortfall_no3DP_Benchmark;
boxplot_data = [boxplot_data; [repmat(length(C3DP_subset)+1, length(no_3dp_data), 1), no_3dp_data(:)]];

% Save mean data
mean_table = array2table([1:length(tick_labels); Mean_Shortfall_python]', ...
    'VariableNames', {'Position', 'MeanShortfall'});
writetable(mean_table, 'Experiment_Data/Relative_Cost_Savings_Shortfall_Varying_3DPFixedCost/benchmark_for_python_shortfalls1.csv');

% Save boxplot data
boxplot_table = array2table(boxplot_data, 'VariableNames', {'Position', 'Shortfall'});
writetable(boxplot_table, 'Experiment_Data/Relative_Cost_Savings_Shortfall_Varying_3DPFixedCost/benchmark_for_python_shortfalls2.csv');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





