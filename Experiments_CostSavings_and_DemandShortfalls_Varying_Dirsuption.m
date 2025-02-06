% =========================================================================
% Script Name:       Experiments_CostSavings_and_DemandShortfalls_Varying_Disruption.m
% Author:            Ziyu He
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
%   - **Focus**: Analyzing the impact of **p and yield loss rate**.
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
  

%% In this experiment, we focus on analyzing the impact of:
%   - Marginal failure rate
%   - Yieldloss ratio


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DISRUPTION MODELING (INDEPENDENT): COMPUTATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% NOW WE TRY DIFFERENT DISRUPTION MODELING (INDEPENDENT)
%  - baseline is (p, yield_loss_rate) = (0.05, 0.05)
%  - we can try: 
%          - p = 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5
%          - yield_loss_rate_set = 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7


p_set = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5];
yield_loss_rate_set = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];


OUTPUT_MEDIUM_BOE_VARYING_DISRUPTIONS = {};
COST_TMONLY_VARYING_DISRUPTIONS = {};


capacity_3dp_percentage = [1e-2, 1e-1*[1:9], 1:2:25, 30:5:50, 75, 100]/100;

for pp = 1 : length(p_set)
    for yy = 1 : length(yield_loss_rate_set)

        if (pp ~= 2) || (yy ~= 2)

            p_medium = p_set(pp);
            yield_loss_rate_medium = yield_loss_rate_set(yy);
    
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
            COST_TMONLY_VARYING_DISRUPTIONS{pp,yy} = sum(output_medium_TM.TM_cost(TM_backup_set))+sum(output_medium_no3dp.opt_val(logical(nobackup_set)));
            
                    
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
            
                input_boe.recompute_sgd_Maxsteps = 5e5;
                
                if k == 1
                    input_boe.A_init = [];
                else
                    input_boe.A_init = OUTPUT_MEDIUM_BOE_VARYING_DISRUPTIONS{pp, yy, k-1}.A_t;
                end
            
                if num_suppliers <= 10
                    output_boe = BoE_Approx_Max_Submod_Exact(input_boe);
                else
                    output_boe = BoE_Approx_Max_Submod_SAA(input_boe);
                end
                          
                OUTPUT_MEDIUM_BOE_VARYING_DISRUPTIONS{pp, yy, k} = output_boe; 
            
                disp("TO NEXT ONE! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ")
            
            end
    

        
        end

    end
end


save("Experiment_Data/Relative_Cost_Savings_Shortfalls_Varying_p_yieldloss/Varying_DisruptionDistr_Ind_Comono_all_in_one.mat")

DDD = load("Experiment_Data/Relative_Cost_Savings_Shortfalls_Varying_p_yieldloss/Benchmark.mat");

COST_TMONLY_BENCMARK = DDD.COST_TMONLY_BENCMARK;
OUTPUT_MEDIUM_BOE_BENCHMARK = DDD.OUTPUT_MEDIUM_BOE_BENCHMARK;

%% Put the benchmark case back to the list
for k = 1 : length(capacity_3dp_percentage)
    OUTPUT_MEDIUM_BOE_VARYING_DISRUPTIONS{2,2,k} = OUTPUT_MEDIUM_BOE_BENCHMARK{k};
end
COST_TMONLY_VARYING_DISRUPTIONS{2,2} = COST_TMONLY_BENCMARK;

%% Summarize the relative cost-savings (PERCENTAGE OF BASELINE COST)

COST_SAVINGS_VARYING_DISRUPTIONS = {};
for pp = 1 : length(p_set)
    for yy = 1 : length(yield_loss_rate_set)

        for i = 1 : length(speed_per_machine_month)  
            for j = 1 : length(cost_of_3dp_per_machine_month)
        
                for k = 1 : length(capacity_3dp_percentage)
                        
                    COST_SAVINGS_VARYING_DISRUPTIONS{pp,yy,i,j}(k) = (COST_TMONLY_VARYING_DISRUPTIONS{pp,yy} - OUTPUT_MEDIUM_BOE_VARYING_DISRUPTIONS{pp,yy, k}.TOTAL_COST_NONZERO(i,j)) / abs(COST_TMONLY_VARYING_DISRUPTIONS{pp,yy})*100;
                    
                end
        
            end
        end

    end
end








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VARYING p AND yieldloss (INDEPENDENT DISRUPTIONS): PLOT COST-SAVINGS VS. K
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Fix all hyperparameters at default values 
%  - Notably, set c_cap = speed_per_machine_month(i) / cost_of_3dp_per_machine_month(j)

%% Generate three plots to analyze cost savings trends:
%  - **Plot 1:** For each disruption probability (p), vary the yield loss rate and record the maximum cost savings across K.
%  - **Plot 2:** For each yield loss rate, vary p and record the maximum cost savings across K.
%  - **Plot 3:** Compare cost savings across five key (p, yield_loss_rate) combinations:
%       - High p, High yield loss rate (High-High)
%       - High p, Low yield loss rate (High-Low)
%       - Low p, High yield loss rate (Low-High)
%       - Low p, Low yield loss rate (Low-Low)
%       - Moderate p, Moderate yield loss rate (Mid-Mid)

%% Insights from the first two plots:
%  - Cost savings generally follow a non-monotonic trend, first increasing and then decreasing with both parameters.
%  - The "High-High" scenario is expected to be the least favorable, while other cases perform reasonably well.


p_subset = [1,7,11];
yield_loss_rate_subset = [1,6,9];


i = 1; j = 1;

%% Plot 1
for ppp = 1 : length(p_subset)
    pp = p_subset(ppp);
    plot(yield_loss_rate_set, COST_SAVINGS_VARYING_DISRUPTIONS_MAX_AMONG_K{i,j}(pp,:), '-o','LineWidth', 1, 'Color', colors2(ppp,:))
    hold on
end
lgd = legend({'1x Baseline', '5x Baseline', '10x Baseline'}, ...
    'FontSize', 12, 'location', 'eastoutside');
title(lgd, 'Marginal Disruption Rate');
xlabel("Yield Loss Rate", 'FontSize',15)

filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls_Varying_p_yieldloss/Varying_p_yield_loss(CostSavings)/COSTSAVINGS_FIXED_YIELDLOSS_CASE', num2str(i), num2str(j), '.pdf');
hold off
saveas(gcf,filename)  % as MATLAB figure
close(gcf)
          
%% Plot 2
for yyy = 1 : length(yield_loss_rate_subset)
    yy = yield_loss_rate_subset(yyy);
    plot(p_set, COST_SAVINGS_VARYING_DISRUPTIONS_MAX_AMONG_K{i,j}(:,yy)', '-o','LineWidth', 1, 'Color', colors1(yyy,:))
    hold on
end
lgd = legend({'1x Baseline', '8x Baseline', '14x Baseline'}, ...
    'FontSize', 12, 'location', 'eastoutside');
title(lgd, 'Yield Loss Rate');
xlabel("Marginal Failure Rate", 'FontSize',15)

filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls_Varying_p_yieldloss/Varying_p_yield_loss(CostSavings)/COSTSAVINGS_FIXED_P_CASE', num2str(i), num2str(j), '.pdf');
hold off
saveas(gcf,filename)  % as MATLAB figure
close(gcf)


%% Plot 3
p_yield_combo = [11,9; 11,1; 7,6; 1,9; 1,1];

for combo = 1 : length(p_yield_combo)

    ttt = p_yield_combo(combo,:);
    pp = ttt(1); yy = ttt(2);

    plot([0,capacity_3dp_percentage]*100, [0,COST_SAVINGS_VARYING_DISRUPTIONS{pp,yy,i,j}], '-o','LineWidth', 1, 'Color', colors3(combo,:))
    hold on

end

xlim([0,30])
gd = legend({'High-High', 'High-Low', 'Mid-Mid', 'Low-High', 'Low-Low'}, 'FontSize', 12, 'location', 'southeast'); 
title(lgd, '(Freq, Yield Loss)');

filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls_Varying_p_yieldloss/Varying_p_yield_loss(CostSavings)/HIGH_MID_LOW_C3DP_CASE', num2str(i), num2str(j), '.pdf');

ax = gca;
ax.XTickLabel = strcat(ax.XTickLabel, '%');
ax.YTickLabel = strcat(ax.YTickLabel, '%');

hold off
saveas(gcf,filename)  % as MATLAB figure
close(gcf)


%% Save CVS data for python
plot1_data = [yield_loss_rate_set', COST_SAVINGS_VARYING_DISRUPTIONS_MAX_AMONG_K{i,j}(p_subset,:)'];
plot2_data = [p_set', COST_SAVINGS_VARYING_DISRUPTIONS_MAX_AMONG_K{i,j}(:, yield_loss_rate_subset)];

plot3_data = [0,capacity_3dp_percentage]'*100;
for combo = 1 : length(p_yield_combo)

    ttt = p_yield_combo(combo,:);
    pp = ttt(1); yy = ttt(2);

    plot3_data(:,combo+1) = [0,COST_SAVINGS_VARYING_DISRUPTIONS{pp,yy,i,j}]';

end

filename = 'Experiment_Data/Relative_Cost_Savings_Shortfalls_Varying_p_yieldloss/varying_disruption_distr_ind_for_python_costsavings.xlsx';

% Save Plot 1
plot1_labels = ["Yield Loss Rate", "Plot1_Series1", "Plot1_Series2", "Plot1_Series3"];
T1 = array2table(plot1_data, 'VariableNames', plot1_labels);
writetable(T1, filename, 'Sheet', 'Plot 1');

% Save Plot 2
plot2_labels = ["Marginal Failure Rate", "Plot2_Series1", "Plot2_Series2", "Plot2_Series3"];
T2 = array2table(plot2_data, 'VariableNames', plot2_labels);
writetable(T2, filename, 'Sheet', 'Plot 2');

% Save Plot 3
plot3_labels = ["Capacity (%)", "High-High", "High-Low", "Mid-Mid", "Low-High", "Low-Low"];
T3 = array2table(plot3_data, 'VariableNames', plot3_labels);
writetable(T3, filename, 'Sheet', 'Plot 3');

















%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VARYING p AND yieldloss (INDEPENDENT DISRUPTIONS): PLOT DEMAND SHORTFALLS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SINGLE System: No 3DP
%  - Identify the set of products without backup (A0).
%  - Sample pairs of demand and disruptions.
%  - For each sample (D, s), compute the unmet demand [D_j - q_j * s_j]^+ and sum over j in A0.
%  - Store the resulting distribution as a vector.

%% Important Note:
%  - Even in the SINGLE system, the results vary with p (disruption probability) and yield loss rate.

%% DUO System: With 3DP Backup
%  - For each capacity K on a predefined grid:
%      - Identify the set of products backed up by 3DP (A).
%      - Sample pairs of demand and disruptions.
%      - For each sample (D, s), determine the optimal 3DP decision q3DP(D, s) and compute the unmet demand:
%        [D_j - q_j * s_j - q_j^3DP(D, s)]^+, summing over j in A.
%      - Store the resulting distribution as a vector.

%% Key Considerations:
%  - For the same K, the demand shortfall remains identical across all c_cap values since q_SP is the same.
%  - Different c_cap values lead to different optimal K, resulting in varying demand shortfalls.

%% Approach:
%  - First, fix a specific c_cap case ("11").
%  - Compute the demand shortfall distribution for each (p, yield_loss_rate) combination:
%      - For each (p, yield_loss_rate) pair, identify the optimal K that maximizes cost savings.
%      - Retrieve the corresponding x and q_SP at the optimal K and compute the demand shortfall.


i = 1; j = 1;

Relative_Demand_shortfall_no3DP_varying_disruptions = {};
Relative_Demand_shotfall_varying_disruptions = {};

for pp = 1 : length(p_set)
    for yy = 1 : length(yield_loss_rate_set)

        fprintf("Working with p case %d, yield loss case %d \n\n",   pp,    yy)
        
        %% A pair of p and yield loss
        p_medium = p_set(pp);
        yield_loss_rate_medium = yield_loss_rate_set(yy);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% NO 3DP CASES (SINGLE SYSTEM) ALSO CHANGES WITH p and yield losses (BUT INVARIANT OF C3DP)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
        
        
        % (PART OF PRE-PROCESSING) Compute the costs of each supplier when they are
        % - Backed-up by TM 
        % - No back-up at all 
        % - Backed-up by 3DP but inf. capacity
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
        
        if sum(nobackup_set) > 0

            num_suppliers = sum(nobackup_set);
            supplier_subset_idx = logical(nobackup_set);
            Monthly_Weight_3scenarios = Monthly_Weight_3scenarios_all(supplier_subset_idx, :);
            Monthly_Quantity_3scenarios = Monthly_Quantity_3scenarios_all(supplier_subset_idx, :);
            Demand_Probability_3scenarios = Demand_Probability_3scenarios_all(supplier_subset_idx, :);
            mean_demand_3scenarios = mean_demand_3scenarios_all(supplier_subset_idx);
                   
            %% Get some samples for (s,D) first 
            input_preprocess_medium_sampled.num_suppliers = num_suppliers;
            input_preprocess_medium_sampled.num_scenarios = num_scenarios;
            input_preprocess_medium_sampled.yield_loss_rate = yield_loss_rate_medium;
            input_preprocess_medium_sampled.p_disrupt = p_medium;
            input_preprocess_medium_sampled.Monthly_Quantity = Monthly_Quantity_3scenarios;
            input_preprocess_medium_sampled.Demand_Probability = Demand_Probability_3scenarios;
            
            input_preprocess_medium_sampled.sample_mode = 2;
            input_preprocess_medium_sampled.disruption_sample_flag = 1; % We don't sample disruption (keep all combinations)
            input_preprocess_medium_sampled.demand_sample_flag = 1;     % For each disruption scenario, sample a fixed number of demand combos
            
            disruption_samplesize_saa = 1000;    
            demand_samplesize_saa = 100;
                 
            input_preprocess_medium_sampled.demand_num_per_disruption = demand_samplesize_saa;
            input_preprocess_medium_sampled.disruption_samplesize = disruption_samplesize_saa;
            
            output_preprocess_medium_sampled = Data_prep_for_MIP(input_preprocess_medium_sampled);
            
            disruption_demand_joint_prob_medium_sampled = output_preprocess_medium_sampled.disruption_demand_joint_prob;
            failure_data_medium_sampled = output_preprocess_medium_sampled.failure_data;
            demand_data_medium_sampled = output_preprocess_medium_sampled.demand_data;
            
            %% Get the distribution of demand shorfall under no 3DP
            opt_q_nobackup = output_medium_no3dp.opt_q(logical(nobackup_set));
            
            
            for ss = 1 : length(disruption_demand_joint_prob_medium_sampled)
            
                D_sample = demand_data_medium_sampled(:, ss); 
                s_sample = failure_data_medium_sampled(:, ss);
                
                Relative_Demand_shortfall_no3DP_varying_disruptions{pp,yy}(ss,1) = sum( max( D_sample - opt_q_nobackup.*s_sample, 0 ) ) / sum(max(Monthly_Quantity_3scenarios_all'))*100 ;
            
            end

        else

            %% When everybody is backedup by TM, then no shortfall
            Relative_Demand_shortfall_no3DP_varying_disruptions{pp,yy} = zeros(1000*100,1);

        end




        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% 3DP case
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [~,kkk] = max(COST_SAVINGS_VARYING_DISRUPTIONS{pp,yy,i,j});
        x_3DP = logical(OUTPUT_MEDIUM_BOE_VARYING_DISRUPTIONS{pp,yy, kkk}.X_FINAL{i,j});
        q_SP = OUTPUT_MEDIUM_BOE_VARYING_DISRUPTIONS{pp,yy, kkk}.Q_FINAL{i,j}(x_3DP);  

        if sum(x_3DP) > 0

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
            input_preprocess_medium_sampled.Monthly_Quantity = Monthly_Quantity_3scenarios;
            input_preprocess_medium_sampled.Demand_Probability = Demand_Probability_3scenarios;
            
            input_preprocess_medium_sampled.sample_mode = 2;
            input_preprocess_medium_sampled.disruption_sample_flag = 1; % We don't sample disruption (keep all combinations)
            input_preprocess_medium_sampled.demand_sample_flag = 1;     % For each disruption scenario, sample a fixed number of demand combos
            
            disruption_samplesize_saa = 1000;    
            demand_samplesize_saa = 100;            
            
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
                
                % if mod(ss,1e5) == 1
                %     fprintf("K case %d;   Sample case %d   \n",  kkk, ss)
                % end
        
                D_sample = demand_data_medium_sampled(:, ss); 
                s_sample = failure_data_medium_sampled(:, ss);   
                D_bar = D_sample - q_SP.*s_sample;
                input_b2b.D_bar = D_bar;
        
                output_b2b = V3DP_b2b(input_b2b);
                q_3DP = output_b2b.q_3DP;
                
                Relative_Demand_shotfall_varying_disruptions{pp,yy,i,j}(ss) = sum( max( D_bar - q_3DP, 0 ) )  / sum(max(Monthly_Quantity_3scenarios_all'))*100 ;
            
            end

        else

            %% When no 3DP is better (SINGLE SYSTEM)
            Relative_Demand_shotfall_varying_disruptions{pp,yy,i,j}  = Relative_Demand_shortfall_no3DP_varying_disruptions{pp,yy};

        end

    
    end
end

%% Initial Verification Box Plots
%  - The plots generated below are for preliminary validation.
%  - The final figures presented in our paper are produced using Python, utilizing the data saved here.

%% For each fixed p, we plot several sets of boxes, each corresponds to a yield loss, and contain two boxes: with and without 3DP
p_subset1 = [2,6,11];
yield_loss_rate_subset1 = [1,2,4,6,8];

tick_labels1 = {'0.2x', '1x', '2x', '4x', '6x', '8x', '10x', '12x', '14x'};

for ppp = 1 : length(p_subset1)

    pp = p_subset1(ppp);

    for yyy = 1 : length(yield_loss_rate_subset1)

        yy = yield_loss_rate_subset1(yyy);
        
        boxplot(Relative_Demand_shotfall_varying_disruptions{pp,yy,i,j}, 'Positions', 4*yyy-3, 'Widths', 0.5, 'Symbol', '');
        hold on
        boxplot(Relative_Demand_shortfall_no3DP_varying_disruptions{pp,yy}, 'Positions', 4*yyy-2, 'Widths', 0.5, 'Symbol', '', 'Colors', 'r', 'MedianStyle', 'target');
        hold on
        
        Mean_Relative_Demand_shotfall_varying_disruptions{i,j}(pp,yy) = mean(Relative_Demand_shotfall_varying_disruptions{pp,yy,i,j});
        Mean_Relative_Demand_shortfall_no3DP_varying_disruptions(pp,yy) = mean(Relative_Demand_shortfall_no3DP_varying_disruptions{pp,yy});
        
    end
    
    plot( 4*[1:length(yield_loss_rate_subset1)]-3, Mean_Relative_Demand_shotfall_varying_disruptions{i,j}(pp,yield_loss_rate_subset1), '-o', 'LineWidth', 0.5, 'MarkerSize', 6, ...
        'MarkerFaceColor', [0, 0.4470, 0.7410])
    hold on
    plot( 4*[1:length(yield_loss_rate_subset1)]-2, Mean_Relative_Demand_shortfall_no3DP_varying_disruptions(pp,yield_loss_rate_subset1), '-o', 'LineWidth', 0.5, 'MarkerSize', 6, ...
        'MarkerFaceColor', [0.8500, 0.3250, 0.0980]) 

    xticks(4*[1:length(yield_loss_rate_subset1)]-2.5); 
    xticklabels(tick_labels1(yield_loss_rate_subset1));

    xline(4*[1:length(yield_loss_rate_subset1)]-0.5, 'Color', [0.5, 0.5, 0.5], 'LineStyle', '-', 'LineWidth', 0.5);

    xlabel('Yield Loss Ratio (Multiple of Baseline)', 'FontSize', 12);
    ylabel('Shortfall (% of Max Demand)', 'FontSize', 12);
    ytickformat('percentage');

    ylim([-0.333,5])
    xlim([-0.5, 4*length(yield_loss_rate_subset1)]-0.5)

    filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls_Varying_p_yieldloss/Varying_p_yield_loss(Shortfalls)/boxplots_C3DPcase_', num2str(i), num2str(j), 'fixed_p_varying_yieldloss_case', num2str(pp), '.pdf');
    saveas(gcf, filename);
    close(gcf)

end

%% Save data for python
% The data for box plots and mean plots
Box_plot_data11 = {};
Box_plot_data12 = {};
mean_plot_data11 = [];
mean_plot_data12 = [];
for ppp = 1 : length(p_subset1)

    pp = p_subset1(ppp);

    for yyy = 1 : length(yield_loss_rate_subset1)

        yy = yield_loss_rate_subset1(yyy);
        Box_plot_data11{1,ppp}(yyy,:) = Relative_Demand_shotfall_varying_disruptions{pp,yy,i,j};
        Box_plot_data12{1,ppp}(yyy,:) = Relative_Demand_shortfall_no3DP_varying_disruptions{pp,yy};

    end

    mean_plot_data11(ppp,:) = Mean_Relative_Demand_shotfall_varying_disruptions{i,j}(pp,yield_loss_rate_subset1);
    mean_plot_data12(ppp,:) = Mean_Relative_Demand_shortfall_no3DP_varying_disruptions(pp,yield_loss_rate_subset1);

end
% Position paramters of the boxes
box_plot_pos11 = 4*[1:length(yield_loss_rate_subset1)] - 3;
box_plot_pos12 = 4*[1:length(yield_loss_rate_subset1)] - 2;

% x ticks labels and their positions
x_ticks_labels1 = tick_labels1(yield_loss_rate_subset1);
x_ticks_pos1 = 4*[1:length(yield_loss_rate_subset1)]-2.5;

% Position of vertical lines
vertline_pos1 = 4*[1:length(yield_loss_rate_subset1)]-0.5;

% x limit and y limit
xlimit1 = [-0.5, 4*length(yield_loss_rate_subset1)]-0.5;
ylimit1 = [-0.333,5];

% Save data to .mat file
save('Experiment_Data/Relative_Cost_Savings_Shortfalls_Varying_p_yieldloss/varying_disruption_distr_ind_for_python_shortfalls1.mat', 'Box_plot_data11', 'Box_plot_data12', 'mean_plot_data11', 'mean_plot_data12', ...
     'box_plot_pos11', 'box_plot_pos12', 'x_ticks_labels1', 'x_ticks_pos1', 'vertline_pos1', 'xlimit1', 'ylimit1');



%% For each fixed yield loss, we plot several sets of boxes, each corresponds ot a p, and contain two boxes: with and without 3DP

yield_loss_rate_subset2 = [2,5,8];
p_subset2 = [1,2,4,6,8];

tick_labels2 = {'0.2x', '1x', '2x', '3x', '4x', '5x', '6x', '7x', '8x', '9x', '10x'};

for yyy = 1 : length(yield_loss_rate_subset2)

    yy = yield_loss_rate_subset2(yyy);

    for ppp = 1 : length(p_subset2)

        pp = p_subset2(ppp);

        boxplot(Relative_Demand_shotfall_varying_disruptions{pp,yy,i,j}, 'Positions', 4*ppp-3, 'Widths', 0.5, 'Symbol', '');
        hold on
        boxplot(Relative_Demand_shortfall_no3DP_varying_disruptions{pp,yy}, 'Positions', 4*ppp-2, 'Widths', 0.5, 'Symbol', '', 'Colors', 'r', 'MedianStyle', 'target');
        hold on
        
    end
    
    plot( 4*[1:length(p_subset2)]-3, Mean_Relative_Demand_shotfall_varying_disruptions{i,j}(p_subset2,yy)', '-^', 'LineWidth', 0.5, 'MarkerSize', 6, ...
        'MarkerFaceColor', [0, 0.4470, 0.7410])
    hold on
    plot( 4*[1:length(p_subset2)]-2, Mean_Relative_Demand_shortfall_no3DP_varying_disruptions(p_subset2,yy)', '-^', 'LineWidth', 0.5, 'MarkerSize', 6, ...
        'MarkerFaceColor', [0.8500, 0.3250, 0.0980])

    xticks(4*[1:length(p_subset2)]-2.5); 
    xticklabels(tick_labels2(p_subset2));

    xline(4*[1:length(p_subset2)]-0.5, 'Color', [0.5, 0.5, 0.5], 'LineStyle', '-', 'LineWidth', 0.5);

    xlabel('Marginal Failure Rate (Multiple of Baseline)', 'FontSize', 12);
    ylabel('Shortfall (% of Max Demand)', 'FontSize', 12);
    ytickformat('percentage');

    ylim([-0.333,5])
    xlim([-0.5, 4*length(p_subset2)]-0.5)
    % grid on;

    filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls_Varying_p_yieldloss/Varying_p_yield_loss(Shortfalls)/boxplots_C3DPcase_', num2str(i), num2str(j), '_fixed_yieldloss_varying_p_case', num2str(yy), '.pdf');
    saveas(gcf, filename);
    close(gcf)

end

%% Save data for python
% The data for box plots and mean plots
Box_plot_data21 = {};
Box_plot_data22 = {};
mean_plot_data21 = [];
mean_plot_data22 = [];
for yyy = 1 : length(yield_loss_rate_subset2)

    yy = yield_loss_rate_subset2(yyy);

    for ppp = 1 : length(p_subset2)

        pp = p_subset2(ppp);
        Box_plot_data21{1,yyy}(ppp,:) = Relative_Demand_shotfall_varying_disruptions{pp,yy,i,j};
        Box_plot_data22{1,yyy}(ppp,:) = Relative_Demand_shortfall_no3DP_varying_disruptions{pp,yy};

    end

    mean_plot_data21(yyy,:) = Mean_Relative_Demand_shotfall_varying_disruptions{i,j}(p_subset2,yy)';
    mean_plot_data22(yyy,:) = Mean_Relative_Demand_shortfall_no3DP_varying_disruptions(p_subset2,yy)';

end
% Position paramters of the boxes
box_plot_pos21 = 4*[1:length(p_subset2)] - 3;
box_plot_pos22 = 4*[1:length(p_subset2)] - 2;

% x ticks labels and their positions
x_ticks_labels2 = tick_labels2(p_subset2);
x_ticks_pos2 = 4*[1:length(p_subset2)]-2.5;

% Position of vertical lines
vertline_pos2 = 4*[1:length(p_subset2)]-0.5;

% x limit and y limit
xlimit2 = [-0.5, 4*length(p_subset2)]-0.5;
ylimit2 = [-0.333,5];

save('Experiment_Data/Relative_Cost_Savings_Shortfalls_Varying_p_yieldloss/varying_disruption_distr_ind_for_python_shortfalls2.mat', ...
    'Box_plot_data21', 'Box_plot_data22', ...
    'mean_plot_data21', 'mean_plot_data22', ...
    'box_plot_pos21', 'box_plot_pos22', ...
    'x_ticks_labels2', 'x_ticks_pos2', ...
    'vertline_pos2', 'xlimit2', 'ylimit2');


