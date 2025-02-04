%% In this document, we study "backup switching"

%% Experiment 1: we want to understand the precentage of product switched
%%      - we let the number of suppliers change (num_suppliers)
%%      - for each num_suppliers, we sample a bunch of subsets of suppliers
%%      - for each sampled subset, we run BoE local search on a grid of K values
%%      - At the optimal K, we will be able to obtain the 3DP set
%% In summary, for each num_suppliers, we would be able to obtain a histogram of "optimal" 3DP set size (relative to num_suppliers)

%% Experiment 2: we want to understand what are driving the products to be switched
%%      - We sample a bunch of subset of suppliers with the same size
%%      - For each sampled subset, we run BoE local search on a grid of K values
%%      - At the optimal K, we will be able to obtain the 3DP set
%%      - To create heterogeniety of disruption distribution, for each subset, we set p and yield_loss_rate differently






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















%% Experiment 1: Understanding the percentage of products switched
%  - Vary the number of suppliers (num_suppliers)
%  - For each num_suppliers, sample multiple subsets of suppliers
%  - For each sampled subset, run BoE local search on a grid of K values
%  - At the optimal K, determine the 3DP set

Random_Suppliers_Num = 50;
capacity_3dp_percentage = [1e-2, 1e-1 * [1:9], 1:2:25, 30:5:50, 75, 100] / 100;

% Initialize output storage
OUTPUT_MEDIUM_BOE = {};
A_t_BOE_SET = {};
X_BOE_SET = {};
TOTAL_COST_SET = {};
X_BOE_OPT_SET = {};
SWITCHED_BOE_SET = {};
SWITCHED_BOE_OPT_SET = {};

SUPPLIER_SUBSET_IDX = {};
TM_BACKUPSET = {};
NUM_SUPPLIERS_SET = [15, 30, 45];

for num_suppliers_case = 1:length(NUM_SUPPLIERS_SET)
    nn = NUM_SUPPLIERS_SET(num_suppliers_case);

    for random_suppliers_num = 1:Random_Suppliers_Num
        fprintf("---------------------------------------------------------------------\n")
        fprintf("Processing case: %d suppliers, iteration #%d \n", nn, random_suppliers_num)
        fprintf("---------------------------------------------------------------------\n")

        % Randomly sample a subset of suppliers
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

        %% Compute costs under different backup strategies
        % No backup scenario
        input_medium_no3dp = struct('n', num_suppliers, 'v', v, 'h', h, 'p', p_medium, ...
            'yield_loss_rate', yield_loss_rate_medium, 'Demand_atoms', Monthly_Quantity_3scenarios, ...
            'Demand_prob', Demand_Probability_3scenarios, 'Demand_mean', mean_demand_3scenarios, 'TM_flag', 0);
        output_medium_no3dp = Cost_No3DP_or_TM(input_medium_no3dp);

        % Backup with infinite 3DP capacity
        input_medium_3DP_infK = input_medium_no3dp;
        input_medium_3DP_infK.v = c_3DP;
        output_medium_3DP_infK = Cost_No3DP_or_TM(input_medium_3DP_infK);

        % Backup with TM
        TM_retainer_ratio = 0.75;
        C_TM = TM_retainer_ratio * c_source .* mean_demand_3scenarios;
        input_medium_TM = input_medium_no3dp;
        input_medium_TM.TM_flag = 1;
        input_medium_TM.c_TM = c_TM;
        input_medium_TM.C_TM = C_TM;
        output_medium_TM = Cost_No3DP_or_TM(input_medium_TM);

        % Identify products originally backed up by TM
        TM_backup_set = output_medium_TM.TM_cost < output_medium_no3dp.opt_val;
        TM_BACKUPSET{num_suppliers_case, random_suppliers_num} = TM_backup_set;

        %% Determine sample size for BoE Submodular Maximization (SAA)
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

        %% Compute key parameters for BoE
        Obj_const = -output_medium_no3dp.opt_val;
        U0_with_vmean = output_medium_no3dp.opt_val + v .* mean_demand_3scenarios;
        U0_no_vmean = output_medium_no3dp.opt_val;
        TM_Delta = output_medium_TM.TM_cost - output_medium_no3dp.opt_val;
        ratio_over_weight = (v - c_3DP) ./ weight;
        q0 = output_medium_no3dp.opt_q;

        % Compute probability of unfilled demand
        pi_p = zeros(num_suppliers, 1);
        pi_0 = zeros(num_suppliers, 1);
        for j = 1:num_suppliers
            tmp1 = max(0, Monthly_Quantity_3scenarios(j, :)' - q0(j) * [1 - yield_loss_rate_medium, 1]);
            tmp2 = Demand_Probability_3scenarios(j, :)' * [p_medium, 1 - p_medium];

            pi_p(j, :) = sum(tmp2(tmp1 > 1e-5));
            pi_0(j, :) = sum(tmp2(tmp1 <= 1e-5));
        end

        %% Run BoE on a grid of K values
        for k = 1:length(capacity_3dp_percentage)
            fprintf("%3.2f%% of Max Yield Shortfall\n", capacity_3dp_percentage(k) * 100)

            % Compute 3DP capacity and cost
            capacity_percentage = capacity_3dp_percentage(k);
            K_3DP_medium = capacity_percentage * sum(max(Monthly_Weight_3scenarios'));
            C_3DP_medium = (K_3DP_medium ./ speed_per_machine_month)' * cost_of_3dp_per_machine_month;

            % Run BoE
            input_boe.K_3DP = K_3DP_medium;
            input_boe.C_3DP = C_3DP_medium;
            output_boe = BoE_Approx_Max_Submod_SAA(input_boe);
            OUTPUT_MEDIUM_BOE{num_suppliers_case, random_suppliers_num, k} = output_boe;
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

        A_t_BOE_SET{num_suppliers_case, random_suppliers_num} = A_t_BOE;
        X_BOE_SET{num_suppliers_case, random_suppliers_num} = X_BOE;
        SWITCHED_BOE_SET{num_suppliers_case, random_suppliers_num} = SWITCHED_BOE;
        TOTAL_COST_SET{num_suppliers_case, random_suppliers_num} = TOTAL_COST;

        %% Extract optimal 3DP backup set
        X_BOE_OPT = {};
        SWITCHED_BOE_OPT = {};
        for i = 1:length(speed_per_machine_month)
            for j = 1:length(cost_of_3dp_per_machine_month)
                [~, iii] = min(TOTAL_COST{i, j});
                X_BOE_OPT{i, j} = X_BOE{i, j}(:, iii);
                SWITCHED_BOE_OPT{i, j} = SWITCHED_BOE{i, j}(:, iii);
            end
        end

        X_BOE_OPT_SET{num_suppliers_case, random_suppliers_num} = X_BOE_OPT;
        SWITCHED_BOE_OPT_SET{num_suppliers_case, random_suppliers_num} = SWITCHED_BOE_OPT;
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


directory = "Experiment_Data/Switch_Backup_vs_n";
if ~exist(directory, 'dir')
    mkdir(directory);
end
save(fullfile(directory, "data_n_less_than_55.mat"));




%% BOX PLOTS
%  - Generate sample plots using MATLAB.
%  - The final plots used in the paper are created in Python from the CSV data saved here.
NUM_SUPPLIERS_SET = [15,30,45];

%% First plot the percentage of "Unprotected Products" vs. "num_suppliers" 
figure;
hold on;

for num_suppliers_case = 2 : length(NUM_SUPPLIERS_SET)           
    boxplot(Ratio_bare{num_suppliers_case}' , 'Positions', NUM_SUPPLIERS_SET(num_suppliers_case), 'Widths', 2)          
end
xlim([min(NUM_SUPPLIERS_SET(2:end)) - 5, max(NUM_SUPPLIERS_SET) + 5]); % Adjust x-axis limits
xticks(NUM_SUPPLIERS_SET(2:end)); % Set x-axis ticks at specified positions
xticklabels(string(NUM_SUPPLIERS_SET(2:end))); % Display the actual positions as tick labels
ylabel('Percentage of Bare Products');
% title('Boxplots for Each Row');
grid on;

fileName = strcat('Experiment_Data/Switch_Backup_vs_n/Num_bare_boxplots.pdf'); % Specify the file name
saveas(gcf, fileName); % Save current figure as a PDF

hold off;

close(gcf);

%% Save the data for python
% Open a file to save the data
file_name = 'Experiment_Data/Switch_Backup_vs_n/num_bare_boxplot_data.csv';
fid = fopen(file_name, 'w');

% Write the CSV header
fprintf(fid, 'Supplier_Count,Ratio_Bare\n');

% Loop through each supplier case and write the data
for num_suppliers_case = 2 : length(NUM_SUPPLIERS_SET)
    supplier_count = NUM_SUPPLIERS_SET(num_suppliers_case);
    ratio_bare_values = Ratio_bare{num_suppliers_case}; % Data for this supplier case
    
    % Write each data point as a new row
    for k = 1:length(ratio_bare_values)
        fprintf(fid, '%d,%.4f\n', supplier_count, ratio_bare_values(k));
    end
end

% Close the file
fclose(fid);

fprintf('Data saved to %s\n', file_name);


%% Now plot the followings:
%%      - the percentage of "3DP Products" (over total number of products) 
%%      - the percentage of "Products Switcehd to 3DP" (over total number of TM producst in the SINGLE system)
%% NOTE WE PLOT FOR EACH "c_cap" 


for i = 1 : length(speed_per_machine_month)  
    for j = 1 : length(cost_of_3dp_per_machine_month)
    
        %% Total 3DP selection
        figure;
        hold on;

        for num_suppliers_case = 2 : length(NUM_SUPPLIERS_SET)           
            boxplot(Ratio_3DP_backup{num_suppliers_case, i, j}' , 'Positions', NUM_SUPPLIERS_SET(num_suppliers_case), 'Widths', 2)          
        end
        xlim([min(NUM_SUPPLIERS_SET(2:end)) - 5, max(NUM_SUPPLIERS_SET) + 5]); % Adjust x-axis limits
        xticks(NUM_SUPPLIERS_SET(2:end)); % Set x-axis ticks at specified positions
        xticklabels(string(NUM_SUPPLIERS_SET(2:end))); % Display the actual positions as tick labels
        ylabel('Percentage of 3DP Backedup');
        % title('Boxplots for Each Row');
        grid on;

        fileName = strcat('Experiment_Data/Switch_Backup_vs_n/Num_3DP_backup_3DPcase_', num2str(i), num2str(j),'_boxplots.pdf'); % Specify the file name
        saveas(gcf, fileName); % Save current figure as a PDF

        hold off;

        close(gcf);

        %% Switched
        figure;
        hold on;

        for num_suppliers_case = 2 : length(NUM_SUPPLIERS_SET)           
            boxplot(Ratio_switched_backup{num_suppliers_case, i, j}' , 'Positions', NUM_SUPPLIERS_SET(num_suppliers_case), 'Widths', 2)          
        end
        xlim([min(NUM_SUPPLIERS_SET(2:end)) - 5, max(NUM_SUPPLIERS_SET) + 5]); % Adjust x-axis limits
        xticks(NUM_SUPPLIERS_SET(2:end)); % Set x-axis ticks at specified positions
        xticklabels(string(NUM_SUPPLIERS_SET(2:end))); % Display the actual positions as tick labels
        ylabel('Percentage of Backedup Switched');
        % title('Boxplots for Each Row');
        grid on;

        fileName = strcat('Experiment_Data/Switch_Backup_vs_n/Num_switched_3DPcase_', num2str(i), num2str(j),'_boxplots.pdf'); % Specify the file name
        saveas(gcf, fileName); % Save current figure as a PDF

        hold off;

        close(gcf);


    end
end


%% For case i = 1, j = 9
%% SAVE THE CSV DATA FOR "total 3DP backup"
% Open a file to save the data
file_name = 'Experiment_Data/Switch_Backup_vs_n/num_3DP_backup_boxplot_data_case_1_9.csv';
fid = fopen(file_name, 'w');

% Write the CSV header
fprintf(fid, 'Supplier_Count,3DP_Backup_Percentage\n');

% Extract and write data for i = 1, j = 9
for num_suppliers_case = 2 : length(NUM_SUPPLIERS_SET)
    supplier_count = NUM_SUPPLIERS_SET(num_suppliers_case);
    backup_percentages = Ratio_3DP_backup{num_suppliers_case, 1, 9}'; % Data for the case
    
    % Write each data point
    for k = 1:length(backup_percentages)
        fprintf(fid, '%d,%.4f\n', supplier_count, backup_percentages(k));
    end
end

% Close the file
fclose(fid);

fprintf('Data saved to %s\n', file_name);

%% SAVE THE CSV DATA FOR "3DP backup switched"
% Open a file to save the data
file_name = 'Experiment_Data/Switch_Backup_vs_n/num_switched_backup_plot_data_case_1_9.csv';
fid = fopen(file_name, 'w');

% Write the CSV header
fprintf(fid, 'Supplier_Count,Switched_Backup_Percentage\n');

% Extract and write data for i = 1, j = 9
for num_suppliers_case = 2 : length(NUM_SUPPLIERS_SET)
    supplier_count = NUM_SUPPLIERS_SET(num_suppliers_case);
    switched_backup_percentages = Ratio_switched_backup{num_suppliers_case, 1, 9}'; % Data for this case
    
    % Write each data point
    for k = 1:length(switched_backup_percentages)
        fprintf(fid, '%d,%.4f\n', supplier_count, switched_backup_percentages(k));
    end
end

% Close the file
fclose(fid);

fprintf('Data saved to %s\n', file_name);
















%% FINALLY LET'S RUN THE CASE WHEN n = 55
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
COST_TMONLY_BENCMARK = sum(output_medium_TM.TM_cost(TM_backup_set))+sum(output_medium_no3dp.opt_val(logical(nobackup_set)));

        
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

        

OUTPUT_MEDIUM_BOE_BENCHMARK = {};

capacity_3dp_percentage_benchmark = [1e-2, 1e-1*[1:9], [1:0.2:2.8], 3:2,25, 30:5:50, 75, 100]/100;

fileID1 = fopen('Log7.txt', 'a'); % 'a' mode appends to the file

for k = 1 : length(capacity_3dp_percentage_benchmark)

    fprintf("%3.2f Percent of Max Yield Shortfall \n\n", capacity_3dp_percentage_benchmark(k)*100)

    capacity_percentage = capacity_3dp_percentage_benchmark(k);
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
        input_boe.A_init = OUTPUT_MEDIUM_BOE_BENCHMARK{k-1}.A_t;
    end

    if num_suppliers <= 10
        output_boe = BoE_Approx_Max_Submod_Exact(input_boe);
    else
        output_boe = BoE_Approx_Max_Submod_SAA(input_boe);
    end
    

    % input_boe.A_init = [1:num_suppliers];
    % output_boe2 = BoE_Approx_Max_Submod1(input_boe);
    % 
    % if output_boe1.TOTAL_COST(1,1) <  output_boe2.TOTAL_COST(1,1) 
    %     output_boe = output_boe1;
    % else
    %     output_boe = output_boe2;
    % end

    OUTPUT_MEDIUM_BOE_BENCHMARK{k} = output_boe; 
    % TIME_BOE(k) = output_boe.solving_time;


    fprintf(fileID1, 'Benchmark Case,     k=%3.2f %% \n',  capacity_3dp_percentage_benchmark(k));
    fprintf(fileID1, '3DP Set: %d ', OUTPUT_MEDIUM_BOE_BENCHMARK{k}.A_t);
    fprintf(fileID1, '\n\n')



    disp("TO NEXT ONE! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ")

end




%% FOR THE BENCMARK CASE, WE ALSO SUMMARIZE THE FOLLOWINGS
%% A_t_BOE:      the 3DP set we obtain before post-processing (comparing to having no 3DP selection)
%% X_BOE:        the 3DP set we obatin after post-processing (under different C3DP coefficients)
%% TOTAL_COST:   the total system cost after post-processing 
%% NOTE: TOTAL_COST is computed by (K is fixed):
%%      - Given "A_t_BOE" obtained from local search method, fix the 3DP set as "A_t_BOE"
%%      - Run SGD on this fixed 3DP set problem
%%      - Given solution of SGD, evaluate the U3DP with a larger sample size
%%      - Compare to the case when no 3DP is selected
%% Threfore, under different K, the "TOTAL_COST" could be computed on different samples
X_BOE_BENCHMARK = {}; 
SWITCHED_BOE_BENCHMARK = {};
for i = 1 : length(speed_per_machine_month)  
    for j = 1 : length(cost_of_3dp_per_machine_month)
        X_BOE_BENCHMARK{i,j} = zeros(num_suppliers,length(capacity_3dp_percentage_benchmark));
        SWITCHED_BOE_BENCHMARK{i,j} = zeros(num_suppliers,length(capacity_3dp_percentage_benchmark));
    end
end
TOTAL_COST_BENCHMARK = {};
A_t_BOE_BENCHMARK = [];
for k = 1 : length(capacity_3dp_percentage_benchmark)

    A_t_BOE_BENCHMARK(OUTPUT_MEDIUM_BOE_BENCHMARK{k}.A_t, k) = 1;

    for i = 1 : length(speed_per_machine_month)  
        for j = 1 : length(cost_of_3dp_per_machine_month)

            TOTAL_COST_BENCHMARK{i,j}(k) = OUTPUT_MEDIUM_BOE_BENCHMARK{k}.TOTAL_COST(i,j);
            X_BOE_BENCHMARK{i,j}(:, k) = OUTPUT_MEDIUM_BOE_BENCHMARK{k}.X_FINAL{i,j};
            if ~isinf(sum(X_BOE_BENCHMARK{i,j}(:,k)))
                SWITCHED_BOE_BENCHMARK{i,j}(:,k) = TM_backup_set - (1-X_BOE_BENCHMARK{i,j}(:,k));
            end

        end
    end

end

%% "X_BOE_OPT" is the 3DP set at the optimal K
%% "SWITCHED_SET_OPT" is the set of products whos backup swithced after intro of 3DP
X_BOE_OPT_BENCHMARK = {}; 
SWITCHED_BOE_OPT_BENCHMARK = {};
for i = 1 : length(speed_per_machine_month)  
    for j = 1 : length(cost_of_3dp_per_machine_month)

        [~, iii] = min(TOTAL_COST_BENCHMARK{i,j});
        X_BOE_OPT_BENCHMARK{i,j} = X_BOE_BENCHMARK{i,j}(:,iii);
        SWITCHED_BOE_OPT_BENCHMARK{i,j} = SWITCHED_BOE_BENCHMARK{i,j}(:,iii);

    end
end



%% Now for n=55 case, we plot the effective 3DP selection and 3DP switches vs. K

for i = 1 : length(speed_per_machine_month)  
    for j = 1 : length(cost_of_3dp_per_machine_month)
    
        
        figure;
        hold on;
        
        % Compute tmp1 and tmp2
        tmp1 = sum(X_BOE_BENCHMARK{i,j}) / 55 * 100;
        notinf_idx1 = ~isinf(tmp1);  % 0-1 indicator where data is valid
        tmp2 = sum(SWITCHED_BOE_BENCHMARK{i,j}) / sum(TM_backup_set) * 100;
        
        % Find the first occurrence where 0 turns to 1 when counting from the end
        reverse_idx = flip(notinf_idx1);  % Flip the logical array
        jth_idx = length(notinf_idx1) - find(reverse_idx, 1, 'first') + 1;  % Find the original index
        
        % Find the corresponding capacity value for the jth element
        capacity_jth = capacity_3dp_percentage_benchmark(jth_idx) * 100;
        
        % Plot Total 3DP selection
        plot([0, capacity_3dp_percentage_benchmark(notinf_idx1)] * 100, [0, tmp1(notinf_idx1)], '-', 'LineWidth', 2);
        
        % Plot Total Switched
        plot([0, capacity_3dp_percentage_benchmark(notinf_idx1)] * 100, [0, tmp2(notinf_idx1)], '-', 'LineWidth', 2);
        
        % Add the vertical grey line at the jth capacity point
        xline(capacity_jth, '--', 'LineWidth', 2, 'Color', [0.5, 0.5, 0.5]);
        
        % Fill the area to the right of the vertical line with light grey
        x_fill = [capacity_jth, 100, 100, capacity_jth];  % X-coordinates for the shaded area
        y_fill = [0, 0, 2*max(max(tmp1(notinf_idx1)), max(tmp2(notinf_idx1))), 2*max(max(tmp1(notinf_idx1)), max(tmp2(notinf_idx1)))];
        fill(x_fill, y_fill, [0.8, 0.8, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
        
        % Add red text to the right of the vertical line
        text_x = capacity_jth + 2;  % Slightly offset to the right of the vertical line
        text_y = 50; % Position the text at 80% of the Y-limit
        text(text_x, text_y, 'Better-off with no 3DP', 'Color', 'red', 'FontSize', 15, 'FontWeight', 'bold');
        
        % Final plot formatting
        xlabel('3DP Capacity (% of Max)', 'FontSize', 15);
        ylabel('Percentage', 'FontSize', 15);
        xlim([0, 75]);
        ylim([0, 100]);  % Set Y-limits slightly above the data
        grid on;
        
        % Add legend
        legend('Total 3DP Selection', 'Total Switched', 'Location', 'SouthEast', 'FontSize', 15);
       
      
        fileName = strcat('Experiment_Data/Switch_Backup_vs_n/All_Suppliers_3DPcase_', num2str(i), num2str(j),'.pdf'); % Specify the file name
        saveas(gcf, fileName); % Save current figure as a PDF

        hold off;

        close(gcf);

    end
end




%% Take i = 1, j = 10 as a typical example 

i = 1; j = 10;

figure;
hold on;

% Compute tmp1 and tmp2
tmp1 = sum(X_BOE_BENCHMARK{i,j}) / 55 * 100;
notinf_idx1 = ~isinf(tmp1);  % 0-1 indicator where data is valid
tmp2 = sum(SWITCHED_BOE_BENCHMARK{i,j}) / sum(TM_backup_set) * 100;

% Find the first occurrence where 0 turns to 1 when counting from the end
reverse_idx = flip(notinf_idx1);  % Flip the logical array
jth_idx = length(notinf_idx1) - find(reverse_idx, 1, 'first') + 1;  % Find the original index

% Find the corresponding capacity value for the jth element
capacity_jth = capacity_3dp_percentage_benchmark(jth_idx) * 100;

% Plot Total 3DP selection
plot(capacity_3dp_percentage_benchmark(notinf_idx1) * 100, tmp1(notinf_idx1), '-', 'LineWidth', 2);

% Plot Total Switched
plot(capacity_3dp_percentage_benchmark(notinf_idx1) * 100, tmp2(notinf_idx1), '-', 'LineWidth', 2);

% Add the vertical grey line at the jth capacity point
xline(capacity_jth, '--', 'LineWidth', 2, 'Color', [0.5, 0.5, 0.5]);

% Fill the area to the right of the vertical line with light grey
x_fill = [capacity_jth, 100, 100, capacity_jth];  % X-coordinates for the shaded area
y_fill = [0, 0, 2*max(max(tmp1(notinf_idx1)), max(tmp2(notinf_idx1))), 2*max(max(tmp1(notinf_idx1)), max(tmp2(notinf_idx1)))];
fill(x_fill, y_fill, [0.8, 0.8, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);

% Add red text to the right of the vertical line
text_x = capacity_jth + 0.1;  % Slightly offset to the right of the vertical line
text_y = 50; % Position the text at 80% of the Y-limit
text(text_x, text_y, 'Better-off with no 3DP', 'Color', 'red', 'FontSize', 15, 'FontWeight', 'bold');

% Final plot formatting
xlabel('3DP Capacity (% of Max)', 'FontSize', 15);
xlim([0, 6]);
ylim([0, 100]);  % Set Y-limits slightly above the data
grid on;

% Add legend
legend('Percentage of 3DP Backup', 'Percentage of Backup Switched', 'Location', 'SouthEast', 'FontSize', 15);


fileName = strcat('Experiment_Data/Switch_Backup_vs_n/All_Suppliers_3DPcase_', num2str(i), num2str(j),'(PRETTIER).pdf'); % Specify the file name
saveas(gcf, fileName); % Save current figure as a PDF

hold off;

close(gcf);




%% Save data for Python
file_name = 'Experiment_Data/Switch_Backup_vs_n/all_suppliers_case_1_10.csv';
fid = fopen(file_name, 'w');

% Write header
fprintf(fid, 'Capacity_3DP_Percentage,Total_3DP_Selection,Total_Switched\n');

% Write data where notinf_idx1 is true
for k = 1:length(capacity_3dp_percentage_benchmark)
    if notinf_idx1(k)  % Only include valid data
        fprintf(fid, '%.4f,%.4f,%.4f\n', ...
            capacity_3dp_percentage_benchmark(k) * 100, tmp1(k), tmp2(k));
    end
end

% Close the file
fclose(fid);

fprintf('Data saved to %s\n', file_name);


































%% Experiment 2: we want to understand what are driving the products to be switched
%%      - We sample a bunch of subset of suppliers with the same size
%%      - For each sampled subset, we run BoE local search on a grid of K values
%%      - At the optimal K, we will be able to obtain the 3DP set
%%      - To create heterogeniety of disruption distribution, for each subset, we set p and yield_loss_rate differently


Random_Suppliers_Num = 500;

% capacity_3dp_percentage = [0.01, 0.1,0.5,1:2:25,30:5:50, 60:10:100]/100;
capacity_3dp_percentage = [1e-2, 1e-1*[1:9], 1:2:25, 30:5:50, 75, 100]/100;
% capacity_3dp_percentage = [1e-2, 1e-1*[1:9],1:30,40,50,75,100]/100;

OUTPUT_MEDIUM_BOE = {};
A_t_BOE_SET = {};
X_BOE_SET = {};
TOTAL_COST_SET = {};
X_BOE_OPT_SET = {};


fileID1 = fopen('Log1.txt', 'a'); % 'a' mode appends to the file

nn = 10;

P_MEDIUM_SET = [];
YIELD_LOSS_RATE_MEDIUM_SET = [];
SUPPLIER_SUBSET_IDX = {};
TM_BACKUPSET = {};
C_TM_SET = {};
TM_RETAINER_RATIO_SET = {}; 


for random_suppliers_num = 1 : Random_Suppliers_Num
    
    startTime = clock;

    display("---------------------------------------------------------------------\n")
    fprintf("WE ARE CURRENTLY WORKING ON: %d suppliers, case #%d \n", nn, random_suppliers_num)
    display("---------------------------------------------------------------------\n")
    
    p_medium = unifrnd(0, 0.5);
    yield_loss_rate_medium = unifrnd(0, 1);

    P_MEDIUM_SET(random_suppliers_num) = p_medium;
    YIELD_LOSS_RATE_MEDIUM_SET(random_suppliers_num) = yield_loss_rate_medium;
    
    %% In what follows, we:
    %%     - Radomly sample a subset of products and treat it as all the products we have here
    %%     - Compare: Full Info MIP (GRB) vs. SAA MIP (GRB) vs. Benders
    %%     - All these are implemented with "fixed K", but under different K values
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
        
    
        % input_boe.A_init = [1:num_suppliers];
        % output_boe2 = BoE_Approx_Max_Submod1(input_boe);
        % 
        % if output_boe1.TOTAL_COST(1,1) <  output_boe2.TOTAL_COST(1,1) 
        %     output_boe = output_boe1;
        % else
        %     output_boe = output_boe2;
        % end
    
        OUTPUT_MEDIUM_BOE{random_suppliers_num,k} = output_boe; 
        % TIME_BOE(k) = output_boe.solving_time;
    
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

    fprintf(fileID1, 'n = %d,  case: %d,  3DP-size = %d, Switched = %d (case 1,1), p=%3.2f, yield_loss_rate=%3.2f \n', ...
                nn, random_suppliers_num, sum(X_BOE_OPT{1,1}), sum(SWITCHED_BOE_OPT{1,1}) , p_medium, yield_loss_rate_medium);

    fprintf(fileID1, 'n = %d,  case: %d , p=%3.2f, yield_loss_rate=%3.2f \n',  nn, random_suppliers_num, p_medium, yield_loss_rate_medium);
    fprintf(fileID1, '(1,1): %d/%d; (1,2): %d/%d; (1,3): %d/%d; (1,4): %d/%d; (1,5): %d/%d; \n', ...
        sum(SWITCHED_BOE_OPT{1,1}), sum(TM_backup_set),...
        sum(SWITCHED_BOE_OPT{1,2}), sum(TM_backup_set),...
        sum(SWITCHED_BOE_OPT{1,3}), sum(TM_backup_set),...
        sum(SWITCHED_BOE_OPT{1,4}), sum(TM_backup_set),...
        sum(SWITCHED_BOE_OPT{1,5}), sum(TM_backup_set));
    fprintf(fileID1, '(1,6): %d/%d; (1,7): %d/%d; (1,8): %d/%d; (1,9): %d/%d; (1,10): %d/%d;  \n\n', ...
        sum(SWITCHED_BOE_OPT{1,6}), sum(TM_backup_set),...
        sum(SWITCHED_BOE_OPT{1,7}), sum(TM_backup_set),...
        sum(SWITCHED_BOE_OPT{1,8}), sum(TM_backup_set),...
        sum(SWITCHED_BOE_OPT{1,9}), sum(TM_backup_set),...
        sum(SWITCHED_BOE_OPT{1,10}), sum(TM_backup_set));
    
    endTime = clock;
    Tot_time = Tot_time + etime(endTime, startTime);

    fprintf("TIME: %3.2f\n\n", Tot_time)

end


% 
%  Num_3DP_backup = {};
% for num_suppliers_case = 1 : length(NUM_SUPPLIERS_SET)
% 
%     for random_suppliers_num = 1 : Random_Suppliers_Num
% 
%         for i = 1 : length(speed_per_machine_month)  
%             for j = 1 : length(cost_of_3dp_per_machine_month)
% 
%                 Num_3DP_backup{i, j}(random_suppliers_num) = sum(X_BOE_OPT_SET{random_suppliers_num}{i,j});
% 
%             end
%         end
% 
% 
%     end
% end





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


NOBACKUP_SET = [];
Q_NOBACKUP_SET = [];

for random_suppliers_num = 1:800

    fprintf("%d; \n", random_suppliers_num)

    p_medium = P_MEDIUM_SET(random_suppliers_num);
    yield_loss_rate_medium = YIELD_LOSS_RATE_MEDIUM_SET(random_suppliers_num);
    
    %% In what follows, we:
    %%     - Radomly sample a subset of products and treat it as all the products we have here
    %%     - Compare: Full Info MIP (GRB) vs. SAA MIP (GRB) vs. Benders
    %%     - All these are implemented with "fixed K", but under different K values
    supplier_subset_idx = SUPPLIER_SUBSET_IDX{random_suppliers_num};
    
    Monthly_Weight_3scenarios = Monthly_Weight_3scenarios_all(supplier_subset_idx, :);
    Monthly_Quantity_3scenarios = Monthly_Quantity_3scenarios_all(supplier_subset_idx, :);
    Demand_Probability_3scenarios = Demand_Probability_3scenarios_all(supplier_subset_idx, :);
    mean_demand_3scenarios = mean_demand_3scenarios_all(supplier_subset_idx);
    
    c_source = c_source_all(supplier_subset_idx);
    
    c_3DP = c_3DP_all(supplier_subset_idx); 
    % c_TM = c_TM_all(supplier_subset_idx);   
    c_TM = C_TM_SET{1,random_suppliers_num};
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
    TM_retainer_ratio = TM_RETAINER_RATIO_SET{1,random_suppliers_num};
    C_TM = TM_retainer_ratio.*c_source.*mean_demand_3scenarios;
    input_medium_TM = input_medium_no3dp;
    input_medium_TM.TM_flag = 1;
    input_medium_TM.c_TM = c_TM; 
    input_medium_TM.C_TM = C_TM; 
    output_medium_TM = Cost_No3DP_or_TM(input_medium_TM);

    NOBACKUP_SET(:,random_suppliers_num) = (output_medium_no3dp.opt_val < output_medium_TM.TM_cost);
    Q_NOBACKUP_SET(:,random_suppliers_num) = output_medium_no3dp.opt_q;
end


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
            
            p_medium = P_MEDIUM_SET(random_suppliers_num);
            yield_loss_rate_medium = YIELD_LOSS_RATE_MEDIUM_SET(random_suppliers_num);
            
            %% In what follows, we:
            %%     - Radomly sample a subset of products and treat it as all the products we have here
            %%     - Compare: Full Info MIP (GRB) vs. SAA MIP (GRB) vs. Benders
            %%     - All these are implemented with "fixed K", but under different K values
            supplier_subset_idx = SUPPLIER_SUBSET_IDX{random_suppliers_num};
            
            Monthly_Weight_3scenarios = Monthly_Weight_3scenarios_all(supplier_subset_idx, :);
            Monthly_Quantity_3scenarios = Monthly_Quantity_3scenarios_all(supplier_subset_idx, :);
            Demand_Probability_3scenarios = Demand_Probability_3scenarios_all(supplier_subset_idx, :);
            mean_demand_3scenarios = mean_demand_3scenarios_all(supplier_subset_idx);
            
            c_source = c_source_all(supplier_subset_idx);
            
            c_3DP = c_3DP_all(supplier_subset_idx); 
            % c_TM = c_TM_all(supplier_subset_idx); 
            c_TM = C_TM_SET{1,random_suppliers_num};
            v = v_all(supplier_subset_idx);         
            h = h_all(supplier_subset_idx);  
            
            weight = weight_all(supplier_subset_idx);
            
           

       
            %% Features in the following orders:
            %% Vanilla features:
            %%      - Cost-parameters: v, h, cTM, c3DP, weight
            %%      - Demand: mean
            %%      - Disruption: failure probability, yield loss rate (only two atoms and one of them is 1)
        
            % raw_feature_block = [v, h, c_TM, c_3DP, weight, mean_demand_3scenarios, p_medium*ones(nn,1), yield_loss_rate_medium*ones(nn,1)];
            % raw_feature_block = [(v-c_3DP)./weight, h, c_TM, mean_demand_3scenarios, p_medium*ones(nn,1), yield_loss_rate_medium*ones(nn,1)];
            raw_feature_block = [v, h, c_3DP, c_TM, weight, (v-c_3DP)./weight, mean_demand_3scenarios, p_medium*ones(nn,1), yield_loss_rate_medium*ones(nn,1)];
            % raw_feature_block = [v, c_TM, c_3DP, weight, mean_demand_3scenarios, p_medium*ones(nn,1), yield_loss_rate_medium*ones(nn,1)];
            
            response_block = SWITCHED_BOE_OPT_SET{random_suppliers_num}{i,j};
        
            RAW_FEATURES = [RAW_FEATURES; raw_feature_block];
            RESPONSES = [RESPONSES; response_block];
        
            %% Synthesize features:
            %%      - Absolute and Relative Expensiveness of 3DP : (v-c3DP/weight) and c_cap/(v-c3DP/weight) 
            %%      - Service Level (profitability) : (v-c)/(v+h)
            %%      - (Relative to mean deamnd) Max or Mean Demand shortfall to deal with: [ D - q*s ]^+, where q stands for the opt. primary order when K is optimal
            
            % [~, iii] = min(TOTAL_COST_SET{random_suppliers_num}{i,j});
            % mean_shortfall = [];
            % for l =  1 : size(Monthly_Weight_3scenarios,1)
            %     mean_shortfall(l,:) = sum(sum( ([p_medium; 1-p_medium] * Demand_Probability_3scenarios(l,:))  .* max(0, Monthly_Quantity_3scenarios(l,:) - OUTPUT_MEDIUM_BOE{random_suppliers_num,iii}.Q_FINAL{i,j}(l)*[1-yield_loss_rate_medium; 1]) ));
            %     mean_shortfall(l,:) = mean_shortfall(l,:)/mean_demand_3scenarios(l,:);
            % end

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
% synth_featureNames = {'Abs 3DP profit', 'Relative 3DP profit', 'service level', 'mean shortfall'};
synth_featureNames = {'Abs 3DP profit', 'Relative 3DP profit', 'service level1', 'service level2', 'service level3', 'mean shortfall', 'DB Retainer Rate'};

base_dir = 'Experiment_Data/Decision_Tree/';

for i = 1:length(speed_per_machine_month)
    for j = 1:length(cost_of_3dp_per_machine_month)
        % Get the current datasets
        raw_features = RAW_FEATURES_SET{i,j}(2001:8000,:);
        synth_features = SYNTH_FEATURES_SET{i,j}(2001:8000,:);
        responses = RESPONSES_SET{i,j}(2001:8000,:);
        
        % Create descriptive filename based on parameters
        filename_base = sprintf('scenario_speed_%d_cost_%d', i, j);
        
        % Save raw features
        raw_table = array2table(raw_features, 'VariableNames', raw_featureNames);
        writetable(raw_table, fullfile(base_dir, ...
            [filename_base '_raw_features3.csv']));
        
        % Save synthetic features
        synth_table = array2table(synth_features, 'VariableNames', synth_featureNames);
        writetable(synth_table, fullfile(base_dir, ...
            [filename_base '_synth_feature3.csv']));
        
        % Save responses
        resp_table = array2table(responses, 'VariableNames', {'response'});
        writetable(resp_table, fullfile(base_dir, ...
            [filename_base '_responses3.csv']));
        
        % Save a metadata file with scenario parameters
        metadata = struct();
        metadata.speed = speed_per_machine_month(i);
        metadata.cost = cost_of_3dp_per_machine_month(j);
        metadata.raw_feature_names = raw_featureNames;
        metadata.synth_feature_names = synth_featureNames;
        save(fullfile(base_dir, ...
            [filename_base '_metadata3.mat']), 'metadata');
        
        % Also save metadata in JSON format for easier reading in Python/R
        fid = fopen(fullfile(base_dir, ...
            [filename_base '_metadata3.json']), 'w');
        fprintf(fid, jsonencode(metadata));
        fclose(fid);
    end
end




save("Experiment_Data/Decision_Tree/data2.mat")




DDD=load("Experiment_Data/Decision_Tree/data.mat");
















































%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% JUNK UNDERNEATH!!!
%% JUNK UNDERNEATH!!!
%% JUNK UNDERNEATH!!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

raw_featureNames = {'v', 'h', 'c3DP', 'cTM', 'weight', '3DP profit', 'demand mean', 'disrupt. prob.', 'yield loss'};
synth_featureNames = {'Abs 3DP profit', 'Relative 3DP profit', 'service level', 'mean shortfall'};

ttt = [2,4,6:9];
% ttt = [1:9];

input_DT.FEATURES = RAW_FEATURES_SET{1,6}(:,ttt);
input_DT.RESPONSES = RESPONSES_SET{1,6};
input_DT.FEATURE_NAMES = raw_featureNames(ttt);

fit_DT(input_DT)




ttt = [2,3,4];
input_DT.FEATURES = SYNTH_FEATURES_SET{1,5}(:,ttt);
input_DT.RESPONSES = RESPONSES_SET{1,5};
input_DT.FEATURE_NAMES = synth_featureNames(ttt);

fit_DT(input_DT)






RAW_FEATURES_SET{i,j}
SYNTH_FEATURES_SET{i,j}
RESPONSES_SET{i,j}











































%% Work on raw features
% raw_featureNames = {'v', 'h', 'cTM', 'c3DP', 'weight', '3DP profit','demand mean', 'disrupt. prob.', 'yield loss', 'c_cap'};
% raw_dataTable = array2table(RAW_FEATURES);
raw_featureNames = {'3DP profit','h','c_TM','demand mean', 'disrupt. prob.', 'yield loss', 'c_cap'};
raw_dataTable = array2table(RAW_FEATURES);
% raw_featureNames = {'v', 'cTM', 'c3DP', 'weight', 'demand mean', 'disrupt. prob.', 'yield loss', 'c_cap'};
% raw_dataTable = array2table(RAW_FEATURES);
raw_dataTable.Properties.VariableNames = raw_featureNames;
raw_dataTable.Response = RESPONSES;

raw_treeModel = fitctree(raw_dataTable, 'Response');
raw_importanceScores = predictorImportance(raw_treeModel);
raw_importanceScores = raw_importanceScores/sum(raw_importanceScores);

% raw_featureNames = dataTable.Properties.VariableNames([1:end-1]); % Exclude 'Response'
for i = 1:numel(raw_featureNames)
    fprintf('Feature: %s, Importance: %.4f\n', raw_featureNames{i}, raw_importanceScores(i));
end

[sortedScores, sortIdx] = sort(raw_importanceScores, 'descend');
sortedFeatureNames = raw_featureNames(sortIdx);

% Plot the bar chart
bar(sortedScores);
set(gca, 'XTickLabel', sortedFeatureNames, 'XTick', 1:numel(sortedFeatureNames));
xlabel('Features');
ylabel('Importance Score');
title('Feature Importance (Sorted)');
grid on;








%% Work on synthetic features
% synth_featureNames = {'Syth. 3DP cost', 'Syth. Profit', 'Synth. Shortfall'};
% synth_dataTable = array2table(SYNTH_FEATURES);
synth_featureNames = {'Syth. 3DP cost', 'Synth. Shortfall'};
synth_dataTable = array2table(SYNTH_FEATURES(:,[1,3]));
synth_dataTable.Properties.VariableNames = synth_featureNames;
synth_dataTable.Response = RESPONSES;

synth_treeModel = fitctree(synth_dataTable, 'Response', 'MaxNumSplits', 3);
synth_importanceScores = predictorImportance(synth_treeModel);
synth_importanceScores = synth_importanceScores/sum(synth_importanceScores);

% synth_featureNames = dataTable.Properties.VariableNames(1:end-1); % Exclude 'Response'
for i = 1:numel(synth_featureNames)
    fprintf('Feature: %s, Importance: %.4f\n', synth_featureNames{i}, synth_importanceScores(i));
end

view(synth_treeModel, 'Mode', 'graph');



%% Plot
feature1 = SYNTH_FEATURES(:,1); % x-axis feature
feature2 = SYNTH_FEATURES(:,3); % y-axis feature
labels = RESPONSES; % 0 or 1 labels

% Separate data points by label
feature1_0 = feature1(labels == 0);
feature2_0 = feature2(labels == 0);
feature1_1 = feature1(labels == 1);
feature2_1 = feature2(labels == 1);

% Plot points with label 0 (black circle with white infill)
scatter(feature1_0, feature2_0, 50, 'k', 'MarkerFaceColor', 'w'); % Size = 50
hold on;

% Plot points with label 1 (solid black circle)
scatter(feature1_1, feature2_1, 50, 'k', 'filled'); % Size = 50

% Add labels and legend
xlabel('3DP Cost');
ylabel('Shortfall');
legend({'Label 0', 'Label 1'}, 'Location', 'best');
title('Scatter Plot with Labels');
hold off;



% Create the first subplot for label 0
subplot(1, 2, 1); % 1 row, 2 columns, first subplot
scatter(feature1_0, feature2_0, 50, 'k', 'MarkerFaceColor', 'w'); % Black circle, white infill
xlabel('3DP Cost');
ylabel('Shortfall');
ylim([0,1])
title('Label 0');
grid on;

% Create the second subplot for label 1
subplot(1, 2, 2); % 1 row, 2 columns, second subplot
scatter(feature1_1, feature2_1, 50, 'k', 'filled'); % Solid black circle
xlabel('3DP Cost');
ylabel('Shortfall');
ylim([0,1])
title('Label 1');
grid on;










treeModel = fitctree(FEATURES, RESPONSES);
% view(treeModel, 'Mode', 'graph'); % Visualize the tree

importanceScores = predictorImportance(treeModel);

% Visualize feature importance
bar(importanceScores);
xlabel('Feature Index');
ylabel('Importance Score');
title('Feature Importance');