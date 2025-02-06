%% In this document, we test the realtive cost savings under various setups

%% IMPORTANT NOTES ON 3DP CAPACITY
%  - The 3DP capacity is defined as the total weight of printing materials we can output during a month
%  - Originally in our data this is measured in gram, but we covert them to kilo-gram so that the landscape of cost is not so "flat" in K !!!
%  - This includes the following changes:
%       - Divide "weight_all" by 1000 (the per unit weight of each product)
%       - Divide "speed_per_machine_month" by 1000 (the total weight of printing materials cosummable per printer per month)
%       - Divide "Monthly_Weight_3scenarios_all" by 1000 (the sceaniros of product demands but measured by weight)



%% Read Data (General Information)
filename = 'Problem_Data/All/Mattel_All_Suppliers_Ave_Weight_Quantity.csv';
all_csvdata = readtable(filename);

weight_all =  all_csvdata.SyntheticProductWeight_gram_; % Weight of the synthetic products (in grams)
weight_all = weight_all/1000; % Rescale the weights to "kg"

c_source_all = all_csvdata.SyntheticSourcingCost;      % Sourcing cost of the synthetic products
c_3DP_source_all = all_csvdata.Synthetic3DPCost;       % 3DP printing cost of the synthetic products
c_TM_source_all = all_csvdata.SyntheticExpeditionCost; % TM cost of the synthetic products
c_price_all  = all_csvdata.SyntheticPrice;             % Selling price of the synethric products

c_3DP_all = c_3DP_source_all - c_source_all; % The c_3DP in our model is the extra cost of 3DP comparing to primary sourcing
c_TM_all = c_TM_source_all - c_source_all;   % The c_TM in our model is the extra cost of TM comparing to primary sourcing
v_all = c_price_all -  c_source_all;         % The lost-sale penalty in our model is the sales margin
h_all = c_source_all;                    % The holding cost in our model is the the loss in salvage (assume salvage income is 0)

num_suppliers_all = length(h_all);


% Percentage of 3DP capacity (to the maximal yield shortfall or maximal demand)
capacity_3dp_percentage = [0.1:0.1:10, 10.2:0.2:15, 15.5:0.5:20, 21:50, 52:2:100]/100;


% Monthly cost of 3DP (for now just assume machine depreciation)
% - $5000 stands for low-end industrial 3DP
% - $10000 stands for entry-level high-end industrial 3DP
% - $50000 stands for standard high-end industrial 3DP
cost_of_3dp_per_machine_month = [5000, 10000, 15000, 17500, 20000, 22500, 25000, 30000, 35000, 40000]/120; 

% Different speed of 3DP (measured in the grams of output materials in a month)
% - 18000 stands for realistically fast with high quality (thickness of layer)
% - 90000 stands for realistically fast with standard quality (thickness of layer)
speed_per_machine_month = [18000, 50000, 90000];
speed_per_machine_month = speed_per_machine_month/1000; % Rescale the weights to "kg"

% Different configurations of disruptions
% - Small disruptions: 10% disruption rate with 1% yield loss
% - Medium disruptions: 5% disruption rate with 5% yield loss
% - Large disruptions: 1% disruption rate with 10% yield loss
p_small = 0.1;   yield_loss_rate_small = 0.01;
p_medium = 0.05; yield_loss_rate_medium = 0.05;
p_large = 0.01;  yield_loss_rate_large = 0.1;




%% We Consider 3-Scenario Random Demand
filename = 'Problem_Data/All/Mattel_All_Suppliers_Ave_Month_Weight_Quantity_3scenarios.csv';
all_csvdata_3scenarios = readtable(filename);
num_scenarios = 3;

weight_sceanarios_col_names = {};
quantity_scenarios_col_names = {};
probability_col_names = {};
for k = 1:num_scenarios
    weight_sceanarios_col_names{1,k} = strcat('WeightScenario', num2str(k), '_grams_');
    quantity_sceanarios_col_names{1,k} = strcat('QuantityScenario', num2str(k));
    probability_col_names{1,k} = strcat('Scenario', num2str(k), 'Probability');
end

% For ease of computation, only take 5
Monthly_Weight_3scenarios_all = [];
Monthly_Quantity_3scenarios_all = [];
Demand_Probability_3scenarios_all = [];
for k = 1:num_scenarios
    Monthly_Weight_3scenarios_all(:,k) = all_csvdata_3scenarios.(weight_sceanarios_col_names{1,k});
    Monthly_Quantity_3scenarios_all(:,k) = all_csvdata_3scenarios.(quantity_sceanarios_col_names{1,k});
    Demand_Probability_3scenarios_all(:,k) = all_csvdata_3scenarios.(probability_col_names{1,k});
end
Monthly_Weight_3scenarios_all = Monthly_Weight_3scenarios_all/1000; % Rescale weights to kg


% Renormalize the probability in case of bad numerical precisions
Demand_Probability_3scenarios_all = Demand_Probability_3scenarios_all./sum(Demand_Probability_3scenarios_all,2);

% Mean demand
mean_demand_3scenarios_all = sum(Monthly_Quantity_3scenarios_all.*Demand_Probability_3scenarios_all,2);






















%% We consider several hyper-parameters:
%% c_cap: controlled by price per printer and printer speed (EMBEDDED IN EVERY RUN)
%% c_3DP: by default is c_source, we can try something cheaper and something more expensive
%% (p, yield_loss_date): by default (0.05, 0.05), we can try other values


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BASED-LINE CASE: COMPUTATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% BASE-LINE CASE:
%%      - c_3DP = 2 * c_source
%%      - (p, yield_loss_date) = (0.05, 0.05)


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

capacity_3dp_percentage = [1e-2, 1e-1*[1:9], 1:2:25, 30:5:50, 75, 100]/100;

fileID1 = fopen('Log4.txt', 'a'); % 'a' mode appends to the file

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


    fprintf(fileID1, 'Benchmark Case,     k=%3.2f %% \n',  capacity_3dp_percentage(k));
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
        X_BOE_BENCHMARK{i,j} = zeros(num_suppliers,length(capacity_3dp_percentage));
        SWITCHED_BOE_BENCHMARK{i,j} = zeros(num_suppliers,length(capacity_3dp_percentage));
    end
end
TOTAL_COST_BENCHMARK = {};
A_t_BOE_BENCHMARK = [];
for k = 1 : length(capacity_3dp_percentage)

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


save("Experiment_Data/Relative_Cost_Savings_Shortfalls/Benchmark.mat")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BASED-LINE CASE: PLOT COST-SAVINGS VS. K
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Summarize the relative cost-savings (PERCENTAGE OF BENCHMARK COST)
COST_SAVINGS_BENCHMARK = [];
for i = 1 : length(speed_per_machine_month)  
    for j = 1 : length(cost_of_3dp_per_machine_month)

        for k = 1 : length(capacity_3dp_percentage)
                
            COST_SAVINGS_BENCHMARK{i,j}(k) = (COST_TMONLY_BENCMARK - OUTPUT_MEDIUM_BOE_BENCHMARK{k}.TOTAL_COST_NONZERO(i,j)) / abs(COST_TMONLY_BENCMARK)*100;
            
        end

    end
end

%% Plot relative cost-savings vs. K 
%% Under different printer-price (fixed printing speed)
cost_of_3dp_per_machine_month_subset = [1,3,7,9];
for i = 1 : length(speed_per_machine_month)  

    for jj = 1 : length(cost_of_3dp_per_machine_month_subset)
       
        j = cost_of_3dp_per_machine_month_subset(jj);
        plot([0,capacity_3dp_percentage]*100, [0,COST_SAVINGS_BENCHMARK{i,j}], '-o','LineWidth', 1)
        hold on
    
    end
    ylimit = max(COST_SAVINGS_BENCHMARK{i,1});
    ylim([-1.1*ylimit,  1.1*ylimit])

    xlim([0,100])

    lgd = legend({'1x Baseline', '3x Baseline', '5x Baseline', '7x Baseline'}, 'FontSize', 12);
    title(lgd, 'Cost per unit 3DP capacity');
    
    ax = gca;
    ax.XTickLabel = strcat(ax.XTickLabel, '%');
    ax.YTickLabel = strcat(ax.YTickLabel, '%');

    filename = strcat( 'Experiment_Data/Relative_Cost_Savings_Shortfalls/Benchmark_Varying_C3DP_Coeffs/Varying_C3DP_Coeff_Fixed_Speed', num2str(i) ,'.pdf');
    
    saveas(gcf,filename)  % as MATLAB figure
    close(gcf)

end

%% Under different printing-speed (fixed printer price)
for jj = 1 : length(cost_of_3dp_per_machine_month_subset)
   
    j = cost_of_3dp_per_machine_month_subset(jj);
    
    for i = 1 : length(speed_per_machine_month) 
        plot([0,capacity_3dp_percentage]*100, [0,COST_SAVINGS_BENCHMARK{i,j}], '-o','LineWidth', 1)
        hold on
    end
    
    ylimit = max(COST_SAVINGS_BENCHMARK{3,j});
    ylim([-1.1*ylimit,  1.1*ylimit])

    xlim([0,100])

    lgd = legend({'1x Baseline', '0.4x Baseline', '0.2x Baseline'}, 'FontSize', 12);
    title(lgd, 'Cost per unit 3DP capacity');

    ax = gca;
    ax.XTickLabel = strcat(ax.XTickLabel, '%');
    ax.YTickLabel = strcat(ax.YTickLabel, '%');
    
    filename = strcat( 'Experiment_Data/Relative_Cost_Savings_Shortfalls/Benchmark_Varying_C3DP_Coeffs/Varying_C3DP_Coeff_Fixed_Printer_Cost', num2str(j) ,'.pdf');
    
    saveas(gcf,filename)  % as MATLAB figure
    close(gcf)

end

%% Combine all the results together
%% Fix the "cost_of_3dp_per_machine_month" as case 1: plot "speed_per_machine_month" case 2, 3
%% Fix the "speed_per_machine_month" as case 1: plot "cost_of_3dp_per_machine_month" case 1, 3, 7, 9
 
figure;
hold on;

iiii = [1,11:length(capacity_3dp_percentage)];

for i = length(speed_per_machine_month) : -1 : 1 
    plot([0,capacity_3dp_percentage(iiii)]*100, [0,COST_SAVINGS_BENCHMARK{i,1}(iiii)], '-o','LineWidth', 1)
    hold on
end

cost_of_3dp_per_machine_month_subset = [1,3,7,9];

for jj = 2 : length(cost_of_3dp_per_machine_month_subset)
   
    j = cost_of_3dp_per_machine_month_subset(jj);
    plot([0,capacity_3dp_percentage(iiii)]*100, [0,COST_SAVINGS_BENCHMARK{1,j}(iiii)], '-o','LineWidth', 1)
    hold on

end

yline(0, '--', 'LineWidth', 1.5, 'Color', [0.5, 0.5, 0.5]);  % Horizontal grey line

x_fill = [0, 100, 100, 0];  % X-coordinates for the shaded area
y_fill = [0, 0, -1e6, -1e6];  % Arbitrary large negative value for shading
fill(x_fill, y_fill, [0.8, 0.8, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);  % Light grey fill

ylimit = max(COST_SAVINGS_BENCHMARK{3,1});
ylim([-1.1*ylimit,  1.1*ylimit])

xlim([0,50])
grid

lgd = legend({'0.2x Baseline', '0.4x Baseline', '1x Baseline', '3x Baseline', '5x Baseline', '7x Baseline'}, 'FontSize', 12, 'location', 'southeast');
title(lgd, 'Cost per unit 3DP capacity');

filename = 'Experiment_Data/Relative_Cost_Savings_Shortfalls/Benchmark_Varying_C3DP_Coeffs/Varying_C3DP_Coeff_All_in_One.pdf';

ax = gca;
ax.XTickLabel = strcat(ax.XTickLabel, '%');
ax.YTickLabel = strcat(ax.YTickLabel, '%');

saveas(gcf,filename)  % as MATLAB figure
close(gcf)



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
file_name = 'Experiment_Data/Relative_Cost_Savings_Shortfalls/benchmark_for_python.csv';
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
% BASED-LINE CASE: PLOT DEMAND SHORTFALLS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SINGLE system: No 3DP
%%      - Get the set of products not backedup A0
%%      - Sample pairs of demand and disruptions
%%      - For each sample (D, s), calculate [D_j - q_j*s_j]^+ and sum over j in A0
%%      - Save the disrbtuionn as a vector

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
Demand_shortfall_no3DP_Benchmark = [];

for ss = 1 : length(disruption_demand_joint_prob_medium_sampled)

    D_sample = demand_data_medium_sampled(:, ss); 
    s_sample = failure_data_medium_sampled(:, ss);
    
    Demand_shortfall_no3DP_Benchmark(ss,1) = sum( max( D_sample - opt_q_nobackup.*s_sample, 0 ) );

end



%% DUO system: After 3DP
%%      - For each capacity K on a grid 
%%      - Get the set of products backedup-by 3DP A
%%      - Sample pairs of demand and disruptions
%%      - For each sample (D, s), calculate optimal 3DP decision q3DP(D,s) and [D_j - q_j*s_j - q_j^3DP(D,s)]^+ and sum over j in A
%%      - Save the disribution as a vector

%% Two important notes:
%%      - For the same K, the deamnd shortfall is the same for all c_cap (q_SP is the same)
%%      - But different c_cap has different opitmla K hence different demand shortfall

%% In what follows, we compute the demand shortfall distribution for each of the c_cap case, specifically:
%%      - for each c_cap, we find the optimal K (max. cost savings)
%%      - retrieve the x and q_SP at the opt K, and comupute demand shortfall
%% (TRUST the cost and q_SP computed by SGD)

Demand_shotfall_varying_C3DP = {};

for i = 1 : length(speed_per_machine_month)  
    for j = 1 : length(cost_of_3dp_per_machine_month)
        
        fprintf("Working with case %d, %d\n\n", i,j)

        %% Get the optimal x and q_SP
        [~,kkk] = min(TOTAL_COST_BENCHMARK{i,j});
        x_3DP = logical(OUTPUT_MEDIUM_BOE_BENCHMARK{1,kkk}.X_FINAL{i,j});
        q_SP = OUTPUT_MEDIUM_BOE_BENCHMARK{1,kkk}.Q_FINAL{i,j}(x_3DP);

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
            
            Demand_shotfall_varying_C3DP{i,j}(ss) = sum( max( D_bar - q_3DP, 0 ) );
    
        end
       


    end
end

%% Get the relative demand shortfall (relative to total max demand)
Relative_Demand_shortfall_varying_C3DP = {};
for i = 1 : length(speed_per_machine_month)  
    for j = 1 : length(cost_of_3dp_per_machine_month)
        Relative_Demand_shortfall_varying_C3DP{i,j} = Demand_shotfall_varying_C3DP{i,j}/sum(max(Monthly_Quantity_3scenarios_all'))*100;
    end
end
Relative_Demand_shortfall_no3DP_Benchmark  = Demand_shortfall_no3DP_Benchmark/sum(max(Monthly_Quantity_3scenarios_all'))*100;


%% First get feeling of what the histogram looks like 
%% Historgarm of demand shortfall with no 3DP
figure;
histogram(Relative_Demand_shortfall_no3DP_Benchmark, 'Normalization', 'probability');%, 'BinWidth', 0.05);

xlabel('Shortfall (% of Max. Demand)', 'FontSize', 12);
ylabel('Probability', 'FontSize', 12);
xt = xticks;
xticklabels(strcat(string(xt), '%'));

% xlim([0,10])
% ylim([0,0.3])

saveas(gcf, 'Experiment_Data/Relative_Cost_Savings_Shortfalls/Benchmark_Varying_C3DP_Coeffs(Shortfalls)/histogram_no3dp.pdf');
close(gcf)

%% Histogram of demand shortfall with 3DP under different c_cap
for i = 1 : length(speed_per_machine_month)  
    for j = 1 : length(cost_of_3dp_per_machine_month)
        
        figure;
        histogram(Relative_Demand_shortfall_varying_C3DP{i,j}, 'Normalization', 'probability');%, 'BinWidth', 0.05);
        
        xlabel('Shortfall (% of Max. Demand)', 'FontSize', 12);
        ylabel('Probability', 'FontSize', 12);
        xt = xticks;
        xticklabels(strcat(string(xt), '%'));
    
        % xlim([0,10])
        % ylim([0,0.3])
        
        filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Benchmark_Varying_C3DP_Coeffs(Shortfalls)/histogram_3dp_varying_C3DP_case', num2str(i), '_', num2str(j), '.pdf');
        saveas(gcf, filename);
        close(gcf)

    end
end


%% Now let's draw the box plots of the distributions
%% First, list box plots side by side
bbb = cost_of_3dp_per_machine_month(1)/speed_per_machine_month(1);

for i = 1 : length(speed_per_machine_month)  

    % Initialize tick labels array
    tick_labels = cell(1, length(cost_of_3dp_per_machine_month) + 1);
    
    % Populate tick labels with 'multiples(i,j)x' for j=1:end-1
    for j = 1:length(cost_of_3dp_per_machine_month)
        multiples(i,j) = (cost_of_3dp_per_machine_month(j)/speed_per_machine_month(i)) / bbb;
        boxplot(Relative_Demand_shortfall_varying_C3DP{i,j}, 'Positions', j, 'Widths', 0.5);
        tick_labels{j} = strcat(num2str(multiples(i,j), '%.2f'), 'x');  % Format with two decimals
        hold on;
    end
    % Add the last boxplot for 'No 3DP'
    boxplot(Relative_Demand_shortfall_no3DP_Benchmark, 'Positions', length(cost_of_3dp_per_machine_month)+1 , 'Widths', 0.5);
    
    % Set the last tick label as 'No 3DP'
    tick_labels{end} = 'No 3DP';
    % Apply tick positions and labels
    xticks(1:length(tick_labels)); 
    xticklabels(tick_labels);

    xlabel('$c^{\mathsf{cap}}$ (Multiple of Baseline)', 'FontSize', 12, 'interpreter', 'latex');
    ylabel('Shortfall (% of Max Demand)', 'FontSize', 12);
    ytickformat('percentage');
    grid on;
    
    % Plot the mean shortdemand
    for j = 1:length(cost_of_3dp_per_machine_month)
        Mean_Relative_Demand_shortfall_varying_C3DP(i,j) = mean(Relative_Demand_shortfall_varying_C3DP{i,j});
    end
    plot([1:length(cost_of_3dp_per_machine_month)+1], [Mean_Relative_Demand_shortfall_varying_C3DP(i,:), mean(Relative_Demand_shortfall_no3DP_Benchmark) ], ...
        '-o', 'LineWidth',2)

    filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Benchmark_Varying_C3DP_Coeffs(Shortfalls)/boxplots_3dp_varying_C3DP_fixed_speed', num2str(i), '.pdf');
    saveas(gcf, filename);
    close(gcf)


end
    
%% Let the c_cap associated with speed case 1 and cost per machine case 1 be the baseline
%% Draw the box plots of the distribution of demand shortfall, and the mean demand shortfall of the following cases:
%%      - 0.2x, 0.4x, 1x, 3x, 5x, no3DP cases

C3DP_subset = [3,1; 2,1; 1,1; 1,3; 1,7];

tick_labels = {'0.2x', '0.4x', '1x', '3x', '5x', 'No 3DP'};

Mean_Relative_Demand_shortfall_varying_C3DP_subset = [];
for ll = 1 : length(C3DP_subset)
    ttt = C3DP_subset(ll,:);
    i = ttt(1);
    j = ttt(2);
    Mean_Relative_Demand_shortfall_varying_C3DP_subset(ll) = Mean_Relative_Demand_shortfall_varying_C3DP(i,j);
end
plot([1:length(C3DP_subset)+1], [Mean_Relative_Demand_shortfall_varying_C3DP_subset, mean(Relative_Demand_shortfall_no3DP_Benchmark) ], ...
    '-o', 'LineWidth',2)
hold on


for ll = 1 : length(C3DP_subset)

    ttt = C3DP_subset(ll,:);
    i = ttt(1);
    j = ttt(2);
    
    boxplot(Relative_Demand_shortfall_varying_C3DP{i,j}, 'Positions', ll, 'Widths', 0.5);
    hold on

end

% Add the last boxplot for 'No 3DP'
boxplot(Relative_Demand_shortfall_no3DP_Benchmark, 'Positions', length(C3DP_subset)+1 , 'Widths', 0.5);

xticks(1:length(tick_labels)); 
xticklabels(tick_labels);

xlabel('$c^{\mathsf{cap}}$ (Multiple of Baseline)', 'FontSize', 12, 'interpreter', 'latex');
ylabel('Shortfall (% of Max Demand)', 'FontSize', 12);
ytickformat('percentage');
grid on;
    
legend({'Mean Deamnd Shortfall', 'Distribution of Demand Shortfall'})


filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Benchmark_Varying_C3DP_Coeffs(Shortfalls)/boxplots_3dp_varying_C3DP_all_in_one.pdf');
saveas(gcf, filename);
close(gcf)



%% Save the data for python
% Initialize arrays for means and boxplot data
Mean_Shortfall_python = [];  % Renamed to avoid overwriting
boxplot_data = [];

% Collect mean data for each subset
for ll = 1:length(C3DP_subset)
    ttt = C3DP_subset(ll,:);
    i = ttt(1);
    j = ttt(2);
    Mean_Shortfall_python(ll) = Mean_Relative_Demand_shotfall_varying_C3DP(i,j);
    % Collect boxplot data for this subset
    current_data = Relative_Demand_shotfall_varying_C3DP{i, j};
    boxplot_data = [boxplot_data; [repmat(ll, length(current_data), 1), current_data(:)]];
end

% Add 'No 3DP' data
Mean_Shortfall_python = [Mean_Shortfall_python, mean(Relative_Demand_shortfall_no3DP_Benchmark)];
no_3dp_data = Relative_Demand_shortfall_no3DP_Benchmark;
boxplot_data = [boxplot_data; [repmat(length(C3DP_subset)+1, length(no_3dp_data), 1), no_3dp_data(:)]];

% Save mean data
mean_table = array2table([1:length(tick_labels); Mean_Shortfall_python]', ...
    'VariableNames', {'Position', 'MeanShortfall'});
writetable(mean_table, 'Experiment_Data/Relative_Cost_Savings_Shortfalls/benchmark_for_python_shortfalls1.csv');

% Save boxplot data
boxplot_table = array2table(boxplot_data, 'VariableNames', {'Position', 'Shortfall'});
writetable(boxplot_table, 'Experiment_Data/Relative_Cost_Savings_Shortfalls/benchmark_for_python_shortfalls2.csv');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


















































%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DIFFERENT c_3DP: COMPUTATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% NOW WE TRY DIFFERENT c_3DP
%%  - baseline is c_3DP = c_source and c_TM = 0.5 * c_source
%%  - we can try: c_3DP = r * c_source, where r = 0.2, 2, 3 
%%          - NOTE WE CAN'T LET c_3DP > 3 * c_source, in such case some products will never be 3DP backed-up)
%%          - when r = 0.2, we let c_TM = 0.1*c_source instead

c_3DP_rate_set = [0.5,2,3];

fileID2 = fopen('Log5.txt', 'a'); % 'a' mode appends to the file

OUTPUT_MEDIUM_BOE_VARYING_c3DP = {};
COST_TMONLY_VARYING_c3DP = {};

for ll = 1 : length(c_3DP_rate_set)



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
    
    % if ll == 1
    %     c_3DP = c_3DP_rate_set(ll) * c_source; 
    %     c_TM = 0.1 * c_source;      
    % else
    %     c_3DP = c_3DP_rate_set(ll) * c_source; 
    %     c_TM = 0.5 * c_source;   
    % end

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
    COST_TMONLY_VARYING_c3DP{ll} = sum(output_medium_TM.TM_cost(TM_backup_set))+sum(output_medium_no3dp.opt_val(logical(nobackup_set)));
    
            
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
            input_boe.A_init = OUTPUT_MEDIUM_BOE_VARYING_c3DP{ll, k-1}.A_t;
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
    
        OUTPUT_MEDIUM_BOE_VARYING_c3DP{ll, k} = output_boe; 
        % TIME_BOE(k) = output_boe.solving_time;
    
    
        fprintf(fileID2, 'c_3DP = %3.2f * c_source,    k=%3.2f %% \n', c_3DP_rate_set(ll), capacity_3dp_percentage(k));
        fprintf(fileID2, '3DP Set: %d', OUTPUT_MEDIUM_BOE_VARYING_c3DP{ll, k}.A_t);
        fprintf(fileID2, '\n\n')
    
    
    
        disp("TO NEXT ONE! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ")
    
    end

end


save("Experiment_Data/Relative_Cost_Savings_Shortfalls/Varying_c3DP.mat")


DDD = load("Experiment_Data/Relative_Cost_Savings_Shortfalls/Benchmark.mat");

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
        
        filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Varying_c3DP(CostSavings)/Varying_c3DP_Speed', num2str(i), '_PrinterCost', num2str(j) ,'.pdf');
        
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

filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Varying_c3DP(CostSavings)/Varying_c3DP_Speed', num2str(i), '_PrinterCost', num2str(j) ,'(PRETTIER).pdf');

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
csv_filename = 'Experiment_Data/Relative_Cost_Savings_Shortfalls/varying_c3DP_for_python_costsavings.csv';
fid = fopen(csv_filename, 'w');
fprintf(fid, '%s,', headers{1:end-1});  % Write column headers
fprintf(fid, '%s\n', headers{end});
fclose(fid);

% Append the data matrix to the file
dlmwrite(csv_filename, data_matrix, '-append', 'precision', '%.6f');

fprintf('Data saved to %s\n', csv_filename);





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BASED-LINE CASE: PLOT DEMAND SHORTFALLS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SINGLE system: No 3DP
%%      - Get the set of products not backedup A0
%%      - Sample pairs of demand and disruptions
%%      - For each sample (D, s), calculate [D_j - q_j*s_j]^+ and sum over j in A0
%%      - Save the disrbtuionn as a vector

%% Get the demand shortfall of the following cases from existing results
%%      - No 3DP (regardless of c_3DP or C3DP)
%%      - Benchmark c_3DP case (all C3DP)
DDD = load("Experiment_Data/Relative_Cost_Savings_Shortfalls/Benchmark.mat");
Relative_Demand_shortfall_no3DP_Benchmark = DDD.Relative_Demand_shortfall_no3DP_Benchmark;
Relative_Demand_shortfall_varying_C3DP = DDD.Relative_Demand_shortfall_varying_C3DP;

%% DUO system: After 3DP
%%      - For each capacity K on a grid 
%%      - Get the set of products backedup-by 3DP A
%%      - Sample pairs of demand and disruptions
%%      - For each sample (D, s), calculate optimal 3DP decision q3DP(D,s) and [D_j - q_j*s_j - q_j^3DP(D,s)]^+ and sum over j in A
%%      - Save the disribution as a vector

%% Two important notes:
%%      - For the same K, the deamnd shortfall is the same for all c_cap (q_SP is the same)
%%      - But different c_cap has different opitmla K hence different demand shortfall 

%% In what follows, we first fix a c_cap case, and then compute the demand shortall distribution for each of c_3DP case:
%%      - for each c_3DP (under the fixed c_cap), we find the optimal K (max. cost savings)
%%      - retrieve the x and q_SP at the opt K, and comupute demand shortfall
%% (TRUST the cost and q_SP computed by SGD)

Demand_shotfall_varying_c3DP = {};

for i = 1 : length(speed_per_machine_month)  
    for j = 1 : length(cost_of_3dp_per_machine_month)
        
        fprintf("Working with case %d, %d\n\n", i,j)

        for ll = 1 : length(c_3DP_rate_set)

            %% Get the optimal x and q_SP
            [~,kkk] = max(COST_SAVINGS_VARYING_c_3DP{ll,i,j});
            x_3DP = logical(OUTPUT_MEDIUM_BOE_VARYING_c3DP{ll,kkk}.X_FINAL{i,j});
            q_SP = OUTPUT_MEDIUM_BOE_VARYING_c3DP{ll,kkk}.Q_FINAL{i,j}(x_3DP);

            %% Preparation
            num_suppliers = sum(x_3DP);
            supplier_subset_idx = x_3DP;
            Monthly_Weight_3scenarios = Monthly_Weight_3scenarios_all(supplier_subset_idx, :);
            Monthly_Quantity_3scenarios = Monthly_Quantity_3scenarios_all(supplier_subset_idx, :);
            Demand_Probability_3scenarios = Demand_Probability_3scenarios_all(supplier_subset_idx, :);
            mean_demand_3scenarios = mean_demand_3scenarios_all(supplier_subset_idx);
                   
            c_source = c_source_all(supplier_subset_idx);        
            c_3DP = c_3DP_all(supplier_subset_idx)*c_3DP_rate_set(ll); 
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
                
                Demand_shotfall_varying_c3DP{i,j,ll}(ss) = sum( max( D_bar - q_3DP, 0 ) );
        
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


%% First get feeling of what the histogram looks like 
%% Historgarm of demand shortfall with no 3DP
figure;
histogram(Relative_Demand_shortfall_no3DP_Benchmark, 'Normalization', 'probability');%, 'BinWidth', 0.05);

xlabel('Shortfall (% of Max. Demand)', 'FontSize', 12);
ylabel('Probability', 'FontSize', 12);
xt = xticks;
xticklabels(strcat(string(xt), '%'));

% xlim([0,10])
% ylim([0,0.3])

saveas(gcf, 'Experiment_Data/Relative_Cost_Savings_Shortfalls/Varying_c3DP(Shortfalls)/histogram_no3dp.pdf');
close(gcf)

%% Histogram of demand shortfall with 3DP under different c_cap
for i = 1 : length(speed_per_machine_month)  
    for j = 1 : length(cost_of_3dp_per_machine_month)
        
        for ll = 1 : length(c_3DP_rate_set)

            figure;
            histogram(Relative_Demand_shortfall_varying_c3DP{i,j,ll}, 'Normalization', 'probability');%, 'BinWidth', 0.05);
            
            xlabel('Shortfall (% of Max. Demand)', 'FontSize', 12);
            ylabel('Probability', 'FontSize', 12);
            xt = xticks;
            xticklabels(strcat(string(xt), '%'));
        
            % xlim([0,10])
            % ylim([0,0.3])
            
            filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Varying_c3DP(Shortfalls)/histogram_3dp_varying_C3DPcase_', num2str(i), '_', num2str(j), '_c3DPcase',num2str(ll), '.pdf');
            saveas(gcf, filename);
            close(gcf)

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
    
        filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Varying_c3DP(Shortfalls)/boxplots_3dp_varying_c3DP_C3DPcase_', num2str(i), '_', num2str(j), '.pdf');
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
writetable(mean_table, 'Experiment_Data/Relative_Cost_Savings_Shortfalls/varying_c3DP_for_python_shortfalls1.csv');

% Save boxplot data
boxplot_table = array2table(boxplot_data, 'VariableNames', {'Position', 'Shortfall'});
writetable(boxplot_table, 'Experiment_Data/Relative_Cost_Savings_Shortfalls/varying_c3DP_for_python_shortfalls2.csv');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%










































%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DISRUPTION MODELING (INDEPENDENT): COMPUTATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% NOW WE TRY DIFFERENT DISRUPTION MODELING (INDEPENDENT)
%%  - baseline is (p, yield_loss_rate) = (0.05, 0.05)
%%  - we can try: 
%%          - p = 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5
%%          - yield_loss_rate = 0.05,  0.25,  0.5,  0.75


p_set = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5];
yield_loss_rate_set = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];


OUTPUT_MEDIUM_BOE_VARYING_DISRUPTIONS = {};
COST_TMONLY_VARYING_DISRUPTIONS = {};

fileID3 = fopen('Log6.txt', 'a'); % 'a' mode appends to the file

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
                
            
                % input_boe.A_init = [1:num_suppliers];
                % output_boe2 = BoE_Approx_Max_Submod1(input_boe);
                % 
                % if output_boe1.TOTAL_COST(1,1) <  output_boe2.TOTAL_COST(1,1) 
                %     output_boe = output_boe1;
                % else
                %     output_boe = output_boe2;
                % end
            
                OUTPUT_MEDIUM_BOE_VARYING_DISRUPTIONS{pp, yy, k} = output_boe; 
                % TIME_BOE(k) = output_boe.solving_time;
            
            
                fprintf(fileID3, 'Varying Disruption %d, %d,     k=%3.2f %% \n', pp,yy, capacity_3dp_percentage(k));
                fprintf(fileID3, '3DP Set: %d ', OUTPUT_MEDIUM_BOE_VARYING_DISRUPTIONS{pp, yy, k}.A_t);
                fprintf(fileID3, '\n\n');
            
            
            
                disp("TO NEXT ONE! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ")
            
            end
    

        
        end

    end
end


% save("Experiment_Data/Relative_Cost_Savings_Shortfalls/Varying_DisruptionDistr_Ind_Comono_all_in_one.mat")



DDD = load("Experiment_Data/Relative_Cost_Savings_Shortfalls/Benchmark.mat");

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


%% Plot Cost-Savings 
%% Always loop over all combos of "speed_per_machine_month" and "cost_of_3dp_per_machine_month"


nColors1 = length(yield_loss_rate_set);
nColors2 = length(p_set);
defaultColors = lines(7); % Default colors for up to 7 lines
if nColors1 <= 7
    colors1 = defaultColors; % Use default colors if nColors <= 7
else
    colors1 = [defaultColors; rand(nColors1 - 7, 3)]; % Default for first 7, random for the rest
end
if nColors2 <= 7
    colors2 = defaultColors; % Use default colors if nColors <= 7
else
    colors2 = [defaultColors; rand(nColors2 - 7, 3)]; % Default for first 7, random for the rest
end

for i = 1 : length(speed_per_machine_month)  
    for j = 1 : length(cost_of_3dp_per_machine_month)

        %% First, fix p, then plot different curves of yield_loss_rate in one plot
        for pp = 1 : length(p_set)

            for yy = 1 : length(yield_loss_rate_set)

                plot([0,capacity_3dp_percentage]*100, [0,COST_SAVINGS_VARYING_DISRUPTIONS{pp,yy,i,j}], '-o','LineWidth', 1, 'Color', colors1(yy,:))
                hold on

            end

            xlim([0,100])
            
            lgd = legend({'0.2x Baseline', '1x Baseline', '2x Baseline', '4x Baseline', '6x Baseline', '8x Baseline', '10x Baseline', '12x Baseline', '14x Baseline'}, ...
                'FontSize', 12, 'location', 'northeast');
            title(lgd, 'Yield Loss Rate');
          
            filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Varying_p_yield_loss(CostSavings)/Varying_yield_loss', '_fixed_p_',num2str(pp), '_C3DP_CASE', num2str(i), num2str(j), '.pdf');
            
            ax = gca;
            ax.XTickLabel = strcat(ax.XTickLabel, '%');
            ax.YTickLabel = strcat(ax.YTickLabel, '%');

            hold off
            saveas(gcf,filename)  % as MATLAB figure
            close(gcf)

        end

        %% Now, fix yield loss, then plot different curves of p in one plot
        for yy = 1 : length(yield_loss_rate_set)

            for pp = 1 : length(p_set)

                plot([0,capacity_3dp_percentage]*100, [0,COST_SAVINGS_VARYING_DISRUPTIONS{pp,yy,i,j}], '-o','LineWidth', 1, 'Color', colors2(pp,:))
                hold on

            end

            xlim([0,100])
            
            lgd = legend({'0.2x Baseline', '1x Baseline', '2x Baseline', '3x Baseline', '4x Baseline', '5x Baseline', '6x Baseline', '7x Baseline', '8x Baseline', '9x Baseline', '10x Baseline'}, ...
                'FontSize', 12, 'location', 'northeast');
            title(lgd, 'Marginal Disruption Rate');
          
            filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Varying_p_yield_loss(CostSavings)/Varying_p', '_fixed_yieldloss_',num2str(yy), '_C3DP_CASE', num2str(i), num2str(j), '.pdf');
            
            ax = gca;
            ax.XTickLabel = strcat(ax.XTickLabel, '%');
            ax.YTickLabel = strcat(ax.YTickLabel, '%');

            hold off
            saveas(gcf,filename)  % as MATLAB figure
            close(gcf)

        end


    end
end

%% Now for each combo of "speed_per_machine_month" and "cost_of_3dp_per_machine_month"
%% Compare the cost-savings under the followings:
%%      - Very Frequent and large scale:          p=0.5, yield_loss=0.7
%%      - Very Frequent and moderate scale:       p=0.5, yield_loss=0.4
%%      - Very Frequent and small scale:          p=0.5, yield_loss=0.01
%%      - Moderately Frequent and large scale:    p=0.3, yield_loss=0.7
%%      - Moderately Frequent and moderate scale: p=0.3, yield_loss=0.4
%%      - Moderately Frequent and small scale:    p=0.3, yield_loss=0.01
%%      - Rare and large scale:                   p=0.01, yield_loss=0.7
%%      - Rare and moderate scale:                p=0.01, yield_loss=0.4
%%      - Rare and small scale:                   p=0.01, yield_loss=0.01

% p_yield_combo = [7,3; 7,2; 7,1; 5,3; 5,2; 5,1; 1,3; 1,2; 1,1];
% p_yield_combo = [7,4; 7,1; 1,4; 1,1];
% p_yield_combo = [11,9; 11,6; 11,2; 5,9; 5,6; 5,2; 1,9; 1,6; 1,2];
p_yield_combo = [11,9; 11,1; 6,6; 1,9; 1,1];
% p_yield_combo = [7,6; 7,1; 5,4; 1,6; 1,1];


nColors3 = length(p_yield_combo);
defaultColors = lines(7); % Default colors for up to 7 lines
if nColors3 <= 7
    colors3 = defaultColors; % Use default colors if nColors <= 7
else
    colors3 = [defaultColors; rand(nColors3 - 7, 3)]; % Default for first 7, random for the rest
end

for combo = 1 : length(p_yield_combo)
    plot([combo,combo],[0,10], 'Color', colors3(combo,:),'LineWidth',2)
    hold on
end
xlim([-1,10])

for i = 1 : length(speed_per_machine_month)  
    for j = 1 : length(cost_of_3dp_per_machine_month)

        for combo = 1 : length(p_yield_combo)
        
            ttt = p_yield_combo(combo,:);
            pp = ttt(1); yy = ttt(2);

            plot([0,capacity_3dp_percentage]*100, [0,COST_SAVINGS_VARYING_DISRUPTIONS{pp,yy,i,j}], '-o','LineWidth', 1, 'Color', colors3(combo,:))
            hold on
        
        end

        xlim([0,100])
        % lgd = legend({'High-High', 'High-Mid', 'High-Low', 'Mid-High', 'Mid-Mid', 'Mid-Low', 'Low-High', 'Low-Mid', 'Low-Low'}, 'FontSize', 12, 'Location', 'eastoutside');
        lgd = legend({'High-High', 'High-Low', 'Mid-Mid', 'Low-High', 'Low-Low'}, 'FontSize', 12, 'location', 'southeast'); 
        title(lgd, '(Freq, Yield Loss)');

        filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Varying_p_yield_loss(CostSavings)/CHIGH_MID_LOW_C3DP_CASE', num2str(i), num2str(j), '.pdf');
        
        ax = gca;
        ax.XTickLabel = strcat(ax.XTickLabel, '%');
        ax.YTickLabel = strcat(ax.YTickLabel, '%');

        hold off
        saveas(gcf,filename)  % as MATLAB figure
        close(gcf)

    end
end



%% For each combo of "speed_per_machine_month" and "cost_of_3dp_per_machine_month", draw two plots
%%      - (Plot 1) for each p, draw the the following curve: let yield loss rate increases, for each yield loss rate, get the max. cost savings among K
%%      - (Plot 2) for each yield loss rate, draw the the following curve: let p increases, for each p, get the max. cost savings among K  

COST_SAVINGS_VARYING_DISRUPTIONS_MAX_AMONG_K = {};

for i = 1 : length(speed_per_machine_month)  
    for j = 1 : length(cost_of_3dp_per_machine_month)
        
        for yy = 1 : length(yield_loss_rate_set)
            for pp = 1 : length(p_set)

                COST_SAVINGS_VARYING_DISRUPTIONS_MAX_AMONG_K{i,j}(pp,yy) = max(COST_SAVINGS_VARYING_DISRUPTIONS{pp,yy,i,j});

            end
        end

    end
end

p_subset = [1,6,11];
yield_loss_rate_subset = [1,6,9];

% p_subset = [1:length(p_set)];
% yield_loss_rate_subset = [1:length(yield_loss_rate_set)];

for i = 1 : length(speed_per_machine_month)  
    for j = 1 : length(cost_of_3dp_per_machine_month)

        %% Plot 1
        for ppp = 1 : length(p_subset)
            pp = p_subset(ppp);
            plot(yield_loss_rate_set, COST_SAVINGS_VARYING_DISRUPTIONS_MAX_AMONG_K{i,j}(pp,:), '-o','LineWidth', 1, 'Color', colors2(ppp,:))
            hold on
        end
        % lgd = legend({'0.2x Baseline', '1x Baseline', '2x Baseline', '3x Baseline', '4x Baseline', '5x Baseline', '6x Baseline', '7x Baseline', '8x Baseline', '9x Baseline', '10x Baseline'}, ...
        %     'FontSize', 12, 'location', 'eastoutside');
        lgd = legend({'1x Baseline', '5x Baseline', '10x Baseline'}, ...
            'FontSize', 12, 'location', 'eastoutside');
        title(lgd, 'Marginal Disruption Rate');
        xlabel("Yield Loss Rate", 'FontSize',15)

        filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Varying_p_yield_loss(CostSavings)/OOPTCOSTSAVINGS_FIXED_YIELDLOSS_CASE', num2str(i), num2str(j), '.pdf');
        hold off
        saveas(gcf,filename)  % as MATLAB figure
        close(gcf)
                  
        %% Plot 2
        for yyy = 1 : length(yield_loss_rate_subset)
            yy = yield_loss_rate_subset(yyy);
            plot(p_set, COST_SAVINGS_VARYING_DISRUPTIONS_MAX_AMONG_K{i,j}(:,yy)', '-o','LineWidth', 1, 'Color', colors1(yyy,:))
            hold on
        end
        % lgd = legend({'0.2x Baseline', '1x Baseline', '2x Baseline', '4x Baseline', '6x Baseline', '8x Baseline', '10x Baseline', '12x Baseline', '14x Baseline'}, ...
        %     'FontSize', 12, 'location', 'eastoutside');
        lgd = legend({'1x Baseline', '8x Baseline', '14x Baseline'}, ...
            'FontSize', 12, 'location', 'eastoutside');
        title(lgd, 'Yield Loss Rate');
        xlabel("Marginal Failure Rate", 'FontSize',15)

        filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Varying_p_yield_loss(CostSavings)/OOPTCOSTSAVINGS_FIXED_P_CASE', num2str(i), num2str(j), '.pdf');
        hold off
        saveas(gcf,filename)  % as MATLAB figure
        close(gcf)

    end
end



%% The story is set now, we plot 3 plots (under a selected combo for C3DP)
%%      - Plot 1: for each p, draw the the following curves: let yield loss rate increases, for each yield loss rate, get the max. cost savings among K
%%      - Plot 2: for each yield loss rate, draw the the following curve: let p increases, for each p, get the max. cost savings among K  
%% From these two pictures, we understand:
%%      - the general rule of thumb is: cost-savings first increase then decrease with both paramters
%%      - we expect "high-high" to be bad, and other cases to be ok
%% So we present plot 3:
%%      - 5 curves: high-high, high-low, low-high, mid-mid

p_subset = [1,7,11];
yield_loss_rate_subset = [1,6,9];


i = 1; j = 1;

%% Plot 1
for ppp = 1 : length(p_subset)
    pp = p_subset(ppp);
    plot(yield_loss_rate_set, COST_SAVINGS_VARYING_DISRUPTIONS_MAX_AMONG_K{i,j}(pp,:), '-o','LineWidth', 1, 'Color', colors2(ppp,:))
    hold on
end
% lgd = legend({'0.2x Baseline', '1x Baseline', '2x Baseline', '3x Baseline', '4x Baseline', '5x Baseline', '6x Baseline', '7x Baseline', '8x Baseline', '9x Baseline', '10x Baseline'}, ...
%     'FontSize', 12, 'location', 'eastoutside');
lgd = legend({'1x Baseline', '5x Baseline', '10x Baseline'}, ...
    'FontSize', 12, 'location', 'eastoutside');
title(lgd, 'Marginal Disruption Rate');
xlabel("Yield Loss Rate", 'FontSize',15)

filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Varying_p_yield_loss(CostSavings)/AAACOSTSAVINGS_FIXED_YIELDLOSS_CASE', num2str(i), num2str(j), '.pdf');
hold off
saveas(gcf,filename)  % as MATLAB figure
close(gcf)
          
%% Plot 2
for yyy = 1 : length(yield_loss_rate_subset)
    yy = yield_loss_rate_subset(yyy);
    plot(p_set, COST_SAVINGS_VARYING_DISRUPTIONS_MAX_AMONG_K{i,j}(:,yy)', '-o','LineWidth', 1, 'Color', colors1(yyy,:))
    hold on
end
% lgd = legend({'0.2x Baseline', '1x Baseline', '2x Baseline', '4x Baseline', '6x Baseline', '8x Baseline', '10x Baseline', '12x Baseline', '14x Baseline'}, ...
%     'FontSize', 12, 'location', 'eastoutside');
lgd = legend({'1x Baseline', '8x Baseline', '14x Baseline'}, ...
    'FontSize', 12, 'location', 'eastoutside');
title(lgd, 'Yield Loss Rate');
xlabel("Marginal Failure Rate", 'FontSize',15)

filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Varying_p_yield_loss(CostSavings)/AAACOSTSAVINGS_FIXED_P_CASE', num2str(i), num2str(j), '.pdf');
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
% lgd = legend({'High-High', 'High-Mid', 'High-Low', 'Mid-High', 'Mid-Mid', 'Mid-Low', 'Low-High', 'Low-Mid', 'Low-Low'}, 'FontSize', 12, 'Location', 'eastoutside');
lgd = legend({'High-High', 'High-Low', 'Mid-Mid', 'Low-High', 'Low-Low'}, 'FontSize', 12, 'location', 'southeast'); 
title(lgd, '(Freq, Yield Loss)');

filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Varying_p_yield_loss(CostSavings)/AAAHIGH_MID_LOW_C3DP_CASE', num2str(i), num2str(j), '.pdf');

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

filename = 'Experiment_Data/Relative_Cost_Savings_Shortfalls/varying_disruption_distr_ind_for_python_costsavings.xlsx';

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
% BASED-LINE CASE: PLOT DEMAND SHORTFALLS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SINGLE system: No 3DP
%%      - Get the set of products not backedup A0
%%      - Sample pairs of demand and disruptions
%%      - For each sample (D, s), calculate [D_j - q_j*s_j]^+ and sum over j in A0
%%      - Save the disrbtuionn as a vector

%% Becareful!!! In this case, even the SINGLE system is changing with p and yield loss

%% DUO system: After 3DP
%%      - For each capacity K on a grid 
%%      - Get the set of products backedup-by 3DP A
%%      - Sample pairs of demand and disruptions
%%      - For each sample (D, s), calculate optimal 3DP decision q3DP(D,s) and [D_j - q_j*s_j - q_j^3DP(D,s)]^+ and sum over j in A
%%      - Save the disribution as a vector

%% Two important notes:
%%      - For the same K, the deamnd shortfall is the same for all c_cap (q_SP is the same)
%%      - But different c_cap has different opitmla K hence different demand shortfall 

%% In what follows, we first fix a c_cap case ("11"), and then compute the demand shortall distribution for each of p and yield_loss case:
%%      - for each pair of p and yield_loss (under the fixed c_cap), we find the optimal K (max. cost savings)
%%      - retrieve the x and q_SP at the opt K, and comupute demand shortfall
%% (TRUST the cost and q_SP computed by SGD)



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




       
%% First get feeling of what the histogram looks like 
%% Historgarm of demand shortfall with no 3DP
for pp = 1 : length(p_set)
    for yy = 1 : length(yield_loss_rate_set)
        figure;
        histogram(Relative_Demand_shortfall_no3DP_varying_disruptions{pp,yy},'Normalization', 'probability')
        xlabel('Shortfall (% of Max. Demand)', 'FontSize', 12);
        ylabel('Probability', 'FontSize', 12);
        xt = xticks;
        xticklabels(strcat(string(xt), '%'));

        saveas(gcf, strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Varying_p_yield_loss(Shortfalls)/histogram_no3dp_p_', num2str(pp), '_yield_loss_', num2str(yy), '.pdf'));
        close(gcf)

    end
end

%% Historgarm of demand shortfall with 3DP
for pp = 1 : length(p_set)
    for yy = 1 : length(yield_loss_rate_set)
        figure;
        histogram(Relative_Demand_shotfall_varying_disruptions{pp,yy,i,j},   'Normalization', 'probability')
        xlabel('Shortfall (% of Max. Demand)', 'FontSize', 12);
        ylabel('Probability', 'FontSize', 12);
        xt = xticks;
        xticklabels(strcat(string(xt), '%'));

        saveas(gcf, strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Varying_p_yield_loss(Shortfalls)/histogram_3dp_C3DPcase_', num2str(i), num2str(j), ...
            '_p_', num2str(pp), '_yieldloss_', num2str(yy), '.pdf'));
        close(gcf)

    end
end


%% Box plots 
%% For each fixed p, we plot several sets of boxes, each corresponds to a yield loss, and contain two boxes: with and without 3DP
%% For each fixed yield loss, we plot several sets of boxes, each corresponds ot a p, and contain two boxes: with and without 3DP

Mean_Relative_Demand_shotfall_varying_disruptions = {};
Mean_Relative_Demand_shortfall_no3DP_varying_disruptions = [];

tick_labels = {'0.2x 3DP', '0.2x No3DP', '1x 3DP', '1x No3DP', '2x 3DP', '2x No3DP', '4x 3DP', '4x No3DP',...
     '6x 3DP', '6x No3DP',  '8x 3DP', '8x No3DP',  '10x 3DP', '10x No3DP',  '12x 3DP', '12x No3DP',  '14x 3DP', '14x No3DP'};

for pp = 1 : length(p_set)

    for yy = 1 : length(yield_loss_rate_set)
        
        boxplot(Relative_Demand_shotfall_varying_disruptions{pp,yy,i,j}, 'Positions', 2*yy-1, 'Widths', 0.5, 'Symbol', '');
        hold on
        boxplot(Relative_Demand_shortfall_no3DP_varying_disruptions{pp,yy}, 'Positions', 2*yy, 'Widths', 0.5, 'Symbol', '', 'Colors', 'r', 'MedianStyle', 'target');
        hold on
        
        Mean_Relative_Demand_shotfall_varying_disruptions{i,j}(pp,yy) = mean(Relative_Demand_shotfall_varying_disruptions{pp,yy,i,j});
        Mean_Relative_Demand_shortfall_no3DP_varying_disruptions(pp,yy) = mean(Relative_Demand_shortfall_no3DP_varying_disruptions{pp,yy});
        
    end
    
    plot( 2*[1:length(yield_loss_rate_set)]-1, Mean_Relative_Demand_shotfall_varying_disruptions{i,j}(pp,:), '-o', 'LineWidth', 1.5)
    hold on
    plot( 2*[1:length(yield_loss_rate_set)], Mean_Relative_Demand_shortfall_no3DP_varying_disruptions(pp,:), '-o', 'LineWidth', 1.5)

    xticks(1:length(tick_labels)); 
    xticklabels(tick_labels);

    xlabel('Yield Loss Ratio (Multiple of Baseline)', 'FontSize', 12);
    ylabel('Shortfall (% of Max Demand)', 'FontSize', 12);
    ytickformat('percentage');

    ylim([-0.333,6])
    % grid on;

    filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Varying_p_yield_loss(Shortfalls)/boxplots_C3DPcase_', num2str(i), num2str(j), '_varying_yieldloss_fixed_p_case', num2str(pp), '.pdf');
    saveas(gcf, filename);
    close(gcf)

end


tick_labels = {'0.2x 3DP', '0.2x No3DP', '1x 3DP', '1x No3DP', '2x 3DP', '2x No3DP', '3x 3DP', '3x No3DP', '4x 3DP', '4x No3DP', ...
     '5x 3DP', '5x No3DP', '6x 3DP', '6x No3DP', '7x 3DP', '7x No3DP', '8x 3DP', '8x No3DP', '9x 3DP', '9x No3DP', '10x 3DP', '10x No3DP'};

for yy = 1 : length(yield_loss_rate_set)

    for pp = 1 : length(p_set)

        boxplot(Relative_Demand_shotfall_varying_disruptions{pp,yy,i,j}, 'Positions', 2*pp-1, 'Widths', 0.5, 'Symbol', '');
        hold on
        boxplot(Relative_Demand_shortfall_no3DP_varying_disruptions{pp,yy}, 'Positions', 2*pp, 'Widths', 0.5, 'Symbol', '', 'Colors', 'r', 'MedianStyle', 'target');
        hold on
        
    end
    
    plot( 2*[1:length(p_set)]-1, Mean_Relative_Demand_shotfall_varying_disruptions{i,j}(:,yy)', '-o', 'LineWidth', 1.5)
    hold on
    plot( 2*[1:length(p_set)], Mean_Relative_Demand_shortfall_no3DP_varying_disruptions(:,yy)', '-o', 'LineWidth', 1.5)

    xticks(1:length(tick_labels)); 
    xticklabels(tick_labels);

    xlabel('Marginal Failure Rate (Multiple of Baseline)', 'FontSize', 12);
    ylabel('Shortfall (% of Max Demand)', 'FontSize', 12);
    ytickformat('percentage');

    ylim([-0.333,6])
    % grid on;

    filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Varying_p_yield_loss(Shortfalls)/boxplots_C3DPcase_', num2str(i), num2str(j), '_varying_p_fixed_yieldloss_case', num2str(yy), '.pdf');
    saveas(gcf, filename);
    close(gcf)

end







%% The box plots we have so far are too raw and dense, let's select some subsets to plot
%% Fix p varying yield loss
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

    filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Varying_p_yield_loss(Shortfalls)/AAAAboxplots_C3DPcase_', num2str(i), num2str(j), 'fixed_p_varying_yieldloss_case', num2str(pp), '.pdf');
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
save('Experiment_Data/Relative_Cost_Savings_Shortfalls/varying_disruption_distr_ind_for_python_shortfalls1.mat', 'Box_plot_data11', 'Box_plot_data12', 'mean_plot_data11', 'mean_plot_data12', ...
     'box_plot_pos11', 'box_plot_pos12', 'x_ticks_labels1', 'x_ticks_pos1', 'vertline_pos1', 'xlimit1', 'ylimit1');







%% Fix yield loss varying p

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

    filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Varying_p_yield_loss(Shortfalls)/AAAboxplots_C3DPcase_', num2str(i), num2str(j), '_fixed_yieldloss_varying_p_case', num2str(yy), '.pdf');
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

save('Experiment_Data/Relative_Cost_Savings_Shortfalls/varying_disruption_distr_ind_for_python_shortfalls2.mat', ...
    'Box_plot_data21', 'Box_plot_data22', ...
    'mean_plot_data21', 'mean_plot_data22', ...
    'box_plot_pos21', 'box_plot_pos22', ...
    'x_ticks_labels2', 'x_ticks_pos2', ...
    'vertline_pos2', 'xlimit2', 'ylimit2');







% %% Box plots can be messy, the main reason is that now the no 3DP baseline also changes, make it a little harder to present
% %% We just present all the cases in mean, in this case, we can compare the improvement to the no 3DP case
% 
% Improved_Mean_Relative_Demand_shortfall_varying_disruptions = ...
%     Mean_Relative_Demand_shortfall_no3DP_varying_disruptions - Mean_Relative_Demand_shotfall_varying_disruptions{i,j} ;
% 
% curves_name1 = {'0.2x', '1x', '2x', '3x', '4x', '5x', '6x', '7x', '8x', '9x', '10x'};
% p_subset = [2:2:11];
% % p_subset = [1,7,11];
% 
% for ppp = 1 : length(p_subset)
%     pp = p_subset(ppp);
%     plot(yield_loss_rate_set, Improved_Mean_Relative_Demand_shortfall_varying_disruptions(pp,:), '-o', 'LineWidth', 2)
%     hold on
% end
% 
% x_fill = [0, 0, 0.7, 0.7]; % X coordinates of the corners
% y_fill = [0, -2, -2, 0]; % Y coordinates of the corners
% fill(x_fill, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.333, 'EdgeColor', 'none'); % Transparent grey fill
% 
% lgd = legend(curves_name1(p_subset), 'Location', 'southwest', 'FontSize', 12);
% title(lgd, 'Marginal Disruption Rate');
% xlabel("Yield Loss Ratio", 'FontSize',15)
% 
% ylabel('Improved Mean Shortfall (% of Max Demand)', 'FontSize', 12);
% ytickformat('percentage');
% 
% filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Varying_p_yield_loss(Shortfalls)/mean_shortall_C3DPcase_', num2str(i), num2str(j), '_varying_p_fixed_yieldloss_mean_shortall.pdf');
% saveas(gcf, filename);
% close(gcf)
% 
% 
% curves_name2 = {'0.2x', '1x', '2x', '4x', '6x', '8x', '10x', '12x', '14x'};
% yield_loss_rate_subset = [1:2:9];
% % yield_loss_rate_subset = [1,6,9];
% 
% for yyy = 1 : length(yield_loss_rate_subset)
%     yy = yield_loss_rate_subset(yyy);
%     plot(p_set, Improved_Mean_Relative_Demand_shortfall_varying_disruptions(:,yy)', '-o', 'LineWidth', 2)
%     hold on
% end
% 
% x_fill = [0, 0, 0.5, 0.5]; % X coordinates of the corners
% y_fill = [0, -2, -2, 0]; % Y coordinates of the corners
% fill(x_fill, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.333, 'EdgeColor', 'none'); % Transparent grey fill
% 
% lgd = legend(curves_name2(yield_loss_rate_subset), 'Location', 'southwest', 'FontSize', 12);
% title(lgd, 'Yield Loss Ratio');
% xlabel("Marginal Disruption Rate", 'FontSize',15)
% 
% ylabel('Improved Mean Shortfall (% of Max Demand)', 'FontSize', 12);
% ytickformat('percentage');
% 
% filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Varying_p_yield_loss(Shortfalls)/mean_shortall_C3DPcase_', num2str(i), num2str(j), '_varying_yieldloss_fixed_p.pdf');
% saveas(gcf, filename);
% close(gcf)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

















































































%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DISRUPTION MODELING (COMONOTONIC): COMPUTATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% NOW WE TRY COMONOTONIC DISRUPTIONS

p_set = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5];
yield_loss_rate_set = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];


OUTPUT_MEDIUM_BOE_COMONO = {};
COST_TMONLY_COMONO = {};

fileID3 = fopen('Log8.txt', 'a'); % 'a' mode appends to the file

capacity_3dp_percentage = [1e-2, 1e-1*[1:9], 1:2:25, 30:5:50, 75, 100]/100;

for pp = 1 : length(p_set)
    for yy = 1 : length(yield_loss_rate_set)


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
        COST_TMONLY_COMONO{pp,yy} = sum(output_medium_TM.TM_cost(TM_backup_set))+sum(output_medium_no3dp.opt_val(logical(nobackup_set)));
        
                
        %% Sample some data for BoE Submod Max SAA (COMONO CASE)
        input_preprocess_medium_sampled.num_suppliers = num_suppliers;
        input_preprocess_medium_sampled.num_scenarios = num_scenarios;
        input_preprocess_medium_sampled.yield_loss_rate = yield_loss_rate_medium;
        input_preprocess_medium_sampled.p_disrupt = p_medium;
        input_preprocess_medium_sampled.Monthly_Quantity = Monthly_Quantity_3scenarios;
        input_preprocess_medium_sampled.Demand_Probability = Demand_Probability_3scenarios;
        
        input_preprocess_medium_sampled.sample_mode = 3;
        input_preprocess_medium_sampled.demand_sample_flag = 1;     % For each disruption scenario, sample a fixed number of demand combos
        Demand_sample_size_eval = ceil(500*250/2);
        input_preprocess_medium_sampled.demand_samplesize = Demand_sample_size_eval;
        
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
            
            input_boe.recompute_flag = 2;
        
            input_boe.recompute_sample_mode = 3;
            input_boe.recompute_demand_sample_flag = 1;            
            input_boe.recompute_demand_samplesize_eval = 500; 
            input_boe.recompute_demand_samplesize_finaleval = 500;

            input_boe.recompute_distr = 2;
        
            input_boe.recompute_sgd_Maxsteps = 5e5;
            
            if k == 1
                input_boe.A_init = [];
            else
                input_boe.A_init = OUTPUT_MEDIUM_BOE_COMONO{pp, yy, k-1}.A_t;
            end
        
            output_boe = BoE_Approx_Max_Submod_SAA_alternative(input_boe);
            
        
            % input_boe.A_init = [1:num_suppliers];
            % output_boe2 = BoE_Approx_Max_Submod1(input_boe);
            % 
            % if output_boe1.TOTAL_COST(1,1) <  output_boe2.TOTAL_COST(1,1) 
            %     output_boe = output_boe1;
            % else
            %     output_boe = output_boe2;
            % end
        
            OUTPUT_MEDIUM_BOE_COMONO{pp, yy, k} = output_boe; 
            % TIME_BOE(k) = output_boe.solving_time;
        
        
            fprintf(fileID3, 'Varying Disruption %d, %d,     k=%3.2f %% \n', pp,yy, capacity_3dp_percentage(k));
            fprintf(fileID3, '3DP Set: %d ', OUTPUT_MEDIUM_BOE_COMONO{pp, yy, k}.A_t);
            fprintf(fileID3, '\n\n');
        
        
        
            disp("TO NEXT ONE! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ")
        
        end


        

    end
end


% save("Experiment_Data/Relative_Cost_Savings_Shortfalls/Comono_Disruptions_Latest.mat")
save("Experiment_Data/Relative_Cost_Savings_Shortfalls/Varying_DisruptionDistr_Ind_Comono_all_in_one.mat")

%% Summarize the relative cost-savings (PERCENTAGE OF BASELINE COST)

COST_SAVINGS_COMONO = {};
for pp = 1 : length(p_set)
    for yy = 1 : length(yield_loss_rate_set)

        for i = 1 : length(speed_per_machine_month)  
            for j = 1 : length(cost_of_3dp_per_machine_month)
        
                for k = 1 : length(capacity_3dp_percentage)
                        
                    COST_SAVINGS_COMONO{pp,yy,i,j}(k) = (COST_TMONLY_COMONO{pp,yy} - OUTPUT_MEDIUM_BOE_COMONO{pp,yy, k}.TOTAL_COST_NONZERO(i,j)) / abs(COST_TMONLY_COMONO{pp,yy})*100;
                    
                end
        
            end
        end

    end
end






%% Plot Cost-Savings 
%% Always loop over all combos of "speed_per_machine_month" and "cost_of_3dp_per_machine_month"
nColors1 = length(yield_loss_rate_set);
nColors2 = length(p_set);
defaultColors = lines(7); % Default colors for up to 7 lines
if nColors1 <= 7
    colors1 = defaultColors; % Use default colors if nColors <= 7
else
    colors1 = [defaultColors; rand(nColors1 - 7, 3)]; % Default for first 7, random for the rest
end
if nColors2 <= 7
    colors2 = defaultColors; % Use default colors if nColors <= 7
else
    colors2 = [defaultColors; rand(nColors2 - 7, 3)]; % Default for first 7, random for the rest
end

for i = 1 : length(speed_per_machine_month)  
    for j = 1 : length(cost_of_3dp_per_machine_month)

        %% First, fix p, then plot different curves of yield_loss_rate in one plot
        for pp = 1 : length(p_set)

            for yy = 1 : length(yield_loss_rate_set)

                plot([0,capacity_3dp_percentage]*100, [0,COST_SAVINGS_COMONO{pp,yy,i,j}], '-o','LineWidth', 1, 'Color', colors1(yy,:))
                hold on

            end

            xlim([0,100])
            
            lgd = legend({'0.2x Baseline', '1x Baseline', '2x Baseline', '4x Baseline', '6x Baseline', '8x Baseline', '10x Baseline', '12x Baseline', '14x Baseline'}, ...
                'FontSize', 12, 'location', 'northeast');
            title(lgd, 'Yield Loss Rate');
          
            filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Comono_Varying_p_yield_loss(CostSavings)/Varying_yield_loss', '_fixed_p_',num2str(pp), '_C3DP_CASE', num2str(i), num2str(j), '.pdf');
            
            ax = gca;
            ax.XTickLabel = strcat(ax.XTickLabel, '%');
            ax.YTickLabel = strcat(ax.YTickLabel, '%');

            hold off
            saveas(gcf,filename)  % as MATLAB figure
            close(gcf)

        end

        %% Now, fix yield loss, then plot different curves of p in one plot
        for yy = 1 : length(yield_loss_rate_set)

            for pp = 1 : length(p_set)

                plot([0,capacity_3dp_percentage]*100, [0,COST_SAVINGS_COMONO{pp,yy,i,j}], '-o','LineWidth', 1, 'Color', colors2(pp,:))
                hold on

            end

            xlim([0,100])
            
            lgd = legend({'0.2x Baseline', '1x Baseline', '2x Baseline', '3x Baseline', '4x Baseline', '5x Baseline', '6x Baseline', '7x Baseline', '8x Baseline', '9x Baseline', '10x Baseline'}, ...
                'FontSize', 12, 'location', 'northeast');
            title(lgd, 'Marginal Disruption Rate');
          
            filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Comono_Varying_p_yield_loss(CostSavings)/Varying_p', '_fixed_yieldloss_',num2str(yy), '_C3DP_CASE', num2str(i), num2str(j), '.pdf');
            
            ax = gca;
            ax.XTickLabel = strcat(ax.XTickLabel, '%');
            ax.YTickLabel = strcat(ax.YTickLabel, '%');

            hold off
            saveas(gcf,filename)  % as MATLAB figure
            close(gcf)

        end


    end
end

%% Now for each combo of "speed_per_machine_month" and "cost_of_3dp_per_machine_month"
%% Compare the cost-savings under the followings:
%%      - Very Frequent and large scale:          p=0.5, yield_loss=0.7
%%      - Very Frequent and moderate scale:       p=0.5, yield_loss=0.4
%%      - Very Frequent and small scale:          p=0.5, yield_loss=0.01
%%      - Moderately Frequent and large scale:    p=0.3, yield_loss=0.7
%%      - Moderately Frequent and moderate scale: p=0.3, yield_loss=0.4
%%      - Moderately Frequent and small scale:    p=0.3, yield_loss=0.01
%%      - Rare and large scale:                   p=0.01, yield_loss=0.7
%%      - Rare and moderate scale:                p=0.01, yield_loss=0.4
%%      - Rare and small scale:                   p=0.01, yield_loss=0.01

% p_yield_combo = [7,3; 7,2; 7,1; 5,3; 5,2; 5,1; 1,3; 1,2; 1,1];
% p_yield_combo = [7,4; 7,1; 1,4; 1,1];
% p_yield_combo = [11,9; 11,6; 11,2; 5,9; 5,6; 5,2; 1,9; 1,6; 1,2];
p_yield_combo = [11,9; 11,1; 6,6; 1,9; 1,1];
% p_yield_combo = [7,6; 7,1; 5,4; 1,6; 1,1];


nColors3 = length(p_yield_combo);
defaultColors = lines(7); % Default colors for up to 7 lines
if nColors3 <= 7
    colors3 = defaultColors; % Use default colors if nColors <= 7
else
    colors3 = [defaultColors; rand(nColors3 - 7, 3)]; % Default for first 7, random for the rest
end

for combo = 1 : length(p_yield_combo)
    plot([combo,combo],[0,10], 'Color', colors3(combo,:),'LineWidth',2)
    hold on
end
xlim([-1,10])

for i = 1 : length(speed_per_machine_month)  
    for j = 1 : length(cost_of_3dp_per_machine_month)

        for combo = 1 : length(p_yield_combo)
        
            ttt = p_yield_combo(combo,:);
            pp = ttt(1); yy = ttt(2);

            plot([0,capacity_3dp_percentage]*100, [0,COST_SAVINGS_COMONO{pp,yy,i,j}], '-o','LineWidth', 1, 'Color', colors3(combo,:))
            hold on
        
        end

        xlim([0,100])
        % lgd = legend({'High-High', 'High-Mid', 'High-Low', 'Mid-High', 'Mid-Mid', 'Mid-Low', 'Low-High', 'Low-Mid', 'Low-Low'}, 'FontSize', 12, 'Location', 'eastoutside');
        lgd = legend({'High-High', 'High-Low', 'Mid-Mid', 'Low-High', 'Low-Low'}, 'FontSize', 12, 'location', 'southeast'); 
        title(lgd, '(Freq, Yield Loss)');

        filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Comono_Varying_p_yield_loss(CostSavings)/CHIGH_MID_LOW_C3DP_CASE', num2str(i), num2str(j), '.pdf');
        
        ax = gca;
        ax.XTickLabel = strcat(ax.XTickLabel, '%');
        ax.YTickLabel = strcat(ax.YTickLabel, '%');

        hold off
        saveas(gcf,filename)  % as MATLAB figure
        close(gcf)

    end
end

%% For each combo of "speed_per_machine_month" and "cost_of_3dp_per_machine_month", draw two plots
%%      - (Plot 1) for each p, draw the the following curve: let yield loss rate increases, for each yield loss rate, get the max. cost savings among K
%%      - (Plot 2) for each yield loss rate, draw the the following curve: let p increases, for each p, get the max. cost savings among K  

COST_SAVINGS_COMONO_MAX_AMONG_K = {};

for i = 1 : length(speed_per_machine_month)  
    for j = 1 : length(cost_of_3dp_per_machine_month)
        
        for yy = 1 : length(yield_loss_rate_set)
            for pp = 1 : length(p_set)

                COST_SAVINGS_COMONO_MAX_AMONG_K{i,j}(pp,yy) = max(COST_SAVINGS_COMONO{pp,yy,i,j});

            end
        end

    end
end




%% Take a step back, and there are several "hyper-parameters" in the space of modeling disruptions now:
%%      - independent or comonotonic 
%%      - marginal failure rate p
%%      - yield loss rate
%%      - K
%%      - C3DP
%% At this point, we already have a comprehensive showcase of p and yield loss, the focus here is on "independent" vs. "comonotonic"
%% Our strategy here is to 
%%      - fix C3DP
%%      - fix 5 sets of (p, yield_loss) pairs: "Hi-Hi", "Hi-LO", "LO-Hi", "Mid-Mid"
%%      - Compare the cost-savings vs. K curves of "independent" and "comononotonic"

p_yield_combo = [11,9; 11,3; 6,6; 1,9; 2,2];
% p_yield_combo = [11,1; 11,2; 11,3; 10,1; 10,2; 10,3; 9,1; 9,2; 9,3; 8,1; 8,2; 8,3;];

i=1; j=1;

subset = [1,11:length(capacity_3dp_percentage)];

for combo = 1 : length(p_yield_combo)

    ttt = p_yield_combo(combo,:);
    pp = ttt(1); yy = ttt(2);
    
    plot([0,capacity_3dp_percentage(subset)]*100, [0, COST_SAVINGS_VARYING_DISRUPTIONS{pp,yy,i,j}(subset)],  '-o','LineWidth', 2 )
    hold on
    plot([0,capacity_3dp_percentage(subset)]*100, [0, COST_SAVINGS_COMONO{pp,yy,i,j}(subset)] ,  '-^','LineWidth', 2 )

    ylim([-1,5])
    xlim([0,15])
    
    ylabel("Cost Savings (% of No 3DP)")
    xlabel("Capacity (% of Max. Demand)")

    lgd = legend({'Independent', 'Comonotonic'}, 'FontSize', 12, 'location', 'northwest'); 
    title(lgd, 'Disruption Correlations');

    filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Comono_Varying_p_yield_loss(CostSavings)/AAAAHIGH_MID_LOW_C3DP_CASE',...
        num2str(i), num2str(j), '_p_', num2str(pp), '_yieldloss_', num2str(yy), '.pdf');
    
    ax = gca;
    ax.XTickLabel = strcat(ax.XTickLabel, '%');
    ax.YTickLabel = strcat(ax.YTickLabel, '%');
    
    hold off
    saveas(gcf,filename)  % as MATLAB figure
    close(gcf)

end



%% save data for python
subset = [1, 11:length(capacity_3dp_percentage)];

Independent_CostSavings = {};
Comonotonic_CostSavings = {};
Capacities = [];

for combo = 1 : length(p_yield_combo)
    ttt = p_yield_combo(combo, :);
    pp = ttt(1); yy = ttt(2);
    
    % Store data
    Independent_CostSavings{combo} = [0, COST_SAVINGS_VARYING_DISRUPTIONS{pp, yy, i, j}(subset)];
    Comonotonic_CostSavings{combo} = [0, COST_SAVINGS_COMONO{pp, yy, i, j}(subset)];
    Capacities = [0, capacity_3dp_percentage(subset)] * 100; % Same for all combos
end

% Save data to .mat file
save('Experiment_Data/Relative_Cost_Savings_Shortfalls/comono_for_python_costsavings.mat', ...
     'Independent_CostSavings', 'Comonotonic_CostSavings', 'Capacities', 'p_yield_combo');







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BASED-LINE CASE: PLOT DEMAND SHORTFALLS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Becareful!!! In this case, even the SINGLE system is changing with p and yield loss

%% DUO system: After 3DP
%%      - For each capacity K on a grid 
%%      - Get the set of products backedup-by 3DP A
%%      - Sample pairs of demand and disruptions
%%      - For each sample (D, s), calculate optimal 3DP decision q3DP(D,s) and [D_j - q_j*s_j - q_j^3DP(D,s)]^+ and sum over j in A
%%      - Save the disribution as a vector

%% Two important notes:
%%      - For the same K, the deamnd shortfall is the same for all c_cap (q_SP is the same)
%%      - But different c_cap has different opitmla K hence different demand shortfall 

%% In what follows, we first fix a c_cap case ("11"), and then compute the demand shortall distribution for each of p and yield_loss case:
%%      - for each pair of p and yield_loss (under the fixed c_cap), we find the optimal K (max. cost savings)
%%      - retrieve the x and q_SP at the opt K, and comupute demand shortfall
%% (TRUST the cost and q_SP computed by SGD)

%% Compare Independent and Comonotonic case



i = 1; j = 2;

Relative_Demand_shortfall_comono = {};

for pp = 1 : length(p_set)
    for yy = 1 : length(yield_loss_rate_set)

        fprintf("Working with p case %d, yield loss case %d \n\n",   pp,    yy)
        
        %% A pair of p and yield loss
        p_medium = p_set(pp);
        yield_loss_rate_medium = yield_loss_rate_set(yy);
        


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% 3DP case (COMONOTONIC CASE since IND CASE already computed)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [~,kkk] = max(COST_SAVINGS_COMONO{pp,yy,i,j});
        x_3DP = logical(OUTPUT_MEDIUM_BOE_COMONO{pp,yy, kkk}.X_FINAL{i,j});
        q_SP = OUTPUT_MEDIUM_BOE_COMONO{pp,yy, kkk}.Q_FINAL{i,j}(x_3DP);  

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
            
            input_preprocess_medium_sampled.sample_mode = 3;
            if num_suppliers < 10
                input_preprocess_medium_sampled.demand_sample_flag = 0;
            else
                input_preprocess_medium_sampled.demand_sample_flag = 1;     % For each disruption scenario, sample a fixed number of demand combos
                Demand_sample_size_eval = ceil(1000*100/2);
                input_preprocess_medium_sampled.demand_samplesize = Demand_sample_size_eval;
            end
            
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
                
                Relative_Demand_shortfall_comono{pp,yy,i,j}(ss) = sum( max( D_bar - q_3DP, 0 ) )  / sum(max(Monthly_Quantity_3scenarios_all'))*100 ;
            
            end

        else

            %% When no 3DP is better (SINGLE SYSTEM)
            Relative_Demand_shortfall_comono{pp,yy,i,j}  = Relative_Demand_shortfall_no3DP_varying_disruptions{pp,yy};

        end

    
    end
end








%% The box plots we have so far are too raw and dense, let's select some subsets to plot
%% Fix p varying yield loss

DDD = load('Experiment_Data/Relative_Cost_Savings_Shortfalls/varying_disruption_distr_ind_for_python_shortfalls1.mat');

i=1;j=2;

p_subset1 = [2,6,11];
yield_loss_rate_subset1 = [1,2,4,6,8];

tick_labels1 = {'0.2x', '1x', '2x', '4x', '6x', '8x', '10x', '12x', '14x'};

for ppp = 1 : length(p_subset1)

    pp = p_subset1(ppp);

    for yyy = 1 : length(yield_loss_rate_subset1)

        yy = yield_loss_rate_subset1(yyy);
        
        boxplot(DDD.Box_plot_data11{1,ppp}(yyy,:), 'Positions', 4*yyy-3, 'Widths', 0.5, 'Symbol', '');
        hold on
        boxplot(Relative_Demand_shortfall_comono{pp,yy,i,j}, 'Positions', 4*yyy-2, 'Widths', 0.5, 'Symbol', '', 'Colors', 'r', 'MedianStyle', 'target');
        hold on
        
        % Mean_Relative_Demand_shotfall_varying_disruptions{i,j}(pp,yy) = mean(Relative_Demand_shotfall_varying_disruptions{pp,yy,i,j});
        Mean_Relative_Demand_shortfall_comono(pp,yy) = mean(Relative_Demand_shortfall_comono{pp,yy,i,j});
        
    end
    
    plot( 4*[1:length(yield_loss_rate_subset1)]-3, DDD.mean_plot_data11(ppp,:), '-o', 'LineWidth', 0.5, 'MarkerSize', 6, ...
        'MarkerFaceColor', [0, 0.4470, 0.7410])
    hold on
    plot( 4*[1:length(yield_loss_rate_subset1)]-2, Mean_Relative_Demand_shortfall_comono(pp,yield_loss_rate_subset1), '-o', 'LineWidth', 0.5, 'MarkerSize', 6, ...
        'MarkerFaceColor', [0.8500, 0.3250, 0.0980]) 

    xticks(4*[1:length(yield_loss_rate_subset1)]-2.5); 
    xticklabels(tick_labels1(yield_loss_rate_subset1));

    xline(4*[1:length(yield_loss_rate_subset1)]-0.5, 'Color', [0.5, 0.5, 0.5], 'LineStyle', '-', 'LineWidth', 0.5);

    xlabel('Yield Loss Ratio (Multiple of Baseline)', 'FontSize', 12);
    ylabel('Shortfall (% of Max Demand)', 'FontSize', 12);
    ytickformat('percentage');

    ylim([-0.333,16])
    xlim([-0.5, 4*length(yield_loss_rate_subset1)]-0.5)

    filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Comono_Varying_p_yield_loss(Shortfalls)/AAAAboxplots_C3DPcase_', num2str(i), num2str(j), 'fixed_p_varying_yieldloss_case', num2str(pp), '.pdf');
    saveas(gcf, filename);
    close(gcf)

end


%% Save data for python

DDD = load('Experiment_Data/Relative_Cost_Savings_Shortfalls/varying_disruption_distr_ind_for_python_shortfalls1.mat');

i=1;j=2;

% The data for box plots and mean plots
Box_plot_data11 = DDD.Box_plot_data11;
mean_plot_data11 = DDD.mean_plot_data11;
Box_plot_data12 = {};
mean_plot_data12 = [];

for ppp = 1 : length(p_subset1)

    pp = p_subset1(ppp);

    for yyy = 1 : length(yield_loss_rate_subset1)

        yy = yield_loss_rate_subset1(yyy);
        Box_plot_data12{1,ppp}(yyy,:) = Relative_Demand_shortfall_comono{pp,yy,i,j};
    end

    mean_plot_data12(ppp,:) = Mean_Relative_Demand_shortfall_comono(pp,yield_loss_rate_subset1);

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
ylimit1 = [-0.333,16];

% Save data to .mat file
save('Experiment_Data/Relative_Cost_Savings_Shortfalls/comono_for_python_shortfalls1.mat', ...
    'Box_plot_data11', 'Box_plot_data12', 'mean_plot_data11', 'mean_plot_data12', ...
     'box_plot_pos11', 'box_plot_pos12', 'x_ticks_labels1', 'x_ticks_pos1', 'vertline_pos1', 'xlimit1', 'ylimit1');








%% Fix yield loss varying p
DDD = load('Experiment_Data/Relative_Cost_Savings_Shortfalls/varying_disruption_distr_ind_for_python_shortfalls2.mat');

yield_loss_rate_subset2 = [2,5,8];
p_subset2 = [1,2,4,6,8];

tick_labels2 = {'0.2x', '1x', '2x', '3x', '4x', '5x', '6x', '7x', '8x', '9x', '10x'};


for yyy = 1 : length(yield_loss_rate_subset2)

    yy = yield_loss_rate_subset2(yyy);

    for ppp = 1 : length(p_subset2)

        pp = p_subset2(ppp);

        boxplot(DDD.Box_plot_data21{1,yyy}(ppp,:), 'Positions', 4*ppp-3, 'Widths', 0.5, 'Symbol', '');
        hold on
        boxplot(Relative_Demand_shortfall_comono{pp,yy,i,j}, 'Positions', 4*ppp-2, 'Widths', 0.5, 'Symbol', '', 'Colors', 'r', 'MedianStyle', 'target');
        hold on

        Mean_Relative_Demand_shortfall_comono(pp,yy) = mean(Relative_Demand_shortfall_comono{pp,yy,i,j});
        
    end
    
    plot( 4*[1:length(p_subset2)]-3, DDD.mean_plot_data21(yyy,:), '-^', 'LineWidth', 0.5, 'MarkerSize', 6, ...
        'MarkerFaceColor', [0, 0.4470, 0.7410])
    hold on
    plot( 4*[1:length(p_subset2)]-2,  Mean_Relative_Demand_shortfall_comono(p_subset2,yy)', '-^', 'LineWidth', 0.5, 'MarkerSize', 6, ...
        'MarkerFaceColor', [0.8500, 0.3250, 0.0980]) 

    xticks(4*[1:length(p_subset2)]-2.5); 
    xticklabels(tick_labels2(p_subset2));

    xline(4*[1:length(p_subset2)]-0.5, 'Color', [0.5, 0.5, 0.5], 'LineStyle', '-', 'LineWidth', 0.5);

    xlabel('Marginal Failure Rate (Multiple of Baseline)', 'FontSize', 12);
    ylabel('Shortfall (% of Max Demand)', 'FontSize', 12);
    ytickformat('percentage');

    ylim([-0.333,16])
    xlim([-0.5, 4*length(p_subset2)]-0.5)
    % grid on;

    filename = strcat('Experiment_Data/Relative_Cost_Savings_Shortfalls/Comono_Varying_p_yield_loss(Shortfalls)/AAAboxplots_C3DPcase_', num2str(i), num2str(j), '_fixed_yieldloss_varying_p_case', num2str(yy), '.pdf');
    saveas(gcf, filename);
    close(gcf)

end



%% Save data for python
DDD = load('Experiment_Data/Relative_Cost_Savings_Shortfalls/varying_disruption_distr_ind_for_python_shortfalls2.mat');
i=1;j=2;

% The data for box plots and mean plots
Box_plot_data21 = DDD.Box_plot_data21;
mean_plot_data21 = DDD.mean_plot_data21;
Box_plot_data22 = {};
mean_plot_data22 = [];

for yyy = 1 : length(yield_loss_rate_subset2)

    yy = yield_loss_rate_subset2(yyy);

    for ppp = 1 : length(p_subset2)

        pp = p_subset2(ppp);
        Box_plot_data22{1,yyy}(ppp,:) = Relative_Demand_shortfall_comono{pp,yy,i,j};

    end

    mean_plot_data22(yyy,:) = Mean_Relative_Demand_shortfall_comono(p_subset2,yy)';

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
ylimit2 = [-0.333,16];

save('Experiment_Data/Relative_Cost_Savings_Shortfalls/comono_for_python_shortfalls2.mat', ...
    'Box_plot_data21', 'Box_plot_data22', ...
    'mean_plot_data21', 'mean_plot_data22', ...
    'box_plot_pos21', 'box_plot_pos22', ...
    'x_ticks_labels2', 'x_ticks_pos2', ...
    'vertline_pos2', 'xlimit2', 'ylimit2');







    