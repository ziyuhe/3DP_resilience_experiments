% =========================================================================
% Script Name:       Experiment_Basic_Plots_Mattel.m
% Date:              02/01/2025
% Description:       
%   - Generates CSV files for plotting basic statistics of the processed data.
%   - Includes histograms of unit weights and scatter plots of "Price per Unit" vs. "Weight per Unit".
%
% Notes:
%   - "Dedicated Backup" is labeled as "TM" (Traditional Manufacturing).
% - 3DP capacity represents the total weight of printing materials output per month.
% - Originally measured in grams, converted to kilograms to better scale cost in K.
% - Adjustments made:
%     - Divide "weight_all" by 1000 (per unit product weight).
%     - Divide "speed_per_machine_month" by 1000 (material consumption per printer per month).
%     - Divide "Monthly_Weight_3scenarios_all" by 1000 (demand scenarios in weight).
%
% =========================================================================



addpath('Utilities')




%% Read Data (General Information)
% Load supplier data containing product weights, costs, and pricing
filename = 'Problem_Data/All/Mattel_All_Suppliers_Ave_Weight_Quantity.csv';
all_csvdata = readtable(filename);

% Product weights (converted from grams to kg)
weight_all = all_csvdata.SyntheticProductWeight_gram_ / 1000;

% Cost parameters
c_source_all = all_csvdata.SyntheticSourcingCost;      % Primary sourcing cost
c_3DP_source_all = all_csvdata.Synthetic3DPCost;       % 3DP manufacturing cost
c_TM_source_all = all_csvdata.SyntheticExpeditionCost; % TM manufacturing cost
c_price_all = all_csvdata.SyntheticPrice;              % Selling price

% Compute cost differences for 3DP and TM
c_3DP_all = c_3DP_source_all - c_source_all; % Extra cost of 3DP vs. primary sourcing
c_TM_all = c_TM_source_all - c_source_all;   % Extra cost of TM vs. primary sourcing

% Define penalty and holding cost
v_all = c_price_all - c_source_all; % Lost-sale penalty (profit margin)
h_all = c_source_all;               % Holding cost (assume no salvage value)

% Number of suppliers
num_suppliers_all = length(h_all);

%% Read 3-Scenario Random Demand Data
% Load demand scenarios dataset
filename = 'Problem_Data/All/Mattel_All_Suppliers_Ave_Month_Weight_Quantity_3scenarios.csv';
all_csvdata_3scenarios = readtable(filename);
num_scenarios = 3;

% Generate column names dynamically for different scenarios
weight_scenarios_col_names = arrayfun(@(k) strcat('WeightScenario', num2str(k), '_grams_'), 1:num_scenarios, 'UniformOutput', false);
quantity_scenarios_col_names = arrayfun(@(k) strcat('QuantityScenario', num2str(k)), 1:num_scenarios, 'UniformOutput', false);
probability_col_names = arrayfun(@(k) strcat('Scenario', num2str(k), 'Probability'), 1:num_scenarios, 'UniformOutput', false);

% Extract scenario data
Monthly_Weight_3scenarios_all = zeros(height(all_csvdata_3scenarios), num_scenarios);
Monthly_Quantity_3scenarios_all = zeros(height(all_csvdata_3scenarios), num_scenarios);
Demand_Probability_3scenarios_all = zeros(height(all_csvdata_3scenarios), num_scenarios);

for k = 1:num_scenarios
    Monthly_Weight_3scenarios_all(:,k) = all_csvdata_3scenarios.(weight_scenarios_col_names{k});
    Monthly_Quantity_3scenarios_all(:,k) = all_csvdata_3scenarios.(quantity_scenarios_col_names{k});
    Demand_Probability_3scenarios_all(:,k) = all_csvdata_3scenarios.(probability_col_names{k});
end

% Convert weight from grams to kg
Monthly_Weight_3scenarios_all = Monthly_Weight_3scenarios_all / 1000;

% Renormalize probabilities to account for numerical precision issues
Demand_Probability_3scenarios_all = Demand_Probability_3scenarios_all ./ sum(Demand_Probability_3scenarios_all, 2);

% Compute mean demand across scenarios
mean_demand_3scenarios_all = sum(Monthly_Quantity_3scenarios_all .* Demand_Probability_3scenarios_all, 2);








%% Save Data to CSV for Python Plot
%% Histogram for product weights (per unit)
histogram(weight_all * 1000, 'BinWidth', 100)
xlabel('Unit Weight (g)', 'FontSize', 12)
grid on

% Prepare data for histogram
weight_data = weight_all * 1000; % Convert to grams

% Define the directory path
output_dir = 'Experiment_Results/Basic_Pictures_Synthetic_Products';

% Check if the directory exists, if not, create it
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Write the histogram data to CSV
histogram_filename = fullfile(output_dir, 'UnitWeightHistogramData.csv');
writetable(table(weight_data, 'VariableNames', {'UnitWeight_g'}), histogram_filename);


%% Scatter plot on "Price per unit" vs. "Weight per unit" among all products
% Prepare data for CSV
average_weight_per_month = mean(Monthly_Weight_3scenarios_all, 2) * 1000; % in grams
average_demand_per_month = mean_demand_3scenarios_all; % in units
unit_price = c_price_all; % unit price
marker_size = weight_all * 333; % marker size

% Combine data into a table
data_table = table(average_weight_per_month, average_demand_per_month, unit_price, marker_size, ...
    'VariableNames', {'AverageWeightPerMonth_g', 'AverageDemandPerMonth_unit', 'UnitPrice', 'MarkerSize'});

% Write the scatter plot data to CSV
scatter_filename = fullfile(output_dir, 'Synthesized_Product_unitweight_unitprice_demand.csv');
writetable(data_table, scatter_filename);
