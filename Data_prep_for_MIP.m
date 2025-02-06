function output = Data_prep_for_MIP(input)

% =========================================================================
% Script Name:       Data_prep_for_MIP.m
% Author:            Ziyu He
% Date:              02/01/2025
% Description:       
%       - We sample disruptions and demand data 
%
%% NOTE: WE ALWAYS ASSUME DEMAND AND DISRUPTIONS ARE INDEPENDENT!!!
%
%% In this function, we preprocess the data needed for solving the supplier selection through the MIP solver
% There are several modes in terms of sampling of disruption and demand scenarios
%% "sample_mode == 1" (INDEPENDENT DISRUPTIONS)
%   - we can choose not to sample disruption or demand and just keep all the combos
%       - "disruption_sample_flag == 1": sample disruption
%       - "demand_sample_flag == 1": sample demand
%   - at the end, we combine all the disruption and demand combinations together
%       - each disruption combination is repeated "number of demand combintations" times
%       - to pair up with all the demand combinations
%   - this mode allow us to maintain the most complete set of samples
%% "sample_mode == 2" (INDEPENDENT DISRUPTIONS)
%  - we sample disruption combinations or keep all of the combinations
%       - "disruption_sample_flag == 1": sample disruption
%  - for each disruption combnation we sample a number of demand combinations
%  - this mode allow different sampled disruption combinations to have different demand combinations
%% "sample_mode == 3" (COMONOTONIC)
%  - now the disruption only has two scenarios: all ones or all 1-yield_loss_rate
%  - we can choose to keep all demand scenarios (if dimension is small) or sample demand
%  - for both disruption scenarios, we use the same set of demand samples
%% "sample_mode == 4" (AT-MOST-ONE-DISRUPTED)
%  - now the disruption has n+1 scenarios: 
%       - n vector of all ones except for i-th item being "1-yield_loss_rate", where i=1...n 
%       - a vector of all ones
%  - we can choose to keep all demand scenarios (if dimension is small) or sample demand
%  - for both disruption scenarios, we use the same set of demand samples
%% "sample_mode == 5" (INTERPOLATE CORRELATION)
%  - Input: p0, p
%       - simulate binary variables X_j, j=1...n each independent with success rate p
%       - simulate one binary variable Y, with success rate p0
%       - let Z_j = max{ X_j, Y }
%  - Here we assume the disruptions are sampeld as Z_j 
%  - and the combined with demand in the same fashion as "sample_mode == 2"
%  - i.e., each disruption combo has several demand combinations
% FOR SIMPLICITY WE ASSUME DISRUPTIONS AND DEMAND ARE BOTH SAMPLED
% =========================================================================






Monthly_Quantity = input.Monthly_Quantity;
Demand_Probability = input.Demand_Probability;
num_scenarios = input.num_scenarios;

%% The difference between "sample_mode == 1" and "sample_mode == 2" is that the latter mode allow different demand combos across disruption scenarios

sample_mode = input.sample_mode;

if sample_mode == 1

    %% Sample Mode 1

    % "failure_combinations"
    %   - the scenario combinations for disruptions (sampled if disruption_sample_flag == 1)
    %   - each column is a disruption scenario
    % "disruption_prob"
    %   - the associated probabilities (1/S if disruption_sample_flag == 1)
    
    num_suppliers = input.num_suppliers;        % number of suppliers in the scope
    yield_loss_rate = input.yield_loss_rate;    % yield loss rate (we only consider either full delivery or a portion of delivery) 
    p_disrupt = input.p_disrupt;                % probability of disruption
    
    if input.disruption_sample_flag == 1
        
        % disruption_samplesize = input.disruption_samplesize;
        % failure_combinations = randi([0, 1], num_suppliers, disruption_samplesize);
        % failure_combinations(failure_combinations==0) = 1 - yield_loss_rate;
        % disruption_prob = (1/disruption_samplesize)*ones(disruption_samplesize, 1);

        disruption_samplesize = input.disruption_samplesize;
        failure_combinations = [];
        disruption_scenarios = [1-yield_loss_rate, 1];
        for i = 1 : num_suppliers
            failure_combinations(i,:) = disruption_scenarios(randsample(2,disruption_samplesize,true,[p_disrupt,1-p_disrupt]));
        end
        disruption_prob = (1/disruption_samplesize)*ones(disruption_samplesize, 1);

    
    else
    
        failure_combinations_og = dec2bin(0:(2^num_suppliers - 1)) - '0'; % "1"==success, "0"==failure
        
        failure_combinations = failure_combinations_og';
        failure_combinations(failure_combinations==0) = 1 - yield_loss_rate;
        
        q_disrupt = 1 - p_disrupt; 
        disruption_prob = zeros(2^num_suppliers, 1);
        for i = 1 : 2^num_suppliers
            scenario = failure_combinations_og(i, :);
            num_failures = sum(scenario == 0);
            num_successes = num_suppliers - num_failures;
            disruption_prob(i) = p_disrupt^num_failures * q_disrupt^num_successes;
        end
    
    end
    
    output.failure_combinations = failure_combinations;
    output.disruption_prob = disruption_prob;
    
    
    % "demand_atoms_combinations" 
    %   - the scenario combinations for disruptions (sampled if demand_sample_flag == 1)
    %   - each column is a specific combination of demand scenarios
    % "demand_combo_probabilities"
    %   - the associated probabilities
    
    if input.demand_sample_flag == 1

        % demand_samplesize = input.demand_samplesize;
        % demand_idx_combinations = randi(num_scenarios, num_suppliers, demand_samplesize);
        % demand_atoms_combinations = Monthly_Quantity(sub2ind(size(Monthly_Quantity), repmat((1:num_suppliers)', 1, demand_samplesize), demand_idx_combinations));
        % demand_combo_probabilities = (1/demand_samplesize)*ones(demand_samplesize,1);

        demand_samplesize = input.demand_samplesize;
        demand_atoms_combinations = [];
        for i = 1 : num_suppliers
            demand_atoms_combinations(i,:) = Monthly_Quantity(i, randsample(num_scenarios, demand_samplesize, true, Demand_Probability(i,:)));
        end
        demand_combo_probabilities = (1/demand_samplesize)*ones(demand_samplesize,1);

    else

        Grid = {};
        [Grid{1:num_suppliers}] = ndgrid(1:num_scenarios);
        combinations_indices = cell2mat(cellfun(@(x) x(:), Grid, 'UniformOutput', false));
        
        demand_atoms_combinations = zeros(size(combinations_indices, 1), num_suppliers);
        for i = 1:num_suppliers
            demand_atoms_combinations(:, i) = Monthly_Quantity(i, combinations_indices(:, i));
        end
        demand_atoms_combinations = demand_atoms_combinations';
        
        % The corresponding joint probabilities of the combinations of demand atoms
        demand_combo_probabilities = ones(size(combinations_indices, 1), 1);
        for i = 1:num_suppliers
            demand_combo_probabilities = demand_combo_probabilities .* Demand_Probability(i, combinations_indices(:, i))';
        end

    end

    output.demand_atoms_combinations = demand_atoms_combinations;
    output.demand_combo_probabilities = demand_combo_probabilities;

    % Combine all the disruption data and demand data together
    %   - Each disurption scenario is repeated "number of demand combination" times 
    %   - To pair up with all the demand combinations
    % "failure_data"
    %   - Each disruption combination is repeated "number of demand combination" times 
    % "demand_data"
    %   - The full matrix of demand combinations is repeated "number of disruption combination" times
    % "disruption_demand_joint_prob"
    %   - The joint probability of each combination of disruption and demand (as if they are independent)
    
    failure_data = [];
    for i = 1:size(failure_combinations,2)
        failure_data = [failure_data, repmat(failure_combinations(:,i), 1, size(demand_atoms_combinations,2))];
    end
    demand_data = repmat(demand_atoms_combinations, 1, size(failure_combinations,2));
    disruption_demand_joint_prob = reshape(demand_combo_probabilities*disruption_prob', [], 1);

    output.failure_data = failure_data;
    output.demand_data = demand_data;
    output.disruption_demand_joint_prob = disruption_demand_joint_prob;



elseif sample_mode == 2
    %% Sample Mode 2
    % Here the demand combinations are sampled
    % The disruption combinations are either sampled or kept with all combinations
    % Unlike sample mode 1 where each disruption combination is associated with the same set of demand combinations
    % Here, we sample different combinations for each disruption combination

    % "failure_combinations"
    %   - the scenario combinations for disruptions (sampled if disruption_sample_flag == 1)
    %   - each column is a disruption scenario
    % "disruption_prob"
    %   - the associated probabilities (1/S if disruption_sample_flag == 1)
    
    num_suppliers = input.num_suppliers;        % number of suppliers in the scope
    yield_loss_rate = input.yield_loss_rate;    % yield loss rate (we only consider either full delivery or a portion of delivery) 
    p_disrupt = input.p_disrupt;                % probability of disruption
    
    if input.disruption_sample_flag == 1
        
        % disruption_samplesize = input.disruption_samplesize;
        % failure_combinations = randi([0, 1], num_suppliers, disruption_samplesize);
        % failure_combinations(failure_combinations==0) = 1 - yield_loss_rate;
        % disruption_prob = (1/disruption_samplesize)*ones(disruption_samplesize, 1);

        disruption_samplesize = input.disruption_samplesize;
        failure_combinations = [];
        disruption_scenarios = [1-yield_loss_rate, 1];
        for i = 1 : num_suppliers
            failure_combinations(i,:) = disruption_scenarios(randsample(2,disruption_samplesize,true,[p_disrupt,1-p_disrupt]));
        end
        disruption_prob = (1/disruption_samplesize)*ones(disruption_samplesize, 1);
    
    else
    
        failure_combinations_og = dec2bin(0:(2^num_suppliers - 1)) - '0'; % "1"==success, "0"==failure
        
        failure_combinations = failure_combinations_og';
        failure_combinations(failure_combinations==0) = 1 - yield_loss_rate;
        
        q_disrupt = 1 - p_disrupt; 
        disruption_prob = zeros(2^num_suppliers, 1);
        for i = 1 : 2^num_suppliers
            scenario = failure_combinations_og(i, :);
            num_failures = sum(scenario == 0);
            num_successes = num_suppliers - num_failures;
            disruption_prob(i) = p_disrupt^num_failures * q_disrupt^num_successes;
        end
    
    end
    
    output.failure_combinations = failure_combinations;
    output.disruption_prob = disruption_prob;

    

    % For each disruption scenario, we sample "demand_num_per_disruption" combinations of demand
    demand_num_per_disruption = input.demand_num_per_disruption;
    Monthly_Quantity = input.Monthly_Quantity;
    num_scenarios = input.num_scenarios;
    
    % Each disruption combination is repeated "demand_num_per_disruption" times
    failure_data = [];
    for i = 1:size(failure_combinations,2)
        failure_data = [failure_data, repmat(failure_combinations(:,i), 1, demand_num_per_disruption)];
    end

    % We sample "total_samplesize" of demand combinations, assign "demand_num_per_disruption" to each diruption combination
    total_samplesize = demand_num_per_disruption*size(failure_combinations,2);
    % demand_idx_combinations = randi(num_scenarios, num_suppliers, total_samplesize);
    % demand_data = Monthly_Quantity(sub2ind(size(Monthly_Quantity), repmat((1:num_suppliers)', 1, total_samplesize), demand_idx_combinations));
    demand_data = [];
    for i = 1 : num_suppliers
        demand_data(i,:) = Monthly_Quantity(i, randsample(num_scenarios, total_samplesize, true, Demand_Probability(i,:)));
    end

    demand_combo_probabilities = (1/demand_num_per_disruption)*ones(demand_num_per_disruption,1);
    disruption_demand_joint_prob = reshape(demand_combo_probabilities*disruption_prob', [], 1);

    output.failure_data = failure_data;
    output.demand_data = demand_data;
    output.disruption_demand_joint_prob = disruption_demand_joint_prob;


elseif sample_mode == 3

    %% "sample_mode == 3" (COMONOTONIC)
    %  - now the disruption only has two scenarios: all ones or all zeros
    %  - we can choose to keep all demand scenarios (if dimension is small) or sample demand
    %  - for both disruption scenarios, we use the same set of demand samples

    num_suppliers = input.num_suppliers;        % number of suppliers in the scope
    yield_loss_rate = input.yield_loss_rate;    % yield loss rate (we only consider either full delivery or a portion of delivery) 
    p_disrupt = input.p_disrupt;                % probability of disruption
    

    if ~isfield(input, 'disruption_sample_flag')
        input.disruption_sample_flag = 0;
    end
    
    if input.disruption_sample_flag == 1

        disruption_samplesize = input.disruption_samplesize;

        tmp = double(rand(1, disruption_samplesize) < 1-p_disrupt);
        tmp(tmp==0) = 1 - yield_loss_rate;
        failure_combinations = tmp.*ones(num_suppliers, disruption_samplesize);
        disruption_prob = (1/disruption_samplesize)*ones(disruption_samplesize, 1);

        output.failure_combinations = failure_combinations;
        output.disruption_prob = disruption_prob;

    else

        failure_combinations = [ones(num_suppliers,1), (1-yield_loss_rate)*ones(num_suppliers,1)];
        disruption_prob = [1-p_disrupt; p_disrupt];
    
        output.failure_combinations = failure_combinations;
        output.disruption_prob = disruption_prob;

    end
    
    
    if input.demand_sample_flag == 1

        demand_samplesize = input.demand_samplesize;
        demand_atoms_combinations = [];
        for i = 1 : num_suppliers
            demand_atoms_combinations(i,:) = Monthly_Quantity(i, randsample(num_scenarios, demand_samplesize, true, Demand_Probability(i,:)));
        end
        demand_combo_probabilities = (1/demand_samplesize)*ones(demand_samplesize,1);

    else

        Grid = {};
        [Grid{1:num_suppliers}] = ndgrid(1:num_scenarios);
        combinations_indices = cell2mat(cellfun(@(x) x(:), Grid, 'UniformOutput', false));
        
        demand_atoms_combinations = zeros(size(combinations_indices, 1), num_suppliers);
        for i = 1:num_suppliers
            demand_atoms_combinations(:, i) = Monthly_Quantity(i, combinations_indices(:, i));
        end
        demand_atoms_combinations = demand_atoms_combinations';
        
        % The corresponding joint probabilities of the combinations of demand atoms
        demand_combo_probabilities = ones(size(combinations_indices, 1), 1);
        for i = 1:num_suppliers
            demand_combo_probabilities = demand_combo_probabilities .* Demand_Probability(i, combinations_indices(:, i))';
        end

    end

    output.demand_atoms_combinations = demand_atoms_combinations;
    output.demand_combo_probabilities = demand_combo_probabilities;

    % Combine all the disruption data and demand data together
    %   - Each disurption scenario is repeated "number of demand combination" times 
    %   - To pair up with all the demand combinations
    % "failure_data"
    %   - Each disruption combination is repeated "number of demand combination" times 
    % "demand_data"
    %   - The full matrix of demand combinations is repeated "number of disruption combination" times
    % "disruption_demand_joint_prob"
    %   - The joint probability of each combination of disruption and demand (as if they are independent)
    
    failure_data = [];
    for i = 1:size(failure_combinations,2)
        failure_data = [failure_data, repmat(failure_combinations(:,i), 1, size(demand_atoms_combinations,2))];
    end
    demand_data = repmat(demand_atoms_combinations, 1, size(failure_combinations,2));
    disruption_demand_joint_prob = reshape(demand_combo_probabilities*disruption_prob', [], 1);

    output.failure_data = failure_data;
    output.demand_data = demand_data;
    output.disruption_demand_joint_prob = disruption_demand_joint_prob;



elseif input.sample_mode == 4
    
    %% "sample_mode == 4" (AT-MOST-ONE-DISRUPTED)
    %  - now the disruption has n+1 scenarios: 
    %       - n vector of all ones except for i-th item being "1-yield_loss_rate", where i=1...n 
    %       - a vector of all ones
    %  - we can choose to keep all demand scenarios (if dimension is small) or sample demand
    %  - for both disruption scenarios, we use the same set of demand samples

    num_suppliers = input.num_suppliers;        % number of suppliers in the scope
    yield_loss_rate = input.yield_loss_rate;    % yield loss rate (we only consider either full delivery or a portion of delivery) 
    p_disrupt = input.p_disrupt;                % probability of disruption

    %% The marginal disruption rate has to be smaller than 1/n
    if p_disrupt > 1/num_suppliers
        
        display("The marginal probability > 1/n, this can't happen if at-most-one-disrupted!!!")

    else

        failure_combinations = [ones(num_suppliers,1), (1-yield_loss_rate)*eye(num_suppliers)+(1-eye(num_suppliers)) ];
        disruption_prob = [1-num_suppliers*p_disrupt; p_disrupt*ones(num_suppliers,1)];
    
        output.failure_combinations = failure_combinations;
        output.disruption_prob = disruption_prob;
        
        
        if input.demand_sample_flag == 1
    
            demand_samplesize = input.demand_samplesize;
            demand_atoms_combinations = [];
            for i = 1 : num_suppliers
                demand_atoms_combinations(i,:) = Monthly_Quantity(i, randsample(num_scenarios, demand_samplesize, true, Demand_Probability(i,:)));
            end
            demand_combo_probabilities = (1/demand_samplesize)*ones(demand_samplesize,1);
    
        else
    
            Grid = {};
            [Grid{1:num_suppliers}] = ndgrid(1:num_scenarios);
            combinations_indices = cell2mat(cellfun(@(x) x(:), Grid, 'UniformOutput', false));
            
            demand_atoms_combinations = zeros(size(combinations_indices, 1), num_suppliers);
            for i = 1:num_suppliers
                demand_atoms_combinations(:, i) = Monthly_Quantity(i, combinations_indices(:, i));
            end
            demand_atoms_combinations = demand_atoms_combinations';
            
            % The corresponding joint probabilities of the combinations of demand atoms
            demand_combo_probabilities = ones(size(combinations_indices, 1), 1);
            for i = 1:num_suppliers
                demand_combo_probabilities = demand_combo_probabilities .* Demand_Probability(i, combinations_indices(:, i))';
            end
    
        end
    
        output.demand_atoms_combinations = demand_atoms_combinations;
        output.demand_combo_probabilities = demand_combo_probabilities;
    
        % Combine all the disruption data and demand data together
        %   - Each disurption scenario is repeated "number of demand combination" times 
        %   - To pair up with all the demand combinations
        % "failure_data"
        %   - Each disruption combination is repeated "number of demand combination" times 
        % "demand_data"
        %   - The full matrix of demand combinations is repeated "number of disruption combination" times
        % "disruption_demand_joint_prob"
        %   - The joint probability of each combination of disruption and demand (as if they are independent)
        
        failure_data = [];
        for i = 1:size(failure_combinations,2)
            failure_data = [failure_data, repmat(failure_combinations(:,i), 1, size(demand_atoms_combinations,2))];
        end
        demand_data = repmat(demand_atoms_combinations, 1, size(failure_combinations,2));
        disruption_demand_joint_prob = reshape(demand_combo_probabilities*disruption_prob', [], 1);
    
        output.failure_data = failure_data;
        output.demand_data = demand_data;
        output.disruption_demand_joint_prob = disruption_demand_joint_prob;


    end

elseif input.sample_mode == 5

    %% Sample Mode 5
    % Here the demand combinations are sampled
    % The disruption combinations are either sampled or kept with all combinations
    % Unlike sample mode 1 where each disruption combination is associated with the same set of demand combinations
    % Here, we sample different combinations for each disruption combination

    % "failure_combinations"
    %   - the scenario combinations for disruptions (sampled if disruption_sample_flag == 1)
    %   - each column is a disruption scenario
    % "disruption_prob"
    %   - the associated probabilities (1/S if disruption_sample_flag == 1)
    
    num_suppliers = input.num_suppliers;        % number of suppliers in the scope
    yield_loss_rate = input.yield_loss_rate;    % yield loss rate (we only consider either full delivery or a portion of delivery) 
    p_disrupt = input.p_disrupt;                % probability of disruption
    p0 = input.p0;
    
        
    % disruption_samplesize = input.disruption_samplesize;
    % failure_combinations = randi([0, 1], num_suppliers, disruption_samplesize);
    % failure_combinations(failure_combinations==0) = 1 - yield_loss_rate;
    % disruption_prob = (1/disruption_samplesize)*ones(disruption_samplesize, 1);
    

    if input.disruption_sample_flag == 1      

        disruption_samplesize = input.disruption_samplesize;
        disruption_scenarios = [1-yield_loss_rate, 1];
        
        % Comonotonic part (binary failure indicator)
        comonotonic_part = rand(1, disruption_samplesize) < p0;
    
        % Combine wiht independent part
        failure_combinations_binary = [];
        failure_combinations = [];
        pp = (p_disrupt - p0)/(1 - p0);
        for i = 1 : num_suppliers
            failure_combinations_binary(i,:) = 1 - max( comonotonic_part,  (rand(1, disruption_samplesize)<pp) );
            failure_combinations(i,:) = disruption_scenarios( failure_combinations_binary(i,:) + 1 );
        end
    
        disruption_prob = (1/disruption_samplesize)*ones(disruption_samplesize, 1);

    else

        failure_combinations_og = dec2bin(0:(2^num_suppliers - 1)) - '0'; % "1"==success, "0"==failure
        
        failure_combinations = failure_combinations_og';
        failure_combinations(failure_combinations==0) = 1 - yield_loss_rate;
        
        pp = (p_disrupt - p0)/(1 - p0);
        qq = 1 - pp; 
        disruption_prob = zeros(2^num_suppliers, 1);
        for i = 1 : 2^num_suppliers
            scenario = failure_combinations_og(i, :);
            num_failures = sum(scenario == 0);
            num_successes = num_suppliers - num_failures;
            if num_successes > 0
                disruption_prob(i) = pp^num_failures * qq^num_successes * (1-p0);
            else
                disruption_prob(i) = p0 + (1-p0)*pp^num_failures * qq^num_successes;
            end
        end
        
    end

    output.failure_combinations = failure_combinations;
    output.disruption_prob = disruption_prob;

    
    if input.demand_sample_flag == 1

        % For each disruption scenario, we sample "demand_num_per_disruption" combinations of demand
        demand_num_per_disruption = input.demand_num_per_disruption;
        Monthly_Quantity = input.Monthly_Quantity;
        num_scenarios = input.num_scenarios;
        
        % Each disruption combination is repeated "demand_num_per_disruption" times
        failure_data = [];
        for i = 1:size(failure_combinations,2)
            failure_data = [failure_data, repmat(failure_combinations(:,i), 1, demand_num_per_disruption)];
        end
    
        % We sample "total_samplesize" of demand combinations, assign "demand_num_per_disruption" to each diruption combination
        total_samplesize = demand_num_per_disruption*size(failure_combinations,2);
        % demand_idx_combinations = randi(num_scenarios, num_suppliers, total_samplesize);
        % demand_data = Monthly_Quantity(sub2ind(size(Monthly_Quantity), repmat((1:num_suppliers)', 1, total_samplesize), demand_idx_combinations));
        demand_data = [];
        for i = 1 : num_suppliers
            demand_data(i,:) = Monthly_Quantity(i, randsample(num_scenarios, total_samplesize, true, Demand_Probability(i,:)));
        end
    
        demand_combo_probabilities = (1/demand_num_per_disruption)*ones(demand_num_per_disruption,1);
        disruption_demand_joint_prob = reshape(demand_combo_probabilities*disruption_prob', [], 1);
    
        output.failure_data = failure_data;
        output.demand_data = demand_data;
        output.disruption_demand_joint_prob = disruption_demand_joint_prob;

    else

        Grid = {};
        [Grid{1:num_suppliers}] = ndgrid(1:num_scenarios);
        combinations_indices = cell2mat(cellfun(@(x) x(:), Grid, 'UniformOutput', false));

        demand_atoms_combinations = zeros(size(combinations_indices, 1), num_suppliers);
        for i = 1:num_suppliers
            demand_atoms_combinations(:, i) = Monthly_Quantity(i, combinations_indices(:, i));
        end
        demand_atoms_combinations = demand_atoms_combinations';

        % The corresponding joint probabilities of the combinations of demand atoms
        demand_combo_probabilities = ones(size(combinations_indices, 1), 1);
        for i = 1:num_suppliers
            demand_combo_probabilities = demand_combo_probabilities .* Demand_Probability(i, combinations_indices(:, i))';
        end


        output.demand_atoms_combinations = demand_atoms_combinations;
        output.demand_combo_probabilities = demand_combo_probabilities;
        
        failure_data = [];
        for i = 1:size(failure_combinations,2)
            failure_data = [failure_data, repmat(failure_combinations(:,i), 1, size(demand_atoms_combinations,2))];
        end
        demand_data = repmat(demand_atoms_combinations, 1, size(failure_combinations,2));
        disruption_demand_joint_prob = reshape(demand_combo_probabilities*disruption_prob', [], 1);

        output.failure_data = failure_data;
        output.demand_data = demand_data;
        output.disruption_demand_joint_prob = disruption_demand_joint_prob;


    end


end


% All the demand data (each repeated 2^num_suppliers times)
% tt = size(demand_atoms_combinations, 2);
% demand_data = [];
% for i = 1:tt
%     demand_data = [demand_data, repmat(demand_atoms_combinations(:,i), 1, 2^n)];
% end


end


