function output = U3DP_SGD_fixed_suppselect_fixed_K(input)


% =========================================================================
% Script Name:       U3DP_SGD_fixed_suppselect_fixed_K.m
% Date:              02/01/2025
% Description:       
%   - This script applies **Stochastic Gradient Descent (SGD)** to solve the **U3DP problem** 
%     under predefined configurations.
%   - The U3DP problem involves optimizing 3DP backup capacity allocation while 
%     accounting for uncertain disruptions and demand variations.
%   - The experiment settings include:
%       - A **sales-oriented** model.
%       - **Fixed supplier selection** ("x").
%       - **Fixed 3DP capacity** ("K")
%       - Optimization **q** (first-stage order quantity).
%
% Disruption Modeling: Independent Disruptions
%
%% The sampling scheme here:
%  We can choose to sample everything ahead:
%      - input.sample_ahead == 1: we sample ahead
%      - input.sample_ahead == 2: we sample one-at-a-time
%  We can choose to only sample demand:
%      - input.disrupt_sample_flag == 1: we also sample disruptions
%      - input.disrupt_sample_flag == 2: we don't sample disruptions and use all disruptions scenarios (only recommended for non-inedpendent cases)
%  When input.disrupt_sample_flag == 2:   
%       - Disruption scenarios are contained in "input.failure_combinations"
%       - Corresponding probabilities are contained in "input.disruption_prob"
%
% Scope and Exclusions:
%   - The following components are **not** included in this experiment’s objective function:
%       - **Fixed 3DP cost (C_3DP)**.
%       - **Sales margin term** (`-v' * mean_demand`).
% =========================================================================



%% NOTE: THIS ONLY SERVES THE CASE WHEN DISRUPTIONS AND DEMANDS ARE ALL INDEPENDENT!!!
%% FOR THE CASE WHERE WE ALLOW OTHER DISTRIBUTIONS OF DISRUPTIONS, SEE "U3DP_SGD_fixed_suppselect_fixed_K_alternative"

%% We apply SGD to solve the U3DP problem under the following configurations
%  - "sales-oriented" model
%  - fixed supplier selection ("x")
%  - fixed 3DP capacity K_3DP

%% NOTE: In the scope of the problem we don't count:
% - C_3DP
% - -v'*mean_demand


%% Some basic parameters of the problem
n = input.n; % number of products
c_3DP = input.c_3DP; 
v = input.v;
h = input.h;
weight = input.weight; % per unit weight of printing materials for each product
K_3DP = input.K_3DP;

mean_demand = input.mean_demand;

C_3DP = input.C_3DP;

q_current = input.q_init; % Initialization


%% Store the intermediate results: q, subgradient, obj val
Q = [];           % n-by-t matrix to store the solutions
SUBGRAD = [];     % n-by-t matrix to store the subgradients
OBJEVAL = [];     % We store the (empirical) objective value every "objeval_steps" steps


%% Things needed for the intermediate samplings
yield_loss_rate = input.yield_loss_rate;
p_disrupt = input.p_disrupt;
disruption_scenarios = [1-yield_loss_rate, 1];

Monthly_Quantity = input.Monthly_Quantity;
Demand_Probability = input.Demand_Probability;
num_scenarios = input.num_scenarios;

s_SAMPLES = []; % n-by-t matrix to store the disruption sampled
D_SAMPLES = []; % n-by-t matrix to store the demand sampled

Max_Steps = input.Max_Steps;

%% We have two different sampling settings: one-at-a-time or sample ahead
%   - Sampling ahead is in general better, since sampling a batch of data at once in MATLAB is faster than doing it one at a time
%   - However, if we run a large number of steps, sampling everything ahead is a big challenge for memeory
%   - As a comprimise, we sample "input.sample_ahead_batchsize" amount of data once per "sample_ahead_batchsize" steps
if input.sample_ahead == 1 
    for j = 1:n
        s_SAMPLES(j,:) = disruption_scenarios(randsample(2, input.sample_ahead_batchsize, true, [p_disrupt,1-p_disrupt]));
    end
    for j = 1:n
        D_SAMPLES(j,:) = Monthly_Quantity(j, randsample(num_scenarios, input.sample_ahead_batchsize, true, Demand_Probability(j,:)));
    end
end
sample_ahead_count = 1;


%% Things needed for subgradient evaluation
input_subgrad.num_suppliers = n;
input_subgrad.c_3DP = c_3DP;
input_subgrad.v = v;
input_subgrad.h = h;
input_subgrad.weight = weight;
input_subgrad.mean_demand = mean_demand;
input_subgrad.K_3DP = K_3DP;
input_subgrad.obj_only = 0;



%% To evaluate U3DP (either precise or SAA) we need the followings:
S = input.S;                           % number of samples
prob_scenarios = input.prob_scenarios; % probability vector of all combinations of disruption and demand scenarios 
D_scenarios = input.D_scenarios;       % all demand scenarios (n by S)
s_scenarios = input.s_scenarios;       % all disruption scenarios (n by S)

input_objeval = input_subgrad;
input_objeval.obj_only = 1;
objeval_count = 1;


if input.stepsize_flag == 0
    stepsize = input.stepsize_const*ones(Max_Steps,1);
else
    stepsize = input.stepsize_const*1./sqrt([1:Max_Steps]');
end

startTime = clock;
time_spent = 0;



for t = 1 : Max_Steps

    if mod(t,input.display_interval) == 1
        fprintf("Running SGD step: %3.2f%% out of %d Steps\n",t/Max_Steps*100, Max_Steps);
        endTime = clock;
        time_spent = time_spent + etime(endTime, startTime);  
        startTime = clock;
    end
    
    if input.disrupt_sample_flag == 1

        %% Sample a pair of disruption and demand
        if input.sample_ahead == 1 

            if sample_ahead_count < input.sample_ahead_batchsize
                s_current = s_SAMPLES(:,sample_ahead_count);
                D_current = D_SAMPLES(:,sample_ahead_count);
                sample_ahead_count = sample_ahead_count + 1;
            else
                for j = 1:n
                    s_SAMPLES(j,:) = disruption_scenarios(randsample(2, input.sample_ahead_batchsize, true, [p_disrupt,1-p_disrupt]));
                end
                for j = 1:n
                    D_SAMPLES(j,:) = Monthly_Quantity(j, randsample(num_scenarios, input.sample_ahead_batchsize, true, Demand_Probability(j,:)));
                end
                sample_ahead_count = 1;
            end
            
        else
            
            % If we haven't sampled ahead
            s_current = [];
            for j = 1:n
                s_current(j,:) = disruption_scenarios(randsample(2,1,true,[p_disrupt,1-p_disrupt]));
            end
            D_current = [];
            for j = 1:n
                D_current(j,:) = Monthly_Quantity(j, randsample(num_scenarios, 1, true, Demand_Probability(j,:)));
            end
            
            s_SAMPLES(:,t) = s_current;
            D_SAMPLES(:,t) = D_current;

        end
        
        %% Compute the subgradient of V3DP under the current q, s, D ("b2b" + KKT)
        input_subgrad.q = q_current;
        input_subgrad.s = s_current;
        input_subgrad.D = D_current;
        input_subgrad.D_bar = D_current - s_current.*q_current;
        output_subgrad = V3DP_b2b_dual_fixed_suppselect(input_subgrad);
    
        subgrad = output_subgrad.beta;

    else

        %% Only sample demand and keep all the disruption scenarios
        D_current = [];
        for j = 1:n
            D_current(j,:) = Monthly_Quantity(j, randsample(num_scenarios, 1, true, Demand_Probability(j,:)));
        end       
        D_SAMPLES(:,t) = D_current;

        failure_combination = input.failure_combinations;
        disruption_prob = input.disruption_prob;
        
        input_subgrad.q = q_current;
        input_subgrad.D = D_current;

        SUBGRAD_ALL_DISRUPT = [];

        for i = 1 : length(disruption_prob)

            input_subgrad.s = failure_combination(:,i);
            input_subgrad.D_bar = D_current - input_subgrad.s.*q_current;
            output_subgrad = V3DP_b2b_dual_fixed_suppselect(input_subgrad);

            SUBGRAD_ALL_DISRUPT(:,i) = output_subgrad.beta;

        end

        subgrad = SUBGRAD_ALL_DISRUPT*disruption_prob;

    end

    SUBGRAD(:,t) = subgrad;


    %% Update the solution
    q_current = min( max(input.q_lb, q_current-stepsize(t)*subgrad), input.q_ub);
    Q(:,t) = q_current;

    
    %% Based on the scenarios of D_scenarios, s_scenarios (prob_scenarios), we evluate the (SAA) objective
    if mod(t, input.objeval_steps) == 0
        
        if input.show_objeval == 1
            fprintf("Evaluating Objective!")
        end

        % If using the most up-to-date solution to evaluate objective
        if input.ave_flag== 0
            q_eval = q_current;
        else
            ave_range = [floor(t*input.ave_ratio)+1:t];
            if input.stepsize_flag == 0
                % When constant stepsize
                q_eval = mean(Q(:,ave_range), 2);
            else
                % When 1/sqrt(t) stepsize
                ave_coeff = stepsize(ave_range)/sum(stepsize(ave_range));
                q_eval = sum(Q(:,ave_range)*ave_coeff, 2);
            end
        end

        % Records all the V3DP values for all sceanrios 
        V_3DP_all_scenarios = []; 

        for i = 1 : S

            if mod(i,floor(S/10))==1 && input.show_objeval == 1
                fprintf("%3.1f%%, ", i/S*100)
            end
            
            input_objeval.q = q_eval;
            input_objeval.s = s_scenarios(:,i);
            input_objeval.D = D_scenarios(:,i);
            input_objeval.D_bar = input_objeval.D - q_eval.*input_objeval.s;

            output_objeval = V3DP_b2b_dual_fixed_suppselect(input_objeval);

            V_3DP_all_scenarios(i,:) = output_objeval.opt_val_primal;
            
            if i == S && input.show_objeval == 1
                fprintf("Done!\n")
            end
            
        end
        
        OBJEVAL(objeval_count) = prob_scenarios'*V_3DP_all_scenarios;

        endTime = clock;
        time_spent = time_spent + etime(endTime, startTime);  
        startTime = clock;
        
        if input.benchmark_flag == 1

            %% If we have benchmark then we use it as stopping rule
            % Two stopping rules when we have benchmark:
            %   - when we haven improved (relative to the benchmark opt val) more than "stop_threshold_multisteps" in the past "stop_interval" steps
            %   - when we the error relative to the benchmark opt val reached "stop_threshold_singlestep"

            relative_err_to_benchmark = (OBJEVAL(objeval_count)-input.benchmark_optval)/input.benchmark_optval;
            fprintf("SGD Step %d, objval = %3.2f (relative err = %3.4f%%), time: %3.2f\n", t, OBJEVAL(objeval_count), relative_err_to_benchmark*100, time_spent)

            stop_interval = input.stop_interval;
            stop_threshold_mutlisteps = input.stop_threshold_multisteps;
            stop_threshold_singlestep = input.stop_threshold_singlestep;
            if objeval_count > stop_interval
                progress_for_stopping = (OBJEVAL(objeval_count-(stop_interval-1):objeval_count)-OBJEVAL(objeval_count-stop_interval:objeval_count-1))/input.benchmark_optval < -stop_threshold_mutlisteps;
            end

            if relative_err_to_benchmark < stop_threshold_singlestep
                fprintf("REACHED A GOOD ERROR, STOP!!! \n\n")

                break
            else
                if (objeval_count<=stop_interval) || (sum(progress_for_stopping)>0)
                    objeval_count = objeval_count + 1;
                else
                    fprintf("HAVEN'T CHANGED FOR A WHILE, STOP!!! \n\n")
                    break            
                end
            end

        else

            %% If we don't have benchmark 
            % we stop when we haven't improved more than "stop_threshold_multisteps" in the past "stop_interval" steps
            if t > 1
                fprintf("SGD Step %d, objval = %3.2f, imrpoved = %3.2f(%3.5f%%), time: %3.2f \n",...
                    t, OBJEVAL(objeval_count), OBJEVAL(objeval_count-1)-OBJEVAL(objeval_count),...
                    (OBJEVAL(objeval_count-1)-OBJEVAL(objeval_count))/OBJEVAL(objeval_count)*100 , time_spent)
            else
                fprintf("SGD Step %d, objval = %3.2f, time: %3.2f \n", t, OBJEVAL(objeval_count), time_spent)
            end

            stop_interval = input.stop_interval;
            stop_threshold_multisteps = input.stop_threshold_multisteps;
            if objeval_count > 10
                progress_for_stopping ...
                    = (OBJEVAL(objeval_count-stop_interval:objeval_count-1)-OBJEVAL(objeval_count-(stop_interval-1):objeval_count))./(OBJEVAL(objeval_count-stop_interval:objeval_count-1)) ...
                    > stop_threshold_multisteps;
            end

            if (objeval_count<=10) || sum(progress_for_stopping)>0
                objeval_count = objeval_count + 1;
            else
                fprintf("HAVEN'T CHANGED FOR A WHILE, STOP!!!\n\n")
                break            
            end

        end
        


    end
    
    



end

output.Q = Q;
output.OBJ = OBJEVAL;
output.SUBGRAD = SUBGRAD;
output.s_SAMPLES = s_SAMPLES;
output.D_SAMPLES = D_SAMPLES;
output.time_spent = time_spent;

ave_range = [floor(t*input.ave_ratio)+1:t];
ave_coeff = stepsize(ave_range)/sum(stepsize(ave_range));
q_ave_final = sum(Q(:,ave_range)*ave_coeff, 2);
output.q_ave_final = q_ave_final;


end
