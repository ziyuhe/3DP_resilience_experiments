function output = U3DP_MIP_Benders(input)

% =========================================================================
% Script Name:       U3DP_MIP_Benders.m
% Author:            Ziyu He
% Date:              02/01/2025
% Description:       
%   - Implements **Benders Decomposition** to optimize:
%       - First-stage order quantity
%       - 3DP backup selection  
%   - Operates under **fixed 3DP capacity (K)** in a **sales-oriented model**.
%
%% Key Functionalities:
%   - "input.GRB_flag": Uses Gurobi or a specialized solver for dual subproblems.
%   - "input.regularize_flag": Solves the master problem with/without first-stage decision regularization.
%   - "input.warmstart_flag": Warm starts Benders with predefined aggregated cuts.

%% Assumption: Fixed 3DP Capacity (K)
%   - Ideally, a binary variable should indicate 3DP adoption.
%   - Instead, we:
%       1. Solve the problem **ignoring** fixed 3DP cost (C^3DP).
%       2. Postprocess the solution to handle cases where x ~= 0.
% =========================================================================


% Input parameters
n = input.n; % number of products
S = input.S; % number of scenarios

prob_scenarios = input.prob_scenarios; % probability vector of all combinations of disruption and demand scenarios 
D_scenarios = input.D_scenarios;     % all demand scenarios (n by S)
s_scenarios = input.s_scenarios;     % all disruption scenarios (n by S)

c_3DP = input.c_3DP; 
v = input.v;
h = input.h;
weight = input.weight; % per unit weight of printing materials for each product
K_3DP = input.K_3DP;

mean_demand = input.mean_demand;

bigM1 = input.bigM1;    % big M coefficient for the first stage
bigM2 = input.bigM2;    % big M coefficient for the second stage (tighter than the first stage)

TM_cost = input.TM_cost;

C_3DP = input.C_3DP;

cost_of_3dp_per_machine_month = input.cost_of_3dp_per_machine_month;
speed_per_machine_month = input.speed_per_machine_month;


%% We assume the following steps in each Bender's iteration:
% Assume we start with some feasible solutions "q_current" and "x_current" in hand 
%   - First fix q_current and x_current, solve the V3DP problem for all uncertainty scenarios
%   - Second, check optimality
%       - Update ub_current that involves aggregated V3DP value (obj val of the feasible solution "q_current" and "x_current")
%       - Update lb_current that involves the most up-to-date "theta" (the optimal value of a problem with incomplete constraints)
%   - If ub_current =< lb_current or (ub_current-lb_current)/ub_current terminate
%   - Else add the "aggreagted" cut to the master problem (using the "piece" calculated from all scenarios)
%   - Finally, solve the masters problem and update q_current and x_current

Max_Steps = input.Max_Steps;

q_current = input.q_init;
x_current = input.x_init;
q_best = q_current;
x_best = x_current;

Q = [];  % n-by-t matrix
X = [];  % n-by-t matrix
THETA = [];

ub_current = inf;
ub_best = inf;
lb_current = -inf;

UB_BEST = [];    % t-by-1 vector
UB = [];    % t-by-1 vector
LB = [];    % t-by-1 vector

% Collects info needed for all the cuts
if input.warmstart_flag == 1
    CUT_CONST = input.CUT_CONST; % t-by-1 vector
    CUT_X_VEC = input.CUT_X_VEC; % n-by-t matrix
    CUT_Q_VEC = input.CUT_Q_VEC; % n-by-t matrix
    init_num_cuts = length(CUT_CONST);
else
    CUT_CONST = [];     % t-by-1 vector
    CUT_X_VEC = [];     % n-by-t matrix
    CUT_Q_VEC = [];     % n-by-t matrix
    init_num_cuts = 0;
end

startTime = clock;
time_sofar = 0;

if ~isfield(input, 'stopping_time_flag')
    input.stopping_time_flag = 0;
    input.stopping_time = 1e8;
end

for t = 1 : Max_Steps
    
    endTime = clock;
    time_sofar = time_sofar + etime(endTime, startTime);
    
    if input.display_flag == 1 && mod(t,1) == 0
        fprintf("-------------------------------------------------- \n")
        fprintf("Bender's Master Step %d,     Time %3.2f\n", t, time_sofar)
        fprintf("-------------------------------------------------- \n")
    end

    %% If we have reached a computing time limit before reaching the max number of iterations or reaching the stopping criteria, we stop
    if input.stopping_time_flag == 1 && time_sofar > input.stopping_time
        fprintf("-------------------------------------------------- \n")
        fprintf("REACHED TIME LIMIT %3.2f !!! TERMINATE!!!\n\n", input.stopping_time)
        fprintf("-------------------------------------------------- \n")
        break
    end

    startTime = clock;

    %% Solve the subproblems, each corresponds to a particular scenario
    % Records all the V3DP values for all sceanrios 
    V_3DP_all_scenarios = []; 

    % These elements are needed for adding the cut
    GAMMA = []; % 1-by-S
    ALPHA = []; % n-by-S
    BETA = [];  % n-by-S
    RHO = [];   % n-by-S

    input_V3DP.c_3DP = c_3DP;
    input_V3DP.v = v;
    input_V3DP.h = h;
    input_V3DP.weight = weight;
    input_V3DP.x = x_current;
    input_V3DP.q = q_current;
    input_V3DP.M = bigM2;
    input_V3DP.K_3DP = K_3DP;

    ERR = [];

    for i = 1:S

        if mod(i,floor(S/1)) == 0 && input.display_flag == 1 && mod(t,1) == 0
            fprintf("Master Step %d: Subproblem %d out of %d (%3.2f%%)\n", t, i, S, i/S*100)
        end

        input_V3DP.s = s_scenarios(:,i);
        input_V3DP.D = D_scenarios(:,i);
        input_V3DP.D_bar = input_V3DP.D - q_current.*input_V3DP.s;

        if input.GRB_flag == 1
            output_V3DP = V_hat_3DP_dual(input_V3DP);
            V_3DP_all_scenarios(i,:) = output_V3DP.opt_val;
        else
            output_V3DP = V3DP_b2b_dual(input_V3DP);
            V_3DP_all_scenarios(i,:) = output_V3DP.opt_val_primal;
        end

        GAMMA(i,:) = output_V3DP.gamma;
        ALPHA(:,i) = output_V3DP.alpha;
        BETA(:,i) = output_V3DP.beta;
        RHO(i,:) = sum(output_V3DP.rho);

        ERR(i) = V_3DP_all_scenarios(i,:) - (GAMMA(i,:)*K_3DP+RHO(i,:) + x_current'*ALPHA(:,i) + q_current'*BETA(:,i));

    end


    %% Update the upper bound and check termniation
    ub_current = sum((1-x_current).*TM_cost) - sum(v.*mean_demand.*x_current) + prob_scenarios'*V_3DP_all_scenarios;
    ub_current_for_gap = -x_current'*TM_cost - sum(v.*mean_demand.*x_current) + prob_scenarios'*V_3DP_all_scenarios;
    lb_current_for_gap = lb_current - sum(TM_cost);
    UB(t,:) = ub_current;
    
    % Update the incumbent solution if the upper bound is changed
    if ub_current < ub_best
        ub_best = ub_current;
        q_best = q_current;
        x_best = x_current;
        theta_best = prob_scenarios'*V_3DP_all_scenarios;
    end
    UB_BEST(t,:) = ub_best;
    
    abs_gap = (ub_current_for_gap - lb_current_for_gap);
    % relative_gap = abs_gap/abs(ub_current_for_gap);
    relative_gap = abs_gap/abs(lb_current);

    abs_best_gap = (ub_best - lb_current);
    % relative_best_gap = abs_best_gap/abs(ub_current_for_gap);
    relative_best_gap = abs_best_gap/abs(lb_current);

    if isnan(relative_gap) && abs_gap == 0
        relative_gap = 0;
    end
    
    if  relative_gap < input.tolerance
        if input.display_flag == 1 && mod(t,1) == 0
            % fprintf("UB: %3.2f, Best UB: %3.2f, LB: %3.2f \n", ub_current_for_gap, ub_best-sum(TM_cost), lb_current_for_gap )
            fprintf("UB: %3.2f, Best UB: %3.2f, LB: %3.2f \n", ub_current, ub_best, lb_current )
            fprintf("Optimality Gap: (abs) %3.4f, (relative) %3.5f %% \n", abs_gap, relative_gap*100 )
            fprintf("Optimality Gap Reached Tolerance (%3.2f %%) \n", input.tolerance*100)
            fprintf("-------------------------------------------------- \n\n")
        end
        break
    else
        if input.display_flag == 1 && mod(t,1) == 0
            % fprintf("UB: %3.2f, Best UB: %3.2f, LB: %3.2f \n", ub_current_for_gap, ub_best-sum(TM_cost), lb_current_for_gap )
            fprintf("UB: %3.2f, Best UB: %3.2f, LB: %3.2f \n", ub_current, ub_best, lb_current )
            fprintf("Current Optimality Gap: (abs) %3.4f, (relative) %3.5f %%  ( x: %d, q: %3.2f )\n", abs_gap, relative_gap*100, sum(x_current), norm(q_current) )
            fprintf("Best Optimality Gap:    (abs) %3.4f, (relative) %3.5f %%  ( x: %d, q: %3.2f )\n", abs_best_gap, relative_best_gap*100,  sum(x_best), norm(q_best) )
            fprintf("-------------------------------------------------- \n\n")
        end
    end


    %% Add the new cut and solve the master problem
    cut_const = prob_scenarios'*GAMMA*K_3DP + prob_scenarios'*RHO;
    cut_x_vec = ALPHA*prob_scenarios;
    cut_q_vec = BETA*prob_scenarios;

    CUT_CONST(init_num_cuts+t,:) = cut_const;
    CUT_X_VEC(:,init_num_cuts+t) = cut_x_vec;
    CUT_Q_VEC(:,init_num_cuts+t) = cut_q_vec;

    input_master.v = v;
    input_master.TM_cost = TM_cost;
    input_master.mean_demand = mean_demand;
    input_master.bigM = bigM1;
    input_master.q_ub = input.q_ub;
    input_master.q_lb = input.q_lb;
    
    input_master.CUT_CONST = CUT_CONST;
    input_master.CUT_X_VEC = CUT_X_VEC;
    input_master.CUT_Q_VEC = CUT_Q_VEC;
    
    input_master.n = n;
    
    if input.regularize_flag == 0
        output_master = U3DP_Master_Solver(input_master);
    else
        input_master.q_ref = q_best;
        input_master.x_ref = x_best;
        input_master.theta_ref = theta_best;
        input_master.q_reg_coeff = input.q_reg_coeff;
        input_master.x_reg_coeff = input.x_reg_coeff;
        input_master.theta_reg_coeff = input.theta_reg_coeff;
        output_master = U3DP_Master_Solver_Regularized(input_master);
    end

    x_current = (output_master.x > 1e-3);
    q_current = output_master.q;
    theta_current = output_master.theta;

    lb_current = output_master.opt_val;
    
    X(:,t) = x_current;
    Q(:,t) = q_current;
    THETA(t,:) = theta_current;
    LB(t,:) = lb_current;


end

output.X = X;
output.Q = Q;
output.THETA = THETA;
output.UB_BEST = UB_BEST;
output.UB = UB;
output.LB = LB;
output.CUT_CONST = CUT_CONST;
output.CUT_X_VEC = CUT_X_VEC;
output.CUT_Q_VEC = CUT_Q_VEC;

output.q_best = q_best;
output.x_best = x_best;

endTime = clock;
time_sofar = time_sofar + etime(endTime, startTime);
output.solving_time = time_sofar;

output.abs_best_gap = abs_best_gap;
output.relative_best_gap = relative_best_gap;


%% NOTE: The solved problem differs from the actual supplier selection problem under fixed K.
% - In the real problem, when **x = 0** (no suppliers backed up by 3DP), we should include **C_3DP** in the cost.
% - However, in this formulation, **C_3DP** is included regardless of whether 3DP is selected.

%% Possible Fixes:
% 1. **Introduce a binary variable (y)**:
%    - Modify the cost term to **C_3DP * y**, where **y = 1** if 3DP is used, and **y = 0** otherwise.
%    - Requires re-solving the problem when **C_3DP** changes (e.g., due to 3DP speed or depreciation costs).

% 2. **Postprocess the optimal solution**:
%    - If **x = 0**, retain the current solution and total cost.
%    - If **x ~= 0**, compare:
%        1. Current optimal cost **+ C_3DP**.
%        2. Cost when using only TM (sum of TM costs).
%    - If (1) is lower, keep the current solution.
%    - If (2) is lower, set all solutions to zero and use the TM-only cost.


startTime = clock;

x_zero_indicator = (sum(x_best>1e-3) > 0);

Q_FINAL = {}; X_FINAL = {}; 
TOTAL_COST = [];
TOTAL_COST_NONZERO = [];

if x_zero_indicator == 0

    for i = 1 : length(speed_per_machine_month)  
        for j = 1 : length(cost_of_3dp_per_machine_month)
            
            TOTAL_COST(i,j) = sum(TM_cost);
            X_FINAL{i,j} = (x_best>1e-3); Q_FINAL{i,j} = q_best; 
            TOTAL_COST_NONZERO(i,j) = sum(TM_cost);

        end
    end

else

    %% Handling the Case Where No 3DP is More Cost-Effective
    % - The upper bound (UB) at termination is the **SAA objective value** of the final solution 
    %   based on the in-procedure Benders data.
    % - However, this value has two key issues for post-processing:
    %     1. **Small Sample Size**: The evaluation may be based on an insufficient number of samples.
    %     2. **Inaccurate q**: The best solutions (**q_best, x_best**) come from the master problem,
    %        but **q_best** might not be optimal for the given **x_best**.
    
    %% Three Resolution Strategies:
    % - **input.recompute_flag == 0** → No recomputation.
    % - **input.recompute_flag == 1** → Recompute the SAA objective value for **q_best, x_best** 
    %   using a much larger sample size.
    % - **input.recompute_flag == 2** → Use SGD to refine **q_best** while keeping **x_best** fixed,
    %   then evaluate the SAA objective value with a much larger sample size.

    if input.recompute_flag == 0

        %% We simply don't re-compute and use the best upper bound for post-processing

        for i = 1 : length(speed_per_machine_month)  
            for j = 1 : length(cost_of_3dp_per_machine_month)
    
                if (UB_BEST(end) + C_3DP(i,j)) <  sum(TM_cost)
    
                    TOTAL_COST(i,j) = UB_BEST(end) + C_3DP(i,j);
                    X_FINAL{i,j} = (x_best>1e-3); Q_FINAL{i,j} = q_best; 
    
                else
                    
                    TOTAL_COST(i,j) = sum(TM_cost);
                    X_FINAL{i,j} = zeros(n,1); Q_FINAL{i,j} = zeros(n,1);
    
                end

                TOTAL_COST_NONZERO(i,j) = UB_BEST(end) + C_3DP(i,j);
    
            end
        end

    elseif input.recompute_flag == 1

        %% We still use the (SAA) objective of x_best, q_best, for post-processing, but we recompute it with a much larger dataset
        %% First re-sample a larger dataset
        input_recompute_sampled.num_suppliers = input.n;
        input_recompute_sampled.num_scenarios = input.num_scenarios;
        input_recompute_sampled.yield_loss_rate = input.yield_loss_rate;
        input_recompute_sampled.p_disrupt = input.p;
        input_recompute_sampled.Monthly_Quantity = input.Monthly_Quantity;
        input_recompute_sampled.Demand_Probability = input.Demand_Probability;
        
        input_recompute_sampled.sample_mode = input.recompute_sample_mode;
        input_recompute_sampled.disruption_sample_flag = input.recompute_disruption_sample_flag;
        input_recompute_sampled.demand_sample_flag = input.recompute_demand_sample_flag;
        
        if input_recompute_sampled.sample_mode == 1
        
            if input_recompute_sampled.disruption_sample_flag == 1
                input_recompute_sampled.disruption_samplesize = input.recompute_disruption_samplesize;
            end
            if input_recompute_sampled.demand_sample_flag == 1
                input_recompute_sampled.demand_samplesize = input.recompute_demand_samplesize;
            end
        
        else
        
            if input_recompute_sampled.disruption_sample_flag == 1
                input_recompute_sampled.disruption_samplesize = input.recompute_disruption_samplesize;
            end
            input_recompute_sampled.demand_num_per_disruption = input.recompute_demand_samplesize;
        
        end
        
        output_recompute_benders = Data_prep_for_MIP(input_recompute_sampled);
        
        recompute_disruption_demand_joint_prob_benders = output_recompute_benders.disruption_demand_joint_prob;
        recompute_failure_data_benders = output_recompute_benders.failure_data;
        recompute_demand_data_benders = output_recompute_benders.demand_data;

        %% Recompute the best upper bound value
        V_3DP_all_scenarios = []; 

        % These elements are needed for adding the cut
        GAMMA = []; % 1-by-S
        ALPHA = []; % n-by-S
        BETA = [];  % n-by-S
        RHO = [];   % n-by-S
    
        input_V3DP.c_3DP = c_3DP;
        input_V3DP.v = v;
        input_V3DP.h = h;
        input_V3DP.weight = weight;
        input_V3DP.x = x_best;
        input_V3DP.q = q_best;
        input_V3DP.M = bigM2;
        input_V3DP.K_3DP = K_3DP;
    
        ERR = [];
    
        for i = 1:size(recompute_demand_data_benders,2)
    
            if mod(i,1000000) == 0 && input.display_flag == 1 && mod(t,1) == 0
                fprintf("Recomputation: Sample %d\n", i)
            end
    
            input_V3DP.s = recompute_failure_data_benders(:,i);
            input_V3DP.D = recompute_demand_data_benders(:,i);
            input_V3DP.D_bar = input_V3DP.D - q_best.*input_V3DP.s;
    
            if input.GRB_flag == 1
                output_V3DP = V_hat_3DP_dual(input_V3DP);
                V_3DP_all_scenarios(i,:) = output_V3DP.opt_val;
            else
                output_V3DP = V3DP_b2b_dual(input_V3DP);
                V_3DP_all_scenarios(i,:) = output_V3DP.opt_val_primal;
            end
    
            GAMMA(i,:) = output_V3DP.gamma;
            ALPHA(:,i) = output_V3DP.alpha;
            BETA(:,i) = output_V3DP.beta;
            RHO(i,:) = sum(output_V3DP.rho);
    
            ERR(i) = V_3DP_all_scenarios(i,:) - (GAMMA(i,:)*K_3DP+RHO(i,:) + x_best'*ALPHA(:,i) + q_best'*BETA(:,i));
    
        end
    
    
        %% Recompute the best upper bound and post-process
        ub_best_large_sample = sum((1-x_best).*TM_cost) - sum(v.*mean_demand.*x_best) + recompute_disruption_demand_joint_prob_benders'*V_3DP_all_scenarios;

        for i = 1 : length(speed_per_machine_month)  
            for j = 1 : length(cost_of_3dp_per_machine_month)
    
                if (ub_best_large_sample + C_3DP(i,j)) <  sum(TM_cost)
    
                    TOTAL_COST(i,j) = ub_best_large_sample + C_3DP(i,j);
                    X_FINAL{i,j} = (x_best>1e-3); Q_FINAL{i,j} = q_best; 
    
                else
                    
                    TOTAL_COST(i,j) = sum(TM_cost);
                    X_FINAL{i,j} = zeros(n,1); Q_FINAL{i,j} = zeros(n,1);
    
                end

                TOTAL_COST_NONZERO(i,j) = ub_best_large_sample + C_3DP(i,j);
    
            end
        end


    else

        %% Using SGD: We take the current supplier selection (x_best <=> "set_of_3DP") and recompute "q part"
        %% Also compute the objective: U3DP("set_of_3DP", K) + UTM("set_of_TM", K) + C3DP(K)
        
        %% Some important specifications of running SGD here:
        %% - Do we evalute in-process and how much data do we use to do so?
        %% - How much data do we use to evaluate the final solution?

        supplier_3dp_select = (x_best > 1e-3);
    
        p = input.p;
        yield_loss_rate = input.yield_loss_rate;
        num_suppliers = input.n;
        num_scenarios = input.num_scenarios;
    
        Monthly_Quantity = input.Monthly_Quantity;
        Monthly_Weight = input.Monthly_Weight;
        Demand_Probability = input.Demand_Probability;
        mean_demand = input.mean_demand;
    
        one2n = [1:num_suppliers];
    
        set_of_3DP = one2n(supplier_3dp_select);
        set_of_TM = one2n(logical(1-supplier_3dp_select));
    
        num_suppliers_3dp = length(set_of_3DP);


        %% First sample a larger set just for objective evaluation during SGD
        input_preprocess.num_suppliers = num_suppliers_3dp;
        input_preprocess.num_scenarios = num_scenarios;
        input_preprocess.yield_loss_rate = yield_loss_rate;
        input_preprocess.p_disrupt = p;
        input_preprocess.Monthly_Quantity = Monthly_Quantity(set_of_3DP, :);
        input_preprocess.Demand_Probability = Demand_Probability(set_of_3DP, :);

        input_preprocess.sample_mode = input.recompute_sample_mode;
        input_preprocess.disruption_sample_flag = input.recompute_disruption_sample_flag;
        input_preprocess.demand_sample_flag = input.recompute_demand_sample_flag;
        
        if input_preprocess.sample_mode == 1

            if input_preprocess.disruption_sample_flag == 1
                input_preprocess.disruption_samplesize = input.recompute_disruption_samplesize_eval;
            end
            if input_preprocess.demand_sample_flag == 1
                input_preprocess.demand_samplesize = input.recompute_demand_samplesize_eval;
            end
        
        else
        
            if input_preprocess.disruption_sample_flag == 1
                input_preprocess.disruption_samplesize = input.recompute_disruption_samplesize_eval;
            end
            input_preprocess.demand_num_per_disruption = input.recompute_demand_samplesize_eval;
        
        end
        
        output_preprocess = Data_prep_for_MIP(input_preprocess);
        
        disruption_demand_joint_prob_SGD_inprocess_eval = output_preprocess.disruption_demand_joint_prob;
        failure_data_SGD_inprocess_eval = output_preprocess.failure_data;
        demand_data_SGD_inprocess_eval = output_preprocess.demand_data;
        
        %% Second sample a even larger set just for final objective evaluation
        if input_preprocess.sample_mode == 1

            if input_preprocess.disruption_sample_flag == 1
                input_preprocess.disruption_samplesize = input.recompute_disruption_samplesize_finaleval;
            end
            if input_preprocess.demand_sample_flag == 1
                input_preprocess.demand_samplesize = input.recompute_demand_samplesize_finaleval;
            end
        
        else
        
            if input_preprocess.disruption_sample_flag == 1
                input_preprocess.disruption_samplesize = input.recompute_disruption_samplesize_finaleval;
            end
            input_preprocess.demand_num_per_disruption = input.recompute_demand_samplesize_finaleval;
        
        end
        
        output_preprocess = Data_prep_for_MIP(input_preprocess);
        
        disruption_demand_joint_prob_SGD_finaleval = output_preprocess.disruption_demand_joint_prob;
        failure_data_SGD_finaleval = output_preprocess.failure_data;
        demand_data_SGD_finaleval = output_preprocess.demand_data;


        %% Setup SGD
        input_sgd.n = num_suppliers_3dp;
        input_sgd.c_3DP = c_3DP(set_of_3DP);
        input_sgd.v = v(set_of_3DP);
        input_sgd.h = h(set_of_3DP);
        input_sgd.weight = weight(set_of_3DP);
        input_sgd.mean_demand = mean_demand(set_of_3DP);
        
        input_sgd.q_ub = input.q_ub(set_of_3DP);    % HERE WE ADD BOUNDS FOR 1ST STAGE DECISIONS
        input_sgd.q_lb = input.q_lb(set_of_3DP);    % HERE WE ADD BOUNDS FOR 1ST STAGE DECISIONS
        
        input_sgd.prob_scenarios = disruption_demand_joint_prob_SGD_inprocess_eval;
        input_sgd.D_scenarios = demand_data_SGD_inprocess_eval;
        input_sgd.s_scenarios = failure_data_SGD_inprocess_eval;
        input_sgd.S = size(demand_data_SGD_inprocess_eval,2);

        input_sgd.K_3DP = K_3DP;
        input_sgd.C_3DP = C_3DP;
    
        input_sgd.yield_loss_rate = yield_loss_rate;
        input_sgd.p_disrupt = p;
        input_sgd.Monthly_Quantity = Monthly_Quantity(set_of_3DP, :);
        input_sgd.Demand_Probability = Demand_Probability(set_of_3DP, :);
        input_sgd.num_scenarios = num_scenarios;
    
        % Max SGD steps
        input_sgd.Max_Steps = input.recompute_sgd_Maxsteps;
    
        % Steps for evaluating obj val
        input_sgd.show_objeval = 0;
        input_sgd.objeval_steps = input_sgd.Max_Steps+1;
        % input_sgd.objeval_steps = ceil(input_sgd.Max_Steps*0.01);
    
        % When we evaluate the obj val, do we use the averaged solution ("input.ave_flag = 1" <=> do average)
        % We take average between floor(t*ave_ratio)+1 to t
        input_sgd.ave_flag = 1;
        input_sgd.ave_ratio = 1/2;
        
        % Sample disruptions ("disrupt_sample_flag == 1" <=> only sample one disruption combo per step)
        input_sgd.disrupt_sample_flag = 1;
        input_sgd.disruption_prob = output_preprocess.disruption_prob;
        input_sgd.failure_combinations = output_preprocess.failure_combinations;
        
        % We can use the GRB solution to initialize
        input_sgd.q_init = q_best(set_of_3DP);
    
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
    
        % Here we assume we don't have benchmark to help us terminate
        input_sgd.benchmark_flag = 0;
        input_sgd.stop_interval = 5;
        input_sgd.stop_threshold_multisteps = 5e-6;

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



        %% We use the bigger dataset to evaluate the final the objective of SGD solution
        input_objeval.q_eval = output_sgd.q_ave_final;
        input_objeval.n = num_suppliers_3dp;
        input_objeval.c_3DP = c_3DP(set_of_3DP);
        input_objeval.v = v(set_of_3DP);
        input_objeval.h = h(set_of_3DP);
        input_objeval.weight = weight(set_of_3DP);
        input_objeval.mean_demand = mean_demand(set_of_3DP);
        input_objeval.K_3DP = K_3DP;
        input_objeval.prob_scenarios = disruption_demand_joint_prob_SGD_finaleval;
        input_objeval.D_scenarios = demand_data_SGD_finaleval;
        input_objeval.s_scenarios = failure_data_SGD_finaleval;
        input_objeval.S = size(demand_data_SGD_finaleval,2);
        input_objeval.display_flag = 1;
        input_objeval.display_interval = 1e5;
        obj_fullinfo_sgd = U3DP_objeval_fixed_suppselect(input_objeval); % The output here is U3DP(A,K) - sum(v.*mean_demand)

        % The total recomputed value is U3DP(A,K) + C3DP(K) + UTM(A^c), where
        %       - U3DP(A,K) = obj_fullinfo_sgd - sum(mean_demand(set_of_3DP).*v(set_of_3DP))
        %       - UTM(A^c) = sum(TM_cost(set_of_TM)) 
        recomputed_total_cost = obj_fullinfo_sgd - sum(mean_demand(set_of_3DP).*v(set_of_3DP)) + C_3DP + sum(TM_cost(set_of_TM));


        for i = 1 : length(speed_per_machine_month)  
            for j = 1 : length(cost_of_3dp_per_machine_month)
    
                if recomputed_total_cost(i,j) <  sum(TM_cost)
    
                    TOTAL_COST(i,j) = recomputed_total_cost(i,j);
                    X_FINAL{i,j} = (x_best>1e-3); 
                    q_best_update = zeros(n,1); q_best_update(set_of_3DP) = output_sgd.q_ave_final;
                    Q_FINAL{i,j} = q_best_update; 
    
                else
                    
                    TOTAL_COST(i,j) = sum(TM_cost);
                    X_FINAL{i,j} = zeros(n,1); Q_FINAL{i,j} = zeros(n,1);
    
                end
    
                TOTAL_COST_NONZERO(i,j) = recomputed_total_cost(i,j);
    
            end
        end


    end




end


endTime = clock;
output.recompute_time = etime(endTime, startTime);

output.total_time = output.recompute_time + output.solving_time;


output.TOTAL_COST = TOTAL_COST;
output.TOTAL_COST_NONZERO = TOTAL_COST_NONZERO;
output.X_FINAL = X_FINAL;
output.Q_FINAL = Q_FINAL;


if t < Max_Steps
    fprintf("Bender's Stopped by Opt. Gap (<%3.5f%%)!!!\n\n", input.tolerance*100)
else
    fprintf("Bender's Stopped by Reaching Time Limit!!!\n\n")
end








end