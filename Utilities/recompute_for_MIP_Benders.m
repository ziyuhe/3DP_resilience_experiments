function output = recompute_for_MIP_Benders(input)

% =========================================================================
% Script Name:       recompute_for_MIP_Benders.m
% Date:              02/01/2025
% Description:       
%   - A stand-along function to post-process the results obtained by MIP_Benders
%   - See "U3DP_MIP_Benders.m" for more details
% =========================================================================


% Input parameters
n = input.n; % number of products

c_3DP = input.c_3DP; 
v = input.v;
h = input.h;
weight = input.weight; % per unit weight of printing materials for each product
K_3DP = input.K_3DP;

mean_demand = input.mean_demand;

bigM1 = input.bigM1;    % big M coefficient for the first stage
bigM2 = input.bigM2;    % big M coefficient for the second stage (tighter than the first stage)

TM_cost = input.TM_cost;
nobackup_cost = input.nobackup_cost;

C_3DP = input.C_3DP;

cost_of_3dp_per_machine_month = input.cost_of_3dp_per_machine_month;
speed_per_machine_month = input.speed_per_machine_month;

%% Get the solution
x_best = input.x_best;
q_best = input.q_best;

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

    %% Address the possible scenario that having no 3DP is cheaper
    %% The current "UB" at termination is the "SAA obj val" of the final solution using the "in-procedure" data of Benders
    %% There are two key issues with this value when used for post-processing:
    %%      (1) "Small sample": the data used to evaluate it might be too small!
    %%      (2) "Inaccurate q": both "q_best" and "x_best" comes from some masters solution, the "q_best" might not be optimal for this "x_best"
    %% Three resolutions:
    %% If "input.recompute_flag == 0", then we don't recompute
    %% If "input.recompute_flag == 1", then we recompute the "SAA obj val" of "q_best, x_best" with a much larger sample size
    %% If "input.recompute_flag == 2", then we apply SGD to "fixed x_best" problem for a better "q_best", and evaluate the "SAA obj val" with a much large sample size


    if input.auto_recompute == 1
        
        if sum(x_best) < 5
    
            %% For |A_t|>5, we can still apply full info GRB
            input.recompute_flag = 1;
            input.recompute_sample_mode = 1;
            input.recompute_disruption_sample_flag = 0;
            input.recompute_demand_sample_flag = 0;
            input.GRB_display = 0;
    
        elseif sum(x_best) <= 12
            
            %% For 5<|A_t|<=12, we apply SGD and for the final evaluation we only sample demand
            input.recompute_flag = 2;
            input.recompute_sample_mode = 2;
            input.recompute_disruption_sample_flag = 0;
            input.recompute_demand_sample_flag = 1;    
            input.recompute_demand_samplesize_eval = 100; 
            input.recompute_demand_samplesize_finaleval = 100; 
    
        else
    
            %% For |A_t|>12, we apply SGD and for the final evaluation we sample both disruption and demand
            input.recompute_flag = 2;
            input.recompute_sample_mode = 2;
            input.recompute_disruption_sample_flag = 1;
            input.recompute_demand_sample_flag = 1;  
            input.recompute_disruption_samplesize_eval = 2^12; 
            input.recompute_demand_samplesize_eval = 100; 
            input.recompute_disruption_samplesize_finaleval = 2^12; 
            input.recompute_demand_samplesize_finaleval = 100; 
    
        end
    
    end

    if input.recompute_flag == 1

        %% Using GRB: We take the current supplier selection (x <=> "set_of_3DP") and recompute the whole objective value:
        %% U3DP("set_of_3DP", K) + UTM("set_of_TM", K) + C3DP(K)
    
        supplier_3dp_select = x_best;
    
        p = input.p;
        yield_loss_rate = input.yield_loss_rate;
        num_suppliers = input.n;
        num_scenarios = input.num_scenarios;
    
        Monthly_Quantity = input.Monthly_Quantity;
        Monthly_Weight = input.Monthly_Weight;
        Demand_Probability = input.Demand_Probability;
        mean_demand = input.mean_demand;
    
        c_3DP = input.c_3DP;
        v = input.v;
        h = input.h;
        weight = input.weight;
    
        one2n = [1:num_suppliers];
    
        set_of_3DP = one2n(supplier_3dp_select);
        set_of_TM = one2n(logical(1-supplier_3dp_select));
    
        num_suppliers_3dp = length(set_of_3DP);
    
        % Prepare data for Gurobi
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
                input_preprocess.disruption_samplesize = input.recompute_disruption_samplesize;
            end
            if input_preprocess.demand_sample_flag == 1
                input_preprocess.demand_samplesize = input.recompute_demand_samplesize;
            end
        
        else
        
            if input_preprocess.disruption_sample_flag == 1
                input_preprocess.disruption_samplesize = input.recompute_disruption_samplesize;
            end
            input_preprocess.demand_num_per_disruption = input.recompute_demand_samplesize;
        
        end
        
        output_preprocess = Data_prep_for_MIP(input_preprocess);
        
        disruption_demand_joint_prob = output_preprocess.disruption_demand_joint_prob;
        failure_data = output_preprocess.failure_data;
        demand_data = output_preprocess.demand_data;   
    
        % Prepare for Gurobi solver
        input_template.n = num_suppliers_3dp;
        input_template.S = size(demand_data,2);
        input_template.c_3DP = c_3DP(set_of_3DP);
        input_template.v = v(set_of_3DP);
        input_template.h = h(set_of_3DP);
        input_template.weight = weight(set_of_3DP);
        input_template.mean_demand = mean_demand(set_of_3DP); 
        
        input_recompute = input_template;
        input_recompute.prob_scenarios = disruption_demand_joint_prob;
        input_recompute.D_scenarios = demand_data;
        input_recompute.s_scenarios = failure_data;
    
        input_recompute.q_ub = input.q_ub(set_of_3DP);
        input_recompute.q_lb = input.q_lb(set_of_3DP);
    
        prep_recompute = U3DP_fixed_3DPbackup_fixed_K_GRB_Prep(input_recompute);
    
        input_recompute.obj_vec = prep_recompute.obj_vec;
        input_recompute.A = prep_recompute.A;
        input_recompute.b = prep_recompute.b;
    
        input_recompute.K_3DP = K_3DP;
        input_recompute.C_3DP = C_3DP;
    
        input_recompute.GRB_display = input.GRB_display;
    
        output_recompute = U3DP_fixed_3DPbackup_fixed_K_GRB(input_recompute);
    
        % The total recomputed value is U3DP(A,K) + C3DP(K) + UTM(A^c), where
        %       - "output_recompute.total_3DP_cost" = U3DP(A,K)+C3DP(K) 
        %       - "sum(TM_cost(set_of_TM))" =  UTM(A^c)
        recomputed_total_cost = output_recompute.total_3DP_cost + sum(TM_cost(set_of_TM));
    
        %% We compare Term 1 to Term 2, where
        %% Term 1 <=> (approx.) optimal value under the given K>0
        %% Term 2 <=> optimal value when K=0
        for i = 1 : length(speed_per_machine_month)  
            for j = 1 : length(cost_of_3dp_per_machine_month)
    
                if recomputed_total_cost(i,j) <  sum(min(TM_cost, nobackup_cost))
    
                    TOTAL_COST(i,j) = recomputed_total_cost(i,j);
                    X_FINAL{i,j} = x_best; 
                    q_SP = zeros(num_suppliers,1); q_SP(set_of_3DP) = output_recompute.q_SP;
                    Q_FINAL{i,j} = q_SP; 
    
                else
                    %% If K=0 is better, we output A0 (should be understood as without backup)
                    %%      - In X_FINAL, all items in A0 are denoted with INF
                    %%      - In Q_FINAL, we let everything be zero
                    %%      - We let A_FINAL be empty
                    TOTAL_COST(i,j) = sum(min(TM_cost, nobackup_cost));
                    X_FINAL{i,j} = zeros(num_suppliers,1); X_FINAL{i,j}(nobackup_cost<TM_cost) = Inf;
                    Q_FINAL{i,j} = zeros(num_suppliers,1);
    
                end
    
                TOTAL_COST_NONZERO(i,j) = recomputed_total_cost(i,j);
    
            end
        end


    else

        %% Using SGD: We take the current supplier selection (x_best <=> "set_of_3DP") and recompute "q part"
        %% Also compute the objective: U3DP("set_of_3DP", K) + UTM("set_of_TM", K) + C3DP(K)
        
        %% Some important specifications of running SGD here:
        %% - Do we evalute in-process and how much data do we use to do so?
        %% - How much data do we use to evaluate the final solution

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



output.TOTAL_COST = TOTAL_COST;
output.TOTAL_COST_NONZERO = TOTAL_COST_NONZERO;
output.X_FINAL = X_FINAL;
output.Q_FINAL = Q_FINAL;



end
