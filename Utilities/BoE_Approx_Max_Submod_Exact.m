function output = BoE_Approx_Max_Submod_Exact(input)

% =========================================================================
% Script Name:      BoE_Approx_Max_Submod_Exact.m
% Date:              02/01/2025
% Description:       
%   - Implements a **local search algorithm** for **supermodular approximation (BOE)** 
%     under a given **3DP capacity (K)**.
%   - Designed to handle:
%       - **Large-scale cases**.
%   - The expectaion here is computed exactly, hence:
%       - we need finite disruption and demand scenarios
%       - only doable for smaller cases
%
%% Post-Processing Procedure:
%   - Refines the 3DP backup set obtained from local search.
%   - Computes the **optimal first-stage order quantity** and corresponding costs.
%   - Compares the derived **3DP policy** against a **no-3DP** scenario.
%   - Chooses between:
%       - **MIP Approach** <= If the selected 3DP set is small.
%       - **SGD Approach** <= If the selected 3DP set is large.
%
%% NOTE: For Larger n, use "BoE_Approx_Max_Submod_SAA" or "BoE_Approx_Max_Submod_SAA_alternative"
% =========================================================================


%% NOTE THIS METHOD APPLIES LOCAL SEARCH ALGORITHM TO OUR SUPERMODULAR APPROXIMATION (BOE)
%% IN THIS FUNCTION, WE COMPUTE THE EXPECTATION OF THE SUPERMODULAR APPROXIMAITON EXACTLY:
%%      - ASSUME FINITE DISRUPTION AND DEMAND SCEANRIOS
%%      - ONLY DOABLE WIHT <30 SUPPLIERS 
%% FOR LARGER n, CHECK "BoE_Approx_Max_Submod_SAA"

startTime = clock;


num_suppliers = input.num_suppliers;
one2n = [1:num_suppliers];

U0 = input.U0;
Obj_const = input.Obj_const;
TM_Delta = input.TM_Delta ;
K_3DP = input.K_3DP;
C_3DP = input.C_3DP;
r = input.ratio_over_weight;
pi_p = input.pi_p;
pi_0 = input.pi_0;


%% The local search scheme
%   - We don't store subsets in the current 3DP selection A_t, but the subsets are accesible through "get_indices" functions
%   - We keep two vectors with length being the number of subsets in A_t:
%       - alpha_t: for subset A' of A_t, alpha_t is the product of [pi_p in A'] and [pi_0 in A_t\A']
%       - m_t:     for subest A' of A_t, m_t is the maximum of r in A'
%   - Call the increment of adding one more element "Inc_add"
%   - Call the increment of deleting one element "Inc_del"

% A_t stores the absolute index of 3DP selection (not necessarily in order)
A_init = input.A_init;
A_t = A_init;
A_tc = setdiff(one2n, A_t); % A_tc is the complement of A_t

% Get the initial binary matrix "B_t", vectors "alpha_t", "m_t"
n_init = length(A_init);
if n_init > 0

    for combo = 1:2^n_init
       
        subset_idx = logical(bitget(combo-1, n_init:-1:1)); % Get the "0-1" vector representing a subset of A_t

        subset = A_t(subset_idx);
        subset_c = A_t( logical(1-subset_idx) );

        alpha_t(combo,:) = prod([ pi_p(subset); pi_0(subset_c) ]);

        if combo == 1
            m_t(combo,:) = 0;
        else
            m_t(combo,:) = max(r(subset));
        end
    end

end

t = 1;

while true
     
    fprintf("------------------------------------\n")
    fprintf("Step: %d \n", t)

    %% We only add elements if A_t is empty
    if isempty(A_t)

        Inc_add = K_3DP*pi_p.*r + TM_Delta; % Possible increment by adding each possible new item

        if sum(Inc_add>0) > 0

            %% If we can benefit from adding one element
            if input.add_max_pivot_rule == 1
                % We select the one with the maximal increaase 
                [~, idx_add] = max(Inc_add);
            else
                % We select the first one with positive increase
                tmp = (Inc_add>0);
                pos_idx = one2n(tmp);
                idx_add = pos_idx(1);
            end
    
            A_t = idx_add;
            A_tc = setdiff(one2n, A_t);
            alpha_t = [pi_0(idx_add), pi_p(idx_add)]';
            m_t = [0, r(idx_add)]';

            % input_check.B_t = B_t; input_check.pi_p = pi_p(A_t); input_check.pi_0 = pi_0(A_t); input_check.r = r(A_t);
            % output_check = check_bookkeeping(input_check);
            % fprintf("Bookkeeping err: %3.4f \n\n", norm(output_check.alpha-alpha_t)+norm(output_check.m-m_t))
            
            t = t + 1;

            fprintf("Added index %d, Current size of 3DP set: %d \n", idx_add, length(A_t))
            fprintf("------------------------------------------------------------------------\n\n")

        else

            %% If we don't benefit from adding one element, we terminate
            fprintf("Can't improve by adding or deleting an element!!! (Final 3DP set size: %d) \n", length(A_t))
            fprintf("------------------------------------------------------------------------\n\n")
            break            

        end

    %% When A_t is neither empty nor full
    elseif length(A_t) < num_suppliers

        %% We first check if we should add any element
        % Possible increment by adding one more item to current 3DP set A_t
        Inc_add = sum(K_3DP*(pi_p(A_tc) * alpha_t') .* max(0, r(A_tc) - m_t'), 2) + TM_Delta(A_tc);
        Inc_add_pos_idx = A_tc(Inc_add>0);

        % We only add element if any of them will improve the cost
        if ~isempty(Inc_add_pos_idx)

            if input.add_max_pivot_rule == 1
                %% We select the one with the maximal increaase 
                [~, idx_add_relative] = max(Inc_add);
                idx_add = A_tc(idx_add_relative);
            else
                %% We select the first one with positive increase
                idx_add = Inc_add_pos_idx(1);
            end
        
            % We always add the new element to the first place
            A_t = [idx_add, A_t];
            A_tc = setdiff(A_tc, idx_add);
    
            % Update alpha_t and m_t
            alpha_t = [ pi_0(idx_add)*alpha_t ; pi_p(idx_add)*alpha_t ];
            m_t = [ m_t ; max(r(idx_add), m_t)];

            t = t + 1;

            fprintf("Added index %d, Current size of 3DP set: %d \n", idx_add, length(A_t))
            fprintf("------------------------------------------------------------------------\n\n")

        end

        %% We then check if we could delete any element from A_t if we can't add any new element
        if isempty(Inc_add_pos_idx)
            
            Inc_del = [];
            idx_del = inf;

            for l = 1 : length(A_t)
                
                % The absolute index of the (l-th) item we propose to delete from current 3DP set A_t
                idx2_del = A_t(l); 

                % Get all the subsets of A_t containing l and not containing l
                subsets_with_l = get_indices(length(A_t), l, 1);
                subsets_wihtout_l = get_indices(length(A_t), l, 0);
                % subsets_wihtout_l = setminue([1:2^length(A_tc)], subset_with_l);

                % The possible increment if this item is deleted
                Inc_del(l) = -K_3DP * sum( alpha_t(subsets_with_l) .* max(0, r(idx2_del) - m_t(subsets_wihtout_l)) ) - TM_Delta(idx2_del);
                
                % If our pivoting rule is to pop out the first index with positive gain      
                if (input.delete_max_pivot_rule == 0) && (Inc_del(l) > 0)
                    idx_del = idx2_del;
                    idx_del_relative = l;
                    break
                end

            end

            if (input.delete_max_pivot_rule == 1) && (sum(Inc_del>0) > 0)
                % We pop out the index with maximal positive gain
                [~, idx_del_relative] = max(Inc_del);
                idx_del = A_t(idx_del_relative);
            end
            
            if ~isinf(idx_del)

                %% If we can benefit from popping out an element we update
                % Pop out this element from A_t
                A_t(idx_del_relative) = [];
                A_tc = setdiff(one2n, A_t);

                % Update vectors alpha_t, m_t
                null_rows = get_indices(length(A_t)+1, idx_del_relative, 0);
                alpha_t = alpha_t(null_rows)/pi_0(idx_del);
                m_t = m_t(null_rows);
                
                t = t + 1;

                fprintf("Deleted index %d, Current size of 3DP set: %d \n", idx_del, length(A_t))
                fprintf("------------------------------------------------------------------------\n\n")

            else

                fprintf("Can't improve by adding or deleting an element!!! (Final 3DP set size: %d) \n", length(A_t))
                fprintf("------------------------------------------------------------------------\n\n")
                break

            end

        end


    %% When A_t is full we only delete element
    else

        Inc_del = [];
        idx_del = inf;

        for l = 1 : length(A_t)
            
            % The absolute index of the (l-th) item we propose to delete from current 3DP set A_t
            idx2_del = A_t(l); 
            
            % Get all the subsets of A_t containing l and not containing l
            subsets_with_l = get_indices(length(A_t), l, 1);
            subsets_wihtout_l = get_indices(length(A_t), l, 0);

            % The possible increment if this item is deleted
            Inc_del(l) = -K_3DP * sum( alpha_t(subsets_with_l) .* max(0, r(idx2_del) - m_t(subsets_wihtout_l)) ) - TM_Delta(idx2_del);

            % If our pivoting rule is to pop out the first index with positive gain      
            if (input.delete_max_pivot_rule == 0) && (Inc_del(l) > 0)
                idx_del = idx2_del;
                idx_del_relative = l;
                break
            end

        end

        if (input.delete_max_pivot_rule == 1) && (sum(Inc_del>0) > 0)
            % We pop out the index with maximal positive gain
            [~, idx_del_relative] = max(Inc_del);
            idx_del = A_t(idx_del_relative);
        end
        
        if ~isinf(idx_del)

            %% If we can benefit from popping out an element we update
            % Pop out this element from A_t
            A_t(idx_del_relative) = [];
            A_tc = setdiff(one2n, A_t);

            % Update vectors alpha_t, m_t
            null_rows = get_indices(length(A_t)+1, idx_del_relative, 0);
            alpha_t = alpha_t(null_rows)/pi_0(idx_del);
            m_t = m_t(null_rows);            

            t = t + 1;

            fprintf("Deleted index %d, Current size of 3DP set: %d \n", idx_del, length(A_t))
            fprintf("------------------------------------------------------------------------\n\n")
            
        else

            fprintf("Can't improve by adding or deleting an element!!! (Final 3DP set size: %d) \n", length(A_t))
            fprintf("------------------------------------------------------------------------\n\n")
            break

        end        
        
    end

        
end

output.A_t = sort(A_t);
output.A_tc = sort(A_tc);

endTime = clock;
output.solving_time = etime(endTime, startTime);



%% NOTE ON POST-PROCESSING
%% Our original problem is: 
%%      min_{A, K} U3DP(A,K) + C3DP(K)I{K>0} + UTM(A^c) + CTM(A^c)

%% When K = 0 (baseline), let the optimal solution be A0 (proudcts not backedup at all)
%% We rewrite the problem as:
%%      min {     min_{A, K>0} U3DP(A,K) + C3DP(K) + UTM(A^c) + CTM(A^c),   U3DP(A0,0) + UTM(A0^c) + CTM(A0^c)    }

%% Optimizing A and K jointly is hard, we typically do a grid search on K; namely, we solve the original problem in the following way
%%     min_{K>0}  min{     Term 1,   Term 2    }
%% where
%%     - Term 1  =  min_{A} U3DP(A,K) + C3DP(K) + UTM(A^c) + CTM(A^c)
%%     - Term 2  =  U3DP(A0,0) + UTM(A0^c) + CTM(A0^c)

%% The solution we obtain sofar (A_t) should be understood as the optimizer of Term 1 (REGARDLESS OF C3DP!!!)
%% We can also do a little further and post-process to get the solution of min{     Term 1,   Term 2    }
%% Namley:
%%     - First compute the optimal value of Term 1 under the current A_t (DEPENDS ON C3DP!!!)
%%     - Compare this to Term 2
%%     - If Term 1 is better than Term 2, then just use A_t
%%     - If Term 2 is better than Term 1, then A0 is the current solution

%% Before such comparison, note that we don't have the objective value for comparison (Term 1 vs. Term 2), we don't even have the "q part" for evaluation
%% So we would have to solve the "fixed supplier selection" problem using:
%%  - input.recompute_flag == 1: GRB (Full Info. or SAA)
%%  - input.recompute_flag == 2: SGD


speed_per_machine_month = input.speed_per_machine_month;
cost_of_3dp_per_machine_month = input.cost_of_3dp_per_machine_month;
TM_cost = input.TM_cost;
nobackup_cost = input.nobackup_cost;

x = zeros(num_suppliers, 1);
x(A_t) = 1;
x = logical(x);

x_zero_indicator = (sum(x) > 0);

%% "TOTAL_COST":              records the value of min( Term 1, Term 2 )
%% "TOTAL_COST_NONZERO":      records the value of Term 1
%% Q_FINAL, X_FINAL, A_FINAL: records the solution after comparing Term 1 and 2
%% NOTE: Technically if Term 2 is better, the solution should be 3DP set = A0, and K = 0 ("3DP set" = "Nobackup set" if K=0)
%% To distinguish this with regular cases when K>0 (so A_t is really "3DP set"):
%%      - in "X_FINAL", we won't denote elements in A0 with 1, but with Inf
%%      - in "Q_FINAL", we just let them be zero
%%      - in "A_FINAL", we let it be [] 

Q_FINAL = {}; X_FINAL = {}; A_FINAL = {};
TOTAL_COST = [];
TOTAL_COST_NONZERO = [];


if input.auto_recompute == 1
    
    if sum(x) < 5

        %% For |A_t|>5, we can still apply full info GRB
        input.recompute_flag = 1;
        input.recompute_sample_mode = 1;
        input.recompute_disruption_sample_flag = 0;
        input.recompute_demand_sample_flag = 0;  

    elseif sum(x) <= 12
        
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

    supplier_3dp_select = x;

    p = input.p;
    yield_loss_rate = input.yield_loss_rate;
    num_suppliers = input.num_suppliers;
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
                X_FINAL{i,j} = x; 
                A_FINAL{i,j} = A_t;
                q_SP = zeros(num_suppliers,1); q_SP(set_of_3DP) = output_recompute.q_SP;
                Q_FINAL{i,j} = q_SP; 

            else
                %% If K=0 is better, we output A0 (should be understood as without backup)
                %%      - In X_FINAL, all items in A0 are denoted with INF
                %%      - In Q_FINAL, we let everything be zero
                %%      - We let A_FINAL be empty
                TOTAL_COST(i,j) = sum(min(TM_cost, nobackup_cost));
                X_FINAL{i,j} = zeros(num_suppliers,1); X_FINAL{i,j}(nobackup_cost<TM_cost) = Inf;
                A_FINAL{i,j} = [];
                Q_FINAL{i,j} = zeros(num_suppliers,1);

            end

            TOTAL_COST_NONZERO(i,j) = recomputed_total_cost(i,j);

        end
    end


    %% Note that the key of our BOE method is the BOE approximation of U3DP (excluding sum_j v_jmu_j), we check how accurate it is
    if x_zero_indicator > 0
        U3DP_BOE = sum(U0(A_t)) - K_3DP*sum(alpha_t.*m_t);
        U3DP_benchmark = output_recompute.opt_val;
        err_BOE = (U3DP_BOE-U3DP_benchmark)/U3DP_benchmark;
        err_BOE2 = (U3DP_BOE-U3DP_benchmark)/(U3DP_benchmark-sum(mean_demand(set_of_3DP).*v(set_of_3DP)));
    end

    
else

    %% Using SGD: We take the current supplier selection (x_best <=> "set_of_3DP") and recompute "q part"
    %% Also compute the objective: U3DP("set_of_3DP", K) + UTM("set_of_TM", K) + C3DP(K)
    
    %% Some important specifications of running SGD here:
    %% - Do we evalute in-process and how much data do we use to do so?
    %% - How much data do we use to evaluate the final solution

    supplier_3dp_select = (x > 1e-3);

    p = input.p;
    yield_loss_rate = input.yield_loss_rate;
    num_suppliers = input.num_suppliers;
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
    input_sgd.q_init = unifrnd(input_sgd.q_lb, input_sgd.q_ub);

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


    %% We compare Term 1 to Term 2, where
    %% Term 1 <=> (approx.) optimal value under the given K>0
    %% Term 2 <=> optimal value when K=0
    for i = 1 : length(speed_per_machine_month)  
        for j = 1 : length(cost_of_3dp_per_machine_month)

            if recomputed_total_cost(i,j) <  sum(min(TM_cost, nobackup_cost))

                TOTAL_COST(i,j) = recomputed_total_cost(i,j);
                X_FINAL{i,j} = (x>1e-3); 
                A_FINAL{i,j} = A_t;
                q_best_update = zeros(num_suppliers,1); q_best_update(set_of_3DP) = output_sgd.q_ave_final;
                Q_FINAL{i,j} = q_best_update; 

            else
                %% If K=0 is better, we output A0 (should be understood as without backup)
                %%      - In X_FINAL, all items in A0 are denoted with INF
                %%      - In Q_FINAL, we let everything be zero
                %%      - We let A_FINAL be empty
                TOTAL_COST(i,j) = sum(min(TM_cost, nobackup_cost));
                X_FINAL{i,j} = zeros(num_suppliers,1); X_FINAL{i,j}(nobackup_cost<TM_cost) = Inf;
                A_FINAL{i,j} = [];
                Q_FINAL{i,j} = zeros(num_suppliers,1);

            end

            TOTAL_COST_NONZERO(i,j) = recomputed_total_cost(i,j);

        end
    end

    %% Note that the key of our BOE method is the BOE approximation of U3DP (excluding sum_j v_jmu_j), we check how accurate it is
    if x_zero_indicator > 0
        U3DP_BOE = sum(U0(A_t)) - K_3DP*sum(alpha_t.*m_t);
        U3DP_benchmark = obj_fullinfo_sgd;
        err_BOE = (U3DP_BOE-U3DP_benchmark)/U3DP_benchmark;
        err_BOE2 = (U3DP_BOE-U3DP_benchmark)/(U3DP_benchmark-sum(mean_demand(set_of_3DP).*v(set_of_3DP)));
    end

end
  




endTime = clock;
output.recompute_time = etime(endTime, startTime);

output.total_time = output.recompute_time + output.solving_time;

output.TOTAL_COST = TOTAL_COST;
output.TOTAL_COST_NONZERO = TOTAL_COST_NONZERO;
output.X_FINAL = X_FINAL;
output.A_FINAL = A_FINAL;
output.Q_FINAL = Q_FINAL;

if x_zero_indicator > 0
    output.U3DP_BOE = U3DP_BOE;
    output.U3DP_benchmark = U3DP_benchmark;
    output.err_BOE = err_BOE;
    output.err_BOE2 = err_BOE2;
end



end
