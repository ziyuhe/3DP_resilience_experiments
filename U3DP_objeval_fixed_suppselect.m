function opt_val = U3DP_objeval_fixed_suppselect(input)

%% In this function, we evaluate the objective value of U3DP under a fixed supplier selection
% NOTE: here the U3DP can be evaluated based on samples

startTime = clock;

q_eval = input.q_eval; % the solution we evaluate

n = input.n; % number of products
c_3DP = input.c_3DP; 
v = input.v;
h = input.h;
weight = input.weight; % per unit weight of printing materials for each product
K_3DP = input.K_3DP;

mean_demand = input.mean_demand;


S = input.S;                           % number of samples
prob_scenarios = input.prob_scenarios; % probability vector of all combinations of disruption and demand scenarios 
D_scenarios = input.D_scenarios;       % all demand scenarios (n by S)
s_scenarios = input.s_scenarios;       % all disruption scenarios (n by S)


input_objeval.num_suppliers = n;
input_objeval.c_3DP = c_3DP;
input_objeval.v = v;
input_objeval.h = h;
input_objeval.weight = weight;
input_objeval.mean_demand = mean_demand;
input_objeval.K_3DP = K_3DP;


V_3DP_all_scenarios = []; 

for i = 1 : S
    
    if (input.display_flag == 1) && (mod(i,input.display_interval) == 1)
        fprintf("evaluating sample %d (out of %d, finished %3.2f%%)\n", i, S, i/S*100)
    end

    input_objeval.q = q_eval;
    input_objeval.s = s_scenarios(:,i);
    input_objeval.D = D_scenarios(:,i);
    input_objeval.D_bar = input_objeval.D - q_eval.*input_objeval.s;
    
    input_objeval.obj_only = 1;
    output_objeval = V3DP_b2b_dual_fixed_suppselect(input_objeval);

    V_3DP_all_scenarios(i,:) = output_objeval.opt_val_primal;
    
end

%% NOTE THIS IS NOT U3DP IN OUR CASE (THIS = U3DP + sum(v.*demand))
opt_val = prob_scenarios'*V_3DP_all_scenarios;

endTime = clock;

fprintf("Time: %3.2f\n", etime(endTime, startTime))


end