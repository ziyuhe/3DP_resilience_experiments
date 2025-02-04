function output = Cost_No3DP_or_TM(input)

n = input.n; % number of products
if input.TM_flag == 0
    v_pen = input.v;
    v_sell = input.v;
    C_TM = 0;
else
    v_pen = input.c_TM;
    v_sell = input.v;
    C_TM = input.C_TM;
end
h = input.h;

p = input.p;
yield_loss_rate = input.yield_loss_rate;

Demand_atoms = input.Demand_atoms; % n-by-S, each row denotes all the atoms of a demand  
Demand_prob = input.Demand_prob; % n-by-S, each row denote the probabiliyt of the atoms of a demand
Demand_mean = input.Demand_mean;

opt_q = [];
opt_val = [];
for j = 1:n
    demand_atoms = Demand_atoms(j,:);
    demand_prob = Demand_prob(j,:);

    yield_atoms = [1-yield_loss_rate, 1];
    disruption_prob = [p,1-p];
    
    [V1, V2] = meshgrid(yield_atoms, demand_atoms);
    yield_comb = V1(:);
    demand_comb = V2(:);
    
    [P1_mesh, P2_mesh] = meshgrid(disruption_prob, demand_prob);
    joint_prob = P1_mesh(:) .* P2_mesh(:);

    S = length(joint_prob);
    num_vars = 1 + 2*S;

    obj_vec = [0; v_pen(j)*joint_prob; h(j)*joint_prob];
    
    tmp_A1 = [-yield_comb, -eye(S), zeros(S)];
    tmp_A2 = [yield_comb, zeros(S), -eye(S)];
    tmp_b1 = -demand_comb;
    tmp_b2 = demand_comb;

    A = sparse([tmp_A1;tmp_A2]);
    b = [tmp_b1;tmp_b2];

    model.modelname = 'optimization';
    model.modelsense = 'min';
    model.vtype = 'C'; % Continuous variables
    model.lb = zeros(num_vars, 1); % Lower bounds
    model.obj = obj_vec;
    model.A = A;
    model.rhs = b;
    model.sense = '<'; % All constraints are inequalities

    params.outputflag = 0; % Display output
    result = gurobi(model, params);
    opt_q(j,:) = result.x(1);
    opt_val(j,:) = result.objval - v_sell(j)*Demand_mean(j);

end



output.opt_q = opt_q;
if input.TM_flag == 0
    output.opt_val = opt_val;
    output.opt_val_total = sum(opt_val);
else
    output.opt_val = opt_val;
    output.opt_val_total = sum(opt_val);
    output.TM_cost = opt_val + C_TM;
    output.TM_cost_total = sum(output.TM_cost);
end



end