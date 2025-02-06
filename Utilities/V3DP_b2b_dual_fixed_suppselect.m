function output = V3DP_b2b_dual_fixed_suppselect(input)

%% Solve V3DP under the following configurations
%  - "sales-oriented" model
%  - fixed supplier selection ("x")
%  - fixed 3DP capacity K_3DP
%  - fixed primary order q_SP, disruption s, demand D (aggregated as D_bar = D-sq)

%% Apply "bang-to-buck" + KKT method to get the primal and dual solutions

num_suppliers = input.num_suppliers;
q = input.q;
c_3DP = input.c_3DP;
v = input.v;
h = input.h;
weight = input.weight;
K_3DP = input.K_3DP;
D_bar = input.D_bar;
D = input.D;
s = input.s;


% Initialize the primal solutions
q_3DP = zeros(num_suppliers,1);

% Initialize the subgradients
beta = zeros(num_suppliers,1);
rho = zeros(num_suppliers,1);


%% For all j s.t. Dbar_j <= 0
IDX_neg = (D_bar<1e-5);
% Subgradients
beta(IDX_neg) = s(IDX_neg).*h(IDX_neg);
rho(IDX_neg) = h(IDX_neg);



%% Now focus on all j s.t. Dbar_j > 0
IDX_pos = (D_bar>1e-5);

if sum(IDX_pos) == 0

    gamma = 0;
    A_full = [];
    A_null = [];
    j_star = inf;

else

    if sum(D_bar(IDX_pos).*weight(IDX_pos)) <= K_3DP + 1e-5   
    
        %% If K_3DP is enough to handle all unmet demand, then the solution is trivially q3DP_j = Dbar_j
        q_3DP(IDX_pos) = D_bar(IDX_pos);
        
        if input.obj_only == 0
            % Get the subgradient 
            gamma = 0;
            beta(IDX_pos) = -s(IDX_pos).*c_3DP(IDX_pos);
            rho(IDX_pos) = -c_3DP(IDX_pos);
            
            one2n = [1:num_suppliers];
            A_full = one2n(IDX_pos); 
            A_null = [];
            j_star = inf;
        end
    
    else 

        %% If capacity is not enough to fullfill all unmet demand
    
        % Otherwise, perform the bang-to-buck ratio method
        % Sort the bang-to-buck ratio
        [~, tmp] = sort( (v(IDX_pos) - c_3DP(IDX_pos))./weight(IDX_pos), 'descend' );
    
        one2n = [1:num_suppliers];
        abs_idx = one2n(IDX_pos);       % the absolute indices of products with positive unmet demand 
        sorted_abs_idx = abs_idx(tmp);  % the absolute indices of products with positive unmet demand in order of b2b ratio (from big to small)
        
        leftover_K = K_3DP;
        A_full = [];        % Collects the absolute indices of unmet demand fully fullfilled
        A_null = [];        % Collects the absolute indices of unmet demand not fullfilled at all
        j_star = inf;
    
        for j = 1 : length(sorted_abs_idx)
            
            current_idx = sorted_abs_idx(j);
    
            if (leftover_K > 1e-5) && (D_bar(current_idx)*weight(current_idx) < leftover_K+1e-5)
    
                % When the leftover capacity is enough for the current unmet demand
                q_3DP(current_idx) = D_bar(current_idx);
                leftover_K = leftover_K - D_bar(current_idx)*weight(current_idx);
                A_full = [A_full, current_idx];
    
            elseif (leftover_K > 1e-5) && (D_bar(current_idx)*weight(current_idx) > leftover_K+1e-5)
    
                % When the leftover capacity is positive but not enough for the current unmet demand
                q_3DP(current_idx) = leftover_K/weight(current_idx);
                j_star = current_idx;
                break
    
            end
    
        end
    
        if j_star < inf
            A_null = setdiff(abs_idx, [A_full,j_star]);
        else
            A_null = setdiff(abs_idx, A_full);
        end
        
        A_full = sort(A_full);
        A_null = sort(A_null);

        if input.obj_only == 0
    
            if j_star < inf
               
                % For the unique index j_star s.t. its unmet demand is only fullilled a portion
                % Get the subgradients
                gamma = -(v(j_star) - c_3DP(j_star))/weight(j_star);
                beta(j_star) = -s(j_star)*v(j_star);
                rho(j_star) = -v(j_star);
        
                % For indices j s.t. its unmet demand is all fullfilled
                tmp = gamma*weight(A_full) - c_3DP(A_full);
                % Get the subgradients
                beta(A_full) = s(A_full).*tmp;
                rho(A_full) = tmp;
        
                % For indices j s.t. none of its unmet demand is fullfilled
                % Get the subgradients
                beta(A_null) = -s(A_null).*v(A_null);
                rho(A_null) = -v(A_null);
        
            else
        
                gamma = - max( (v(A_null)-c_3DP(A_null))./weight(A_null) );
        
                % For indices j s.t. its unmet demand is fullfilled
                tmp = gamma*weight(A_full) - c_3DP(A_full);
                % Get the subgradients
                beta(A_full) = s(A_full).*tmp;
                rho(A_full) = tmp;
        
                % For indices j s.t. none of its unmet demand is fullfilled
                % Get the subgradients
                beta(A_null) = -s(A_null).*v(A_null);
                rho(A_null) = -v(A_null);
        
            end

        end
    
    end

end


opt_val_primal = sum( c_3DP.*q_3DP + v.*max(0,D_bar-q_3DP) + h.*max(0,q_3DP-D_bar) );
output.opt_val_primal = opt_val_primal;
output.q_3DP = q_3DP;


if input.obj_only == 0
    dual_cut_const = -D'*rho;
    
    opt_val_dual = gamma*K_3DP + dual_cut_const + q'*beta;
    
    output.opt_val_dual = opt_val_dual;
    
    output.A_full = A_full;
    output.j_star = j_star;
    output.A_null = A_null;
    
    output.beta = beta;
    output.rho = rho;
    output.dual_cut_const = dual_cut_const;
    output.gamma = gamma;
end



