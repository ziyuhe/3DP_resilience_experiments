function output = V3DP_b2b(input)

x = input.x;
c_3DP = input.c_3DP;
v = input.v;
h = input.h;
weight = input.weight;
K_3DP = input.K_3DP;
D_bar = input.D_bar;

num_suppliers = length(x);

% For the following two cases, the optimal q_3DP is trivially 0
%   - For all j s.t. x_j = 0
%   - For all j s.t. x_j = 1 and Dbar_j <= 0
q_3DP = zeros(length(x),1);

% Now focus on all j s.t. x_j = 1 and Dbar_j > 0
IDX = logical(x .* (D_bar>1e-5));

if sum(D_bar(IDX).*weight(IDX)) < K_3DP

    % If K_3DP is enough to handle all unmet demand, then the solution is trivially q3DP_j = Dbar_j
    q_3DP(IDX) = D_bar(IDX);

    one2n = [1:num_suppliers];
    A_full = one2n(IDX); 
    A_null = [];
    j_star = inf;

else

    % Otherwise, perform the bang-to-buck ratio method
    % Sort the bang-to-buck ratio
    [~, tmp] = sort( (v(IDX) - c_3DP(IDX))./weight(IDX), 'descend' );

    one2n = [1:num_suppliers];
    abs_idx = one2n(IDX);           % the absolute indices of products with positive unmet demand 
    sorted_abs_idx = abs_idx(tmp);  % the absolute indices of products with positive unmet demand in order of b2b ratio (from big to small)
    
    leftover_K = K_3DP;
    A_full = [];        % Collects the absolute indices of unmet demand fully fullfilled
    A_null = [];        % Collects the absolute indices of unmet demand not fullfilled at all

    for j = 1 : length(sorted_abs_idx)
        
        current_idx = sorted_abs_idx(j);

        if (leftover_K > 0) && (D_bar(current_idx)*weight(current_idx) < leftover_K+1e-5)

            % When the leftover capacity is enough for the current unmet demand
            q_3DP(current_idx) = D_bar(current_idx);
            leftover_K = leftover_K - D_bar(current_idx)*weight(current_idx);
            A_full = [A_full, current_idx];

        elseif (leftover_K > 0) && (D_bar(current_idx)*weight(current_idx) > leftover_K+1e-5)

            % When the leftover capacity is positive but not enough for the current unmet demand
            q_3DP(current_idx) = leftover_K/weight(current_idx);
            j_star = current_idx;
            A_null = setdiff(abs_idx, [A_full,j_star]);
            break

        end

    end

end


opt_val = sum( x .* (c_3DP.*q_3DP + v.*max(0,D_bar-q_3DP) + h.*max(0,q_3DP-D_bar)) );

output.opt_val = opt_val;
output.q_3DP = q_3DP;
output.A_full = A_full;
output.j_star = j_star;
output.A_null = A_null;

