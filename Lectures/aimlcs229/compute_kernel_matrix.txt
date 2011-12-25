function [K] = compute_kernel_matrix(k, X, Z)
    
    m = size(X,1);
    n = size(Z,1);
    K = zeros(m,n);
    for i = 1:m
        for j = 1:n
            K(i,j) = k(X(i,:)', Z(j,:)');
        end
    end

end