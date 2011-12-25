%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CS 229, Autumn 2007-08
% Gaussian processes demo (Chuong Do)
%
% This code is known to work properly with Octave 2.9.17/GnuPlot 2.4
% and MATLAB 7.5.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% check if running Octave

v = version;
is_octave = v(1) < '5'; 

% define input domain

x_min = 0;
x_step = 0.1;
x_max = 10;
X = [x_min:x_step:x_max]';

% define kernel bandwidth

tau = 1.0;
k = @(x,z) exp(-(x-z)'*(x-z) / (2*tau^2));

% define level of Gaussian noise

sigma = 0.1;

while (true)
  
    disp(' ');
    disp('CS 229 Gaussian processes demo');
    disp('------------------------------');
    disp(' ');
    disp(sprintf('1.  Set kernel bandwidth (currently tau = %f)', tau));
    disp(sprintf('2.  Set noise level (currently sigma = %f)', sigma));
    disp('3.  Show samples from a Gaussian process prior.');
    disp('4.  Create a random regression problem.');
    disp('5.  Run Gaussian process regression.');
    disp('6.  Add more training data.');
    disp(' ');
    disp('Q.  Quit');
    disp(' ');
  
    reply = lower(input('Select a menu option: ', 's'));

    switch (reply)

        % (1) set noise level
        
        case {'1'}
            
            tau = max(1e-8, input('Enter desired kernel bandwidth: '));
            k = @(x,z) exp(-(x-z)'*(x-z) / (2*tau^2));
            
        % (2) set noise level
        
        case {'2'}
            
            sigma = max(1e-8, input('Enter desired noise level: '));
            
        % (3) show samples from a Gaussian process prior

        case {'3'}
            
            num_samples = 3;
            h = zeros(size(X,1), num_samples);
            for i=1:num_samples
                h(:,i) = sample_gp_prior(k,X);
            end
            newplot;
            plot(repmat(X,1,num_samples), h, 'LineWidth', 3);
            title(sprintf('Samples from GP with k(x,z) = exp(-||x-z||^2 / (2*tau^2)), tau = %f', tau));
            
        % (4) create a random regression problem
 
        case {'4'}

            % pick true function to be estimated
            
            h = sample_gp_prior(k,X);
            y_min = min(h) - 1;
            y_max = max(h) + 1;
            
            % create noise-corrupted training set

            m_train = 10;
            i_train = floor(rand(m_train,1) * size(X,1) + 1);
            X_train = X(i_train);
            y_train = h(i_train) + sigma * randn(m_train,1);

            newplot;
            hold on;
            plot(X, h, 'k', 'LineWidth', 3);
            plot(X_train, y_train, 'rx', 'LineWidth', 5);
            axis([x_min x_max y_min y_max]);
            title(sprintf('Random regression problem with N(0,sigma^2) noise, sigma = %f', sigma));
            hold off;
            
        % (5) run Gaussian process regression
        
        case {'5'}
            
            X_test = X;
            K_test_test = compute_kernel_matrix(k,X_test,X_test);
            K_test_train = compute_kernel_matrix(k,X_test,X_train);
            K_train_train = compute_kernel_matrix(k,X_train,X_train);
            G = K_train_train + sigma^2 * eye(size(X_train,1));
            mu_test = K_test_train * (G \ y_train);
            Sigma_test = K_test_test + sigma^2 * eye(size(X_test,1)) - K_test_train * (G \ (K_test_train'));
            stdev_test = sqrt(diag(Sigma_test));
            
            newplot;
            hold on;
            if (is_octave)
                errorbar(X_test, mu_test, stdev_test, '~');
            else
                lower_test = mu_test - 2*stdev_test;
                upper_test = mu_test + 2*stdev_test;
                region_X = [X_test; X_test(end:-1:1,:)];
                region_Y = [lower_test; upper_test(end:-1:1,:)];
                fill(region_X, region_Y, 'g');
            end
            plot(X, h, 'k', 'LineWidth', 3);
            plot(X_train, y_train, 'rx', 'LineWidth', 5);
            plot(X_test, mu_test, 'b', 'LineWidth', 3);
            axis([x_min x_max y_min y_max]);
            title('Gaussian process regression, 95% confidence region');
            hold off;
            
        % (6) add more training data
        
        case {'6'}
        
            % add more noise-corrupted points to training set
            
            m_train = size(X_train,1);
            i_train = floor(rand(m_train,1) * size(X,1) + 1);
            X_train = [X_train; X(i_train)];
            y_train = [y_train; h(i_train) + sigma * randn(m_train,1)];

            newplot;
            hold on;
            plot(X, h, 'k', 'LineWidth', 3);
            plot(X_train, y_train, 'rx', 'LineWidth', 5);
            axis([x_min x_max y_min y_max]);
            title('Random regression problem');
            hold off;            
        
        % (Q) quit
        
        case {'q'}
            
            break;
    end
end

