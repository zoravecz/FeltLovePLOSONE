%% This code fits an Extended Condorcet Model on data from the love study described in the ms submitted to PLOS One
%% If you don't want to run the model just inspect the results go to line 212
%% You need JAGS installed on your computer to be able to run this code

% Cleanup first
clear all
close all
tic
proj_id = mfilename;

% we use trinity to interface MATLAB with JAGS -- trinity files provided in
% the same folder as this file on GitHub -- 
% trinity has to be installed every time you start a new MATLAB session
% addpath(genpath(cd('c:path for trinity folder')))
% install_trinity

% set number of workers for parallel computing
% also set line in callbayes from  'parallel',  0 ,  to  'parallel' , 1
% myCluster = parcluster('local');
% myCluster.NumWorkers = 8;   

%% First, load data
fullDataMatrix = importdata('MTurkdata.csv');

TFDK = fullDataMatrix.data(:,10:62);  % Extract True/False/Don't know

P = size(TFDK, 1);  % Number of respondents
I = size(TFDK, 2);  % Number of items
  
indexItem    = reshape(repmat(1:I, P, 1), [], 1);  % Vector of item indices
indexPerson  = reshape(repmat(1:P, 1, I), [], 1);  % Vector of person indices
Y            = TFDK(:);  % Make vector of TFDK responses

% Flag missing data and remove from data vectors
missing  = isnan(Y);
indexItem(missing)   = [];
indexPerson(missing) = [];
Y(missing)           = [];

N  = size(Y, 1);  % Total number of data points

% Some covariates, separate by parameter
standardize = @(x)(x-mean(x))./std(x);

intercept     = ones(P, 1);
gender        = fullDataMatrix.data(:,1)==1;
relationship = fullDataMatrix.data(:,3);
% original scale recoding
% into in a relationship or not
%1: Married --  stays 1
relationship(relationship == 2) = 1; % recode cohabiting
relationship(relationship == 6) = 1;  % recode in stable relationship (but not married/cohabiting)
relationship(relationship == 3) = 0;
relationship(relationship == 4) = 0;
relationship(relationship == 5) = 0;
relationship(relationship == 7) = 0;
relationship(relationship == 8) = 0;
relationship(relationship == 9) = 0;
age           = standardize(fullDataMatrix.data(:,7));
familyMembers = standardize(fullDataMatrix.data(:,8));
siblings      = standardize(fullDataMatrix.data(:,9));
gender        = standardize(gender);
relationship  = standardize(relationship);

% matrix of explanatory variables on which ECM parameters are regressed
bigX = [intercept gender relationship age familyMembers siblings];

Xdelta = zeros(I, 1);  % no item covariates

Xg     = bigX; % creating explanatory variable matrix for guessing bias
Xb     = bigX; % creating explanatory variable matrix for willingness to guess
Xtheta = bigX; % creating explanatory variable matrix for ability

% counting nr of explanatory variables for each parameter
nrofthetacov = size(Xtheta, 2); 
nrofgcov     = size(Xg, 2);
nrofbcov     = size(Xb, 2);
nrofdeltacov = size(Xdelta, 2);

% data variable input for JAGS
data = struct('Y', Y, 'N', N, ...
    'person', indexPerson, ...  'I', indexPerson, ...
    'item', indexItem, ...      'K', indexItem, ...
    'nP', P, ...                'P', P, ...
    'nI', I, ...                'Items', I, ...
    'AS_MATRIX_XT', Xtheta, ...          
    'AS_MATRIX_XG', Xg, ...
    'AS_MATRIX_XB', Xb, ...
    'AS_MATRIX_Xdelta', Xdelta, ...
    'nCT', nrofthetacov, ...    
    'nCB', nrofbcov, ...        
    'nCG', nrofgcov, ...
    'nCD', nrofdeltacov); 

%% First, make all inputs that Trinity needs
% Write the JAGS model into a variable (cell variable)
model = {
    'model {'
    '   for (l in 1:N) {'
    '        D[l]     <- ilogit(theta[person[l]] - delta[item[l]])'
    '        piY[1,l] <- b[person[l]] * g[person[l]] + D[l] * Z[item[l]]'
    '                    - D[l] * b[person[l]] * g[person[l]]'
    '        piY[2,l] <- D[l] - D[l] * Z[item[l]]'
    '                    + b[person[l]] * (D[l] - 1) * (g[person[l]] - 1)'
    '        piY[3,l] <- 1 - (piY[1,l] + piY[2,l])'
    '        Y[l]     ~ dcat(piY[,l])'
    '   }'
    ''
    '    for (i in 1:nP){'
    '        muTGB[i,1] <- sum(betaT * XT[i,])'
    '        muTGB[i,2] <- sum(betaG * XG[i,])'
    '        muTGB[i,3] <- sum(betaB * XB[i,])'
    ''
    '        TGB[i,1:3] ~ dmnorm(muTGB[i,1:3], tauTGB[1:3,1:3])'
    ''
    '        theta[i] <- TGB[i,1]'
    '        g[i]     <- ilogit(TGB[i,2])'
    '        b[i]     <- ilogit(TGB[i,3])'
    '    }'
    ''
    '    for (k in 1:nI){'
    '        Z[k] ~ dbern(0.5)'
    '        mudelta[k] <- sum(betadelta * Xdelta[k,])'
    '        delta[k] ~ dnorm(mudelta[k], taudelta)'
    '    }'
    ''
    '    sigmadelta ~ dunif(0.001, 10)'
    '    taudelta <-  pow(sigmadelta, -2.0)'
    ''
    '    for (cov in 1:nCT) {'
    '        betaT[cov] ~ dnorm(0, 0.01)'
    '    }'
    '    for (cov in 1:nCB) {'
    '        betaB[cov] ~ dnorm(0, 0.01)'
    '    }'
    '    for (cov in 1:nCG) {'
    '        betaG[cov] ~ dnorm(0, 0.01)'
    '    }'
    '    for (cov in 1:nCD) {'
    '        betadelta[cov] ~ dnorm(0, 0.01)'
    '    }'
    ''
    '    sigmaT ~ dunif(0.001, 10)'
    '    sigmaG ~ dunif(0.001, 10)'
    '    sigmaB ~ dunif(0.001, 10)'
    '    rhoTG  ~ dunif(-1, 1)'
    '    rhoTB  ~ dunif(-1, 1)'
    '    rhoBG  <- rhoTB * rhoTG'
    ''
    '    covTGB[1,1] <- sigmaT * sigmaT'
    '    covTGB[2,2] <- sigmaG * sigmaG'
    '    covTGB[3,3] <- sigmaB * sigmaB'
    '    covTGB[1,2] <- rhoTG * sigmaT * sigmaG'
    '    covTGB[1,3] <- rhoTB * sigmaT * sigmaB'
    '    covTGB[2,3] <- rhoBG * sigmaB * sigmaG'
    '    covTGB[2,1] <- covTGB[1,2]'
    '    covTGB[3,1] <- covTGB[1,3]'
    '    covTGB[3,2] <- covTGB[2,3]'
    ''
    '    tauTGB[1:3,1:3] <- inverse(covTGB[1:3,1:3])'
    '}'
    };

% List all the parameters of interest (cell variable)
parameters = {
    'theta'  'g'      'b'      'delta' ...
    'betaT'  'betaG'  'betaB'  'betadelta' ...
    'sigmaT' 'sigmaG' 'sigmaB' 'sigmadelta' ...
    'rhoTG' 'rhoTB' 'rhoBG' 'Z'
    };

% Write a function that generates a structure with one random value for
% each random parameter in a field
generator = @()struct( ... 
    'TGB', [randn(P,1) rand(P,1) rand(P,1)], ...
    'sigmaT', rand(1,1) + 0.1, ...
    'sigmaG', rand(1,1) + 0.1, ...
    'sigmaB', rand(1,1) + 0.1, ...
    'rhoTG', randn(1,1)/100, ...
    'rhoTB', randn(1,1)/100 ...
    );

% Tell Trinity which engine to use
engine = 'jags';


%% Run Trinity with the CALLBAYES() function


[stats, chains, diagnostics, info] = callbayes(engine, ...
    'model'          ,     model , ...
    'data'           ,      data , ...
    'outputname'     , 'samples' , ...
    'init'           , generator , ...
    'modelfilename'  ,   proj_id , ...
    'datafilename'   ,   proj_id , ...
    'initfilename'   ,   proj_id , ...
    'scriptfilename' ,   proj_id , ...
    'logfilename'    ,   proj_id , ...
    'nchains'        ,         8 , ...
    'nburnin'        ,      1000 , ...
    'nsamples'       ,      1000 , ...
    'monitorparams'  ,   parameters  , ...
    'thin'           ,         4 , ...
    'refresh'        ,     1000  , ...
    'workingdir'     ,    ['/tmp/' proj_id]  , ...
    'cleanup'        ,    false  , ...
    'verbosity'      ,        5  , ...
    'saveoutput'     ,     true  , ...
    'parallel'       ,         0 , ...
    'modules'        ,  {'dic'}  );

fprintf('%s took %f seconds!\n', upper(engine), toc)

%% Save .mat file

save(proj_id)
%% Posterior Predictive Check

[eigenR,  dataeigenR, sim] = PPC_one_culture(TFDK, chains,500);

savefig('modelfit')
saveas(gcf,'modelfit.jpg')

%% Get some summary statistics
%% if you don't want to run the analysis just browse the results, load the 'ECM_MTurkStudy.mat' file
%% if you want to use the functions below on them, you have to install trinity 
%% by downloading it from here http://sourceforge.net/projects/matlab-trinity/
%% then install_trinity and adding it to the path by addpath(genpath('yourpath'))

%% First, inspect convergence
if any(codatable(chains, @gelmanrubin) > 1.1)
    grtable(chains, 1.1)
    warning('Some chains were not converged!')
else
    disp('Convergence looks good.')
end

%% Now check some basic descriptive statistics averaged over all chains
% stats on the consensus answers
codatable(chains, 'Z', @median, @mean, @std);
% item difficulty estimates
codatable(chains, '^delta',  @mean);

temp = codatable(chains, '^delta',  @mean);

%% plots

ability = codatable(chains, '^theta',  @mean);
item_difficulty = codatable(chains, '^delta',  @mean);
guessing_bias = codatable(chains, '^g_',  @mean);
willingness_toguess = codatable(chains, '^b_',  @mean);

close all
subplot(2,2,1)
nhist(ability,'color','summer')
xlabel('Person-specific ability', 'Fontweight', 'bold', 'Fontsize', 15)
set(gca,'fontWeight','bold', 'Fontsize', 14, 'Xtick',[-4 -2 0 2 4])
ylabel('Number of participants')
subplot(2,2,4)
nhist(guessing_bias,'color','summer')
xlabel('Person-specific guessing bias', 'Fontweight', 'bold', 'Fontsize', 15)
set(gca,'fontWeight','bold', 'Fontsize', 14, 'Xtick',[0 0.5 1])
ylabel('Number of participants')
subplot(2,2,3)
nhist(willingness_toguess,'color','summer')
xlabel('Person-specific willingness to guess','Fontweight', 'bold', 'Fontsize', 15)
set(gca,'fontWeight','bold', 'Fontsize', 14, 'Xtick',[0 0.5 1])
ylabel('Number of participants')
subplot(2,2,2)
nhist(item_difficulty,'color','summer')
xlabel('Item-specific item difficulty','Fontweight', 'bold', 'Fontsize', 15)
set(gca,'fontWeight','bold', 'Fontsize', 14, 'Xtick',[-4 -2 0 2 4])
ylabel('Number of participants')
%%

codatable(chains, 'beta')
%covariates (standardized ) --> [1:intercept 2: gender 3: in a relationship or not 4: age 5: family size 6: nr of siblings]
codatable(chains, 'sigma',  @mean,  @std)
