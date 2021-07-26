function [net, varargout] = hebbRNN_learn_model(x0, net, F, perturbProb, eta, varargin)

% net = hebbRNN_learn_model(x0, net, F, perturbProb, eta, varargin)
%
% This function trains a recurrent neural network using reward-modulated
% Hebbian learning to produce desired outputs. During each trial the
% activations of random neurons are randomly perturbed. All fluctuations in
% the activation of each neuron are accumulated (supra-linearly) as an
% elegibility trace. At the end of each trial the error of the output is
% compared against the expected error and the difference is used to
% reinforce connectivity changes (net.J) that produce the desired output.
%
% The details of training the network are based on those
% presented in the following work:
% "Flexible decision-making in recurrent neural networks trained with a
% biologically plausible rule. Thomas Miconi (2016)"
% Published on BioRxiv. The current version can be found under the following URL:
% http://biorxiv.org/content/early/2016/07/26/057729
%
%
% INPUTS:
%
% x0 -- the initial activation (t == 0) of all neurons
% Must be of size: net.N x 1
%
% net -- the network structure created by hebbRNN_create_model
%
% F -- the desired output
% Must be a cell of size: 1 x conditions
% Each cell must be of size: net.B x time points
%
% perturbProb -- the probability of perturbing the activation of each neuron
% per second
%
% eta -- the learning rate
%
%
% OPTIONAL INPUTS:
%
% input -- the input to the network
% Must be a cell of size: 1 x conditions
% Each cell must be of size: net.I x time points
% Default: []
%
% targettimes -- the time points used to generate the error signal
% Default: entire trial
%
% beta -- the variance of the neural perturbation
% Default: 0.5. Don't change this unless you know what you're doing
%
% maxdJ -- the absolute connectivity values above this level will be
% clipped
% Default: 1e-4. Don't change this unless you know what you're doing
%
% alphaX -- the weight given to previous time points of the activation
% trace
% Default: 0. Don't change this unless you know what you're doing
%
% alphaR -- the weight given to previous time points of the error
% prediction trace
% Default: 0.33. Don't change this unless you know what you're doing
%
% targetFun -- the handle of a function that uses the firing rates of the
% output units to produce some desired output. Function must follow
% conventions of supplied default function.
% Default: @defaultTargetFunction
%
% targetFunPassthrough -- a user-defined structure that is automatically
% passed through to the targetFun, permitting custom variables to be passed
% Default: []
%
% tolerance -- at what error level below which the training will terminate
% Default: 0 (will train forever).
%
% batchType -- conditions are train either in random order each pass
% (pseudorand), or always in order (linear)
% Default: 'pseudorand'
%
% plotFun -- the handle of a function that plots information about the
% network during the learning process. Function must follow conventions
% of supplied default function.
% Default: @defaultPlottingFunction
%
% evalOpts -- a vector of size 2, specifying how much information should be
% displayed during training (0 - nothing, 1 - text only, 2 - text +
% figures), and how often the network should be evaluated. This vector is
% passed to the plotting function.
% Default: [0 50]
%
%
% OUTPUTS:
%
% net -- the network structure
%
% errStats -- the structure containing error information from learning
% (optional)
%
%
% Copyright (c) Jonathan A Michaels 2016
% German Primate Center
% jonathanamichaels AT gmail DOT com
%
% If used in published work please see repository README.md for citation
% and license information: https://github.com/JonathanAMichaels/hebbRNN


% Start counting
tic

% Variable output considerations
nout = max(nargout,1)-1;

% Variable input considerations
optargin = size(varargin,2);

inp = []; % Default inputs
targettimes = 1:size(F,1); % Default output times included in error
maxdJ = 1e-4; % Default clipping of connectivity changes
alphaX = 0; % Default weight of activation memory trace
alphaR = 0.33; % Default weight of expected error memory trace
beta = 0.5; % Default variance of perturbation distribution
tol = 0; % Default termination error condition
targetFun = @defaultTargetFunction; % Default output function (native)
plotFun = @defaultPlottingFunction; % Default plotting function (native)
targetFunPassthrough = []; % Default passthrough to output function
batchType = 'pseudorand'; % Default condition order for each batch
evalOpts = [0 50]; % Default evaluation values [plottingOptions evaluateEveryXIterations]

for iVar = 1:2:optargin
    switch varargin{iVar}
        case 'beta'
            beta = varargin{iVar+1};
        case 'maxdJ'
            maxdJ = varargin{iVar+1};
        case 'alphaX'
            alphaX = varargin{iVar+1};
        case 'alphaR'
            alphaR = varargin{iVar+1};
            
        case 'targettimes'
            targettimes = varargin{iVar+1};
        case 'input'
            inp = varargin{iVar+1};
            
        case 'targetFun'
            targetFun = varargin{iVar+1};
        case 'targetFunPassthrough'
            targetFunPassthrough = varargin{iVar+1};
            
        case 'tolerance'
            tol = varargin{iVar+1};
            
        case 'batchType'
            batchType = varargin{iVar+1};
            
        case 'plotFun'
            plotFun = varargin{iVar+1};
        case 'evalOpts'
            evalOpts = varargin{iVar+1};
    end
end

%% Save options to network structure for posterity
net.perturbProb = perturbProb;
net.beta = beta;
net.eta = eta;
net.maxdJ = maxdJ;
net.alphaX = alphaX;
net.alphaR = alphaR;
net.targettimes = targettimes;
net.input = inp;
net.targetFun = targetFun;
net.targetFunPassthrough = targetFunPassthrough;
net.tolerance = tol;
net.batchType = batchType;

N = net.N;
B = net.B;
I = net.I;
niters = size(F{1},2);

%% Checks
% The input can be either empty, or specified at each time point by the user.
hasInput = ~isempty(inp);
if (hasInput)
    assert(size(inp{1},2) == niters, 'All learning input vectors should have the same length.');
    assert(size(inp{1},1) == I, 'There must be an input entry for each input vector.');
end
assert(strcmp(batchType, 'pseudorand') || strcmp(batchType, 'linear'), 'Invalid value for batchType')

J = net.J;
wIn = net.wIn;
wFb = net.wFb;
dt = net.dt;
tau = net.tau;
dt_div_tau = dt/tau;
netNoiseSigma = net.netNoiseSigma;
actFun = net.actFun;
biasNeurons = net.useBiasNeurons;
bias = net.biasValue;
biasUnitIdent = net.biasUnitIdent;
c1 = net.energyCost;
outputUnitIdent = net.outputUnitIdent;

% Change perturbation probability to per millisecond
perturbProb = perturbProb / 1000 * dt;

mErr = Inf; % Initial error
err = Inf; % Initial error
pass = 1; % Initial pass number
errStats.err = []; errStats.pass = []; % Initialize error statistics
Rpred = zeros(1,length(F)); % Initialize expected error
%% Initialize network activity
allZ = zeros(niters,B);
allR = zeros(niters,N);
allX = zeros(niters,N);

%% Forward-pass to establish expected reward
switch batchType
    case 'pseudorand'
        condList = randperm(length(F));
    case 'linear'
        condList = 1:length(F);
end
for cond = 1:length(condList)
    thisCond = condList(cond);
    if hasInput
        thisInp = inp{thisCond};
    end
    thisTarg = F{thisCond};
    targetFeedforward = [];
    
    %% Run internal model WITHOUT perturbations
    internalRunModel(-1)
    %%
    
    Rpred(thisCond) = sum(err); % Save predicted error
end

%% Main Program %%
% Runs until tolerated error is met or stop button is pressed
figure(97)
set(gcf, 'Position', [0 5000 100 50], 'MenuBar', 'none', 'ToolBar', 'none', 'Name', 'Stop', 'NumberTitle', 'off')
UIButton = uicontrol('Style', 'togglebutton', 'String', 'STOP', 'Position', [0 0 100 50], 'FontSize', 25);
while mErr > tol && UIButton.Value == 0
    switch batchType
        case 'pseudorand'
            condList = randperm(length(F));
        case 'linear'
            condList = 1:length(F);
    end
    
    alldJ = zeros(1,length(F)); % Initialize connectivity change
    percentClipped = zeros(1,length(condList)); % Initialize percentage of change clipped
    for cond = 1:length(condList)
        thisCond = condList(cond);
        if hasInput
            thisInp = inp{thisCond};
        end
        thisTarg = F{thisCond};
        targetFeedforward = [];
        
        %% Run internal model WITH perturbations
        internalRunModel(perturbProb)
        
        %% Be sure to maintain original connectivity sparsity
        e = e';
        e(J == 0) = 0;
        
        %% Update internal weights
        dJ = - eta * e * (sum(err) - Rpred(thisCond));
        
        %% Clip change in connectivity
        percentClipped(thisCond) = sum(dJ(:) > maxdJ | dJ(:) < -maxdJ) / size(dJ,1)^2 * 100;
        dJ(dJ > maxdJ) = maxdJ;
        dJ(dJ < -maxdJ) = -maxdJ;
        dJ(isnan(dJ)) = 0; % just in case
        
        %% Commit changes
        J = J + dJ;
        
        alldJ(thisCond) = mean(abs(dJ(:))); % Save magnitude of change
        
        %% Update predicted error trace
        Rpred(thisCond) = alphaR * Rpred(thisCond) + (1.0 - alphaR) * sum(err);
    end
    
    %% Check overall error using a clean run and plot if required
    if mod(pass,evalOpts(2)) == 0 || pass == 1
        allErr = zeros(length(F),2);
        bigZ = zeros(size(allZ,1), size(allZ,2), length(condList));
        bigR = zeros(size(allR,1), size(allR,2), length(condList));
        bigX = zeros(size(allX,1), size(allX,2), length(condList));
        for cond = 1:length(condList)
            thisCond = condList(cond);
            if hasInput
                thisInp = inp{thisCond};
            end
            thisTarg = F{thisCond};
            targetFeedforward = [];
            
            %% Run internal model WITHOUT perturbations
            internalRunModel(-1)
            %%
            
            bigZ(:,:,thisCond) = allZ;
            bigR(:,:,thisCond) = allR;
            bigX(:,:,thisCond) = allX;
            allErr(thisCond,:) = err;
        end
        
        %% Save stats
        errStats.err(:,:,end+1) = allErr;
        errStats.pass(end+1) = pass;
        
        mdJ = mean(alldJ); % Calculate mean dJ for displaying
        mErr = mean(sum(allErr,2)); % Calculate mean error for stop conditions
        
        %% Populate statistics for plotting function
        plotStats.mErr = mErr;
        plotStats.mdJ = mdJ;
        plotStats.allErr = allErr;
        plotStats.percentClipped = percentClipped;
        plotStats.bigR = bigR;
        plotStats.bigZ = bigZ;
        plotStats.bigX = bigX;
        plotStats.pass = pass;
        plotStats.F = F;
        
        %% Run supplied plotting function
        plotFun(plotStats, errStats, evalOpts)
    end
    
    pass = pass + 1;
end

%% Save hard-earned results
net.J = J;
%%

%% Output error statistics if required
if ( nout >= 1 )
    errStats.err(:,:,1) = [];
    varargout{1} = errStats;
end

disp('Training time required:')
toc

%% Recurrent Neural Network Engine %%
    function internalRunModel(thisPerturbProb)
        x = x0;
        if (biasNeurons)
            x(biasUnitIdent) = bias; % Overwrite bias units
        end
        
        %% Activation Function
        r = actFun(x);
        
        %% Calculate output using supplied function
        [z, targetFeedforward] = targetFun(0, r(outputUnitIdent), targetFunPassthrough, targetFeedforward);
        
        xtrace = x;
        e = zeros(size(J)); % Initialize elegibility trace
        for i = 1:niters
            if (hasInput)
                input = wIn*thisInp(:,i);
            else
                input = 0;
            end
            
            allZ(i,:) = z;
            allR(i,:) = r;
            allX(i,:) = x;
            
            %% Calculate change in activation
            excitation = -x + J*r + input + wFb*z + netNoiseSigma*randn(N,1);
            
            %% Random fluctuation (Uniform)
            dx = beta * 2.0*(rand(N,1)-0.5);
            %% Occurs with a certain probability independently for each neuron
            dx(thisPerturbProb < rand(N,1)) = 0;
            
            %% Add all activation changes together
            x = x + dt_div_tau*excitation + dx;
            
            if (biasNeurons)
                x(biasUnitIdent) = bias; % Overwrite bias units
            end
            
            rprev = r;
            %% Activation function
            r = actFun(x);
            
            %% Calculate output using supplied function
            [z, targetFeedforward] = targetFun(i, r(outputUnitIdent), targetFunPassthrough, targetFeedforward);
            
            
            %% Maintain elegibility trace (only if network was perturbed)
            if thisPerturbProb ~= -1              
                % Original version did the cubing before multiplying by the rates
                %deltax = (x - xtrace).^3;
                %e = e + rprev*(deltax');        
                e = e + (rprev*(x - xtrace)').^3; % supra-linear (must maintain sign)
                %% Maintain activation trace
                xtrace = alphaX * xtrace + (1.0 - alphaX) * x;
            end
        end
        %% Calculate error for desired times
        useZ = allZ(targettimes,:);
        useF = thisTarg(:,targettimes)'; 
        useR = allR(:,outputUnitIdent);
        err(1) = mean(abs(useZ(:)-useF(:))); % Output error
        err(2) = c1 * mean(abs(useR(:))); % Energy cost error
    end

    %% Default output function   
    function [z, targetFeedforward] = defaultTargetFunction(~, r, ~, targetFeedforward)
        z = r; % Just passes firing rate
    end

    %% Default plotting function
    function defaultPlottingFunction(plotStats, errStats, evalOptions)
        if evalOptions(1) >= 0
            disp(['Error: ' num2str(plotStats.mErr) ' --- Mean dJ: ' num2str(plotStats.mdJ) ' --- %Clipped: ' num2str(mean(plotStats.percentClipped))])
        end
        if evalOptions(1) >= 1
            figure(98)
            set(gcf, 'Name', 'Error', 'NumberTitle', 'off')
            c = lines(size(plotStats.allErr,2));
            for type = 1:size(plotStats.allErr,2)
                h1(type) = plot(plotStats.pass, mean(plotStats.allErr(:,type)), '.', 'MarkerSize', 20, 'Color', c(type,:));
                hold on
            end                
            axis([1 plotStats.pass+0.1 0 max(squeeze(max(mean(errStats.err,1),[],3)))])
            xlabel('Training Iteration')
            ylabel('Mean Error')
            legend(h1, 'Mean Output Error', 'Mean Energy Cost Error', 'Location', 'SouthWest')
        end
        if evalOptions(1) >= 2
            figure(99)
            set(gcf, 'Name', 'Output and Neural Activity', 'NumberTitle', 'off')
            clf
            subplot(2,1,1)
            hold on
            c = lines(size(plotStats.bigZ,3));
            for condCount = 1:size(plotStats.bigZ,3)
                h2(condCount,:) = plot(plotStats.bigZ(:,:,condCount), 'Color', c(condCount,:));
                h3(condCount,:) = plot(targettimes, plotStats.F{condCount}(:,targettimes)', '.', 'MarkerSize', 8, 'Color', c(condCount,:));
            end
            legend([h2(1,1) h3(1,1)], 'Network Output', 'Target Output', 'Location', 'SouthWest')
            xlabel('Time Steps')
            ylabel('Output')
            set(gca, 'XLim', [1 size(plotStats.bigZ,1)])
            subplot(2,1,2)
            hold on
            for condCount = 1:size(plotStats.bigR,3)
                plot(plotStats.bigR(:,outputUnitIdent,condCount), 'Color', c(condCount,:))
            end
            xlabel('Time Steps')
            ylabel('Firing Rate')
            set(gca, 'XLim', [1 size(plotStats.bigR,1)])
        end
        drawnow
    end
end