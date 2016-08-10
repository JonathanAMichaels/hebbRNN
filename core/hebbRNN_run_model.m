function [Z, R, X, varargout] = hebbRNN_run_model(x0, net, F, varargin)

% net = hebbRNN_run_model(x0, net, F, varargin)
%
% This function runs the network structure (net) initialized by
% hebbRNN_create_model and trained by hebbRNN_learn_model with the desired
% input.
% NOTE: Networks must have been trained by hebbRNN_learn_model in order to be run
% by this function
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
%
% OPTIONAL INPUTS:
%
% input -- the input to the network
% Must be a cell of size: 1 x conditions
% Each cell must be of size: net.I x time points
% Default: []
%
%
% OUTPUTS:
%
% Z -- the output of the network
%
% R -- the firing rate of all neurons in the network
%
% X -- the activation of all neurons in the network
%
% errStats -- the structure containing error information from learning
% (optional)
%
% targetOut -- structure containing the output produced by targetFun
% (optional)
%
%
% Copyright (c) Jonathan A Michaels 2016
% German Primate Center
% jonathanamichaels AT gmail DOT com
%
% If used in published work please see repository README.md for citation
% and license information: https://github.com/JonathanAMichaels/hebbRNN


% Variable output considerations
nout = max(nargout,1)-1;

% Variable input considerations
optargin = size(varargin,2);

inp = [];

targetFun = net.targetFun;
targetFunPassthrough = net.targetFunPassthrough;
targettimes = net.targettimes;

for iVar = 1:2:optargin
    switch varargin{iVar}
        case 'input'
            inp = varargin{iVar+1};
    end
end

N = net.N;
B = net.B;
I = net.I;
niters = size(F{1},2);

% The input can be either empty, or specified at each time point by the user.
hasInput = ~isempty(inp);
if (hasInput)
    assert(size(inp{1},2) == niters, 'All learning input vectors should have the same length.');
    assert(size(inp{1},1) == I, 'There must be an input entry for each input vector.');
end

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
outputUnitIdent = net.outputUnitIdent;
c1 = net.energyCost;

condList = 1:length(F);
Z = cell(1,length(condList));
R = cell(1,length(condList));
X = cell(1,length(condList));
allErr = zeros(length(condList),2);
for cond = 1:length(condList)
    thisCond = condList(cond);
    if hasInput
        thisInp = inp{thisCond};
    end
    thisTarg = F{thisCond};
    targetFeedforward = [];
    
    allZ = zeros(niters,B);
    allR = zeros(niters,N);
    allX = zeros(niters,N);
    
    x = x0;
    if (biasNeurons)
        x(biasUnitIdent) = bias; % Overwrite bias units
    end
    
    %% Activation function
    r = actFun(x);
    
    %% Calculate output using supplied function
    [z, targetFeedforward] = targetFun(0, r(outputUnitIdent), targetFunPassthrough, targetFeedforward);
    
    for i = 1:niters
        if (hasInput)
            input = wIn*thisInp(:,i);
        else
            input = 0;
        end
        
        allZ(i,:) = z;
        allR(i,:) = r;
        allX(i,:) = x;
        if i == niters
            saveTarg = targetFeedforward;
        end
        
        %% Calculate change in activation
        excitation = -x + J*r + input + wFb*z + netNoiseSigma*randn(N,1);
        %% Add all activation changes together
        x = x + dt_div_tau*excitation;
        
        if (biasNeurons)
            x(biasUnitIdent) = bias; % Overwrite bias units
        end
        
        %% Activation function
        r = actFun(x);
        
        %% Calculate output using supplied function
        [z, targetFeedforward] = targetFun(i, r(outputUnitIdent), targetFunPassthrough, targetFeedforward);
    end
    %% Calculate error for desired times
    useZ = allZ(targettimes,:);
    useF = thisTarg(:,targettimes)';
    useR = allR(:,outputUnitIdent);
    err(1) = mean(abs(useZ(:)-useF(:)));
    err(2) = c1 * mean(abs(useR(:)));
    allErr(thisCond,:) = err;
    %% Save all states
    Z{cond} = allZ';
    R{cond} = allR';
    X{cond} = allX';
    if ~isempty(saveTarg)
        targetOut(cond) = saveTarg;
    end
end

%% Populate error statistics
errStats.err = allErr;

%% Output error statistics if required
if (nout >= 1)
    varargout{1} = errStats;
end
if (nout >= 2)
    if exist('targetOut', 'var')
        varargout{2} = targetOut;
    else
        varargout{2} = [];
    end
end

%% Default output function   
    function [z, targetFeedforward] = defaultTargetFunction(~, r, ~, targetFeedforward)
        z = r; % Just passes firing rate
    end
end