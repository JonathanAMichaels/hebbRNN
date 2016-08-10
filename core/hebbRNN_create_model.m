function net = hebbRNN_create_model(N, B, I, p, g, dt, tau, varargin)

% net = hebbRNN_create_model(N, B, I, p, g, dt, tau, varargin)
%
% This function initializes a recurrent neural network for later training
% and execution
%
% INPUTS:
%
% N -- the number of recurrent neurons in network
%
% B -- the number of outputs
%
% I -- the number of inputs
%
% p -- the sparseness of the J (connectivity) matrix, (range: 0-1)
%
% g -- the spectral scaling of J
%
% dt - the integration time constant
%
% tau - the time constant of each neuron
%
%
% OPTIONAL INPUTS:
%
% actFun -- the activation function used to tranform activations into
% firing rates
% Default: 'tanh'
%
% netNoiseSigma - the variance of random gaussian noise added at each time
% point
% Default: 0
%
% useBiasNeurons -- whether or not to included neurons that have a fixed
% value
% Default: false
%
% numBiasNeurons -- how many bias neurons to use
% Default: 0
%
% feedback -- whether or not to feed the output of the network back
% Default: false
%
% energyCost -- how much to weight the firing rate of the output units in
% the calculation of total error. This parameter encourages the network to
% find solutions with as low firing rate as possible.
% Default: 0
%
% outputUnitIdent -- user-defined indexing of output units
% Default: 1:B
%
% biasUnitIdent -- user-defined indexing of bias units
% Default: []
%
%
% OUTPUTS:
%
% net -- the network structure
%
%
% Copyright (c) Jonathan A Michaels 2016
% German Primate Center
% jonathanamichaels AT gmail DOT com
%
% If used in published work please see repository README.md for citation
% and license information: https://github.com/JonathanAMichaels/hebbRNN


actFunType = 'tanh'; % Default activation function
netNoiseSigma = 0.0; % Default noise-level
useBiasNeurons = false; % Default use of bias neurons
numBiasNeurons = 0; % Default number of bias neurons
feedback = false; % Default use of output feedback
energyCost = 0; % Default weight of cost function
outputUnitIdent = 1:B; % Default identity of output units
biasUnitIdent = []; % Default identity of bias units

co(1:2) = false; % Both unit identities must be provided if defaults are not used
optargin = size(varargin,2);
for i = 1:2:optargin
    switch varargin{i}
        case 'actFun'
            actFunType = varargin{i+1};
        case 'netNoiseSigma'
            netNoiseSigma = varargin{i+1};
        case 'useBiasNeurons'
            useBiasNeurons = varargin{i+1};
        case 'numBiasNeurons'
            numBiasNeurons = varargin{i+1};
        case 'feedback'
            feedback = varargin{i+1};
        case 'energyCost'
            energyCost = varargin{i+1};
            
        case 'outputUnitIdent'
            outputUnitIdent = varargin{i+1};
            co(1) = true;
        case 'biasUnitIdent'
            biasUnitIdent = varargin{i+1};
            co(2) = true;
    end
end

%% Assertions
assert(co(1) == co(2), 'Both identities must be provided.')
assert(islogical(feedback), 'Must be logical.')
assert(p >= 0 && p <= 1, 'Sparsity must be between 0 and 1.')

%% Set bias unit indices
if useBiasNeurons
    if isempty(biasUnitIdent)
        net.biasUnitIdent = B+1 : B+numBiasNeurons;
    else
        net.biasUnitIdent = biasUnitIdent;
    end
else
    net.biasUnitIdent = [];
end

%% Initialize internal connectivity
% Connectivity is normally distributed, scaled by the size of the network,
% the sparity, and spectral scaling factor, g.
J = zeros(N,N);
for i = 1:N
    for j = 1:N
        if rand <= p
            J(i,j) = g * randn / sqrt(p*N);
        end
    end
end

net.I = I;
net.B = B;
net.N = N;
net.p = p;
net.g = g;
net.J = J;
net.outputUnitIdent = outputUnitIdent;
net.netNoiseSigma = netNoiseSigma;
net.dt = dt;
net.tau = tau;

%% Initialize input weights
net.wIn = 2*(rand(N,I)-0.5); % range from -1 to 1

%% Initialize feedback weights
net.wFb = zeros(N,B);
if feedback
    net.wFb = 2*(rand(N,B)-0.5); % range from -1 to 1
end

net.useBiasNeurons = useBiasNeurons;
if useBiasNeurons
    net.biasValue = 2*(rand(1,numBiasNeurons)-0.5); % range from -1 to 1
else
    net.biasValue = [];
end

%% Activation function
switch actFunType
    case 'tanh'
        net.actFun = @tanh;
        net.actFunDeriv = @(r) 1.0-r.^2;
    case 'recttanh'
        net.actFun = @(x) (x > 0) .* tanh(x);
        net.actFunDeriv = @(r) (r > 0) .* (1.0 - r.^2);
    case 'baselinetanh' % Similar to Rajan et al. (2010)
        net.actFun = @(x) (x > 0) .* (1 - 0.1) .* tanh(x / (1 - 0.1)) ...
            + (x <= 0) .* 0.1 .* tanh(x / 0.1);
    case 'linear'
        net.actFun = @(x) x;
    otherwise
        assert(false, 'Nope!');
end

%% Cost function
net.energyCost = energyCost;
end