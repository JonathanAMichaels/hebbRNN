% hebbRNN_Example_DNMS
%
% This function illustrates an example of reward-modulated Hebbian learning
% in recurrent neural networks to complete a delayed nonmatch-to-sample
% task.
%
%
% Version release date: ##.##.####
%
% Copyright (c) Jonathan A Michaels 2016
% German Primate Center
% jonathanamichaels AT gmail DOT com
%
% If used in published work, please link to:
% ??


clear
close all

%% Generate inputs and outputs
inp = cell(1,4);
targ = cell(1,4);
level = 1;
cue1Time = 1:200;
cue2Time = 401:600;
totalTime = 1000;
target1 = 1;
target2 = -1;
for type = 1:4
    inp{type} = zeros(2, totalTime);
    if type == 1
        inp{type}(1, [cue1Time cue2Time]) = level;
        targ{type} = ones(1, totalTime)*target1;
    elseif type == 2
        inp{type}(2, [cue1Time cue2Time]) = level;
        targ{type} = ones(1, totalTime)*target1;
    elseif type == 3
        inp{type}(1, cue1Time) = level;
        inp{type}(2, cue2Time) = level;
        targ{type} = ones(1, totalTime)*target2;
    elseif type == 4
        inp{type}(2, cue1Time) = level;
        inp{type}(1, cue2Time) = level;
        targ{type} = ones(1, totalTime)*target2;
    end
end


%% Initialize network parameters
N = 200; % Neurons
B = size(targ{1},1); % Outputs
I = size(inp{1},1); % Inputs
p = 1; % Sparsity
g = 1.3; % Spectral scaling
dt = 1; % Time step
tau = 30; % Time constant

%% Initialize learning parameters
eta = 0.02; % Learning rate
perturbProb = 3; % Frequency of neural perturbation per neuron (Hz)
systemNoise = 0.05; % Network noise level
x0 = zeros(N,1); % Initial activation state
tolerance = 0.05; % Desired error tolerance
evalOpts = [2 20]; % Plotting level and frequency of evaluation
targettimes = size(targ{1},2)-199:size(targ{1},2); % times which are evaluated in the error calculation

rng(0)
%% Create network
net = hebbRNN_create_model(N, B, I, p, g, dt, tau, 'netNoiseSigma', systemNoise,...
    'feedback', false, 'actFun', 'tanh');

%% Train network
[net, learnStats] = hebbRNN_learn_model(x0, net, targ, perturbProb, eta, 'input', inp, ...
    'tolerance', tolerance, 'evalOpts', evalOpts, 'targettimes', targettimes);

%% Run network
[Z, R, X, errStats] = hebbRNN_run_model(x0, net, targ, 'input', inp);
