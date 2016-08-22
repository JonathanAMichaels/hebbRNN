% hebbRNN_Example_CO
%
% This function illustrates an example of reward-modulated Hebbian learning
% in recurrent neural networks to complete a center-out reaching task.
%
%
% Copyright (c) Jonathan A Michaels 2016
% German Primate Center
% jonathanamichaels AT gmail DOT com
%
% If used in published work please see repository README.md for citation
% and license information: https://github.com/JonathanAMichaels/hebbRNN


clear
close all

numConds = 4; % Number of peripheral targets. Try changing this number to alter the difficulty!
moveTime = 150; % Total trial time
L = [3 3]; % Length of each segment of the arm

%% Populate target function passthrough data
% This is information that the user can define and passthrough to the
% network output function
targetFunPassthrough.L = L;
targetFunPassthrough.kinTimes = moveTime-100 : moveTime; % Only allow movement for last 150ms

%% General inputs and output
inp = cell(1,numConds);
targ = cell(1,numConds);
ang = linspace(0, 2*pi - 2*pi/numConds, numConds);
for cond = 1:numConds
    inp{cond} = zeros(numConds+1, moveTime);
    inp{cond}(cond,:) = 1;
    inp{cond}(numConds+1,1:targetFunPassthrough.kinTimes-1) = 1;
    targ{cond} = [ones(moveTime,1)*sin(ang(cond)) ones(moveTime,1)*cos(ang(cond))]';
end
% In the center-out reaching task the network needs to produce the joint angle
% velocities of a two-segment arm to reach to a number of peripheral
% targets spaced along a circle in the 2D plane, based on the desired target
% specified by the input.

%% Initialize network parameters
N = 150; % Number of neurons
B = size(targ{1},1); % Outputs
I = size(inp{1},1); % Inputs
p = 1; % Sparsity
g = 1.1; % Spectral scaling
dt = 1; % Time step
tau = 30; % Time constant

%% Initialize learning parameters
eta = 0.5; % Learning rate
perturbProb = 3; % Frequency of neural perturbation per neuron (Hz)
systemNoise = 0.0; % Network noise level
x0 = zeros(N,1); % Initial activation state
tolerance = 0.08; % Desired error tolerance
evalOpts = [2 20]; % Plotting level and frequency of evaluation
targettimes = size(targ{cond},2)-1:size(targ{cond},2); % times which are evaluated in the error calculation
targetFun = @hebbRNN_COTargetFun; % handle of custom target function

rng(0)
%% Create network
% This function initializes your network. Look inside to see information
% about the many optional parameters
net = hebbRNN_create_model(N, B, I, p, g, dt, tau, 'netNoiseSigma', systemNoise, 'useBiasNeurons', true, 'numBiasNeurons', 4, ...
    'feedback', false, 'actFun', 'tanh', 'energyCost', 0.1);

%% Train network
% This step should take about 5 minutes, depending on your processor.
% Can be stopped at any time by pressing the STOP button.
% Look inside to see information about the many optional parameters.
[net, learnStats] = hebbRNN_learn_model(x0, net, targ, perturbProb, eta, 'input', inp, ...
    'tolerance', tolerance, 'evalOpts', evalOpts, ...
    'targettimes', targettimes, ...
    'targetFun', targetFun, 'targetFunPassthrough', targetFunPassthrough);

%% Run network
[Z, R, X, errStats, kin] = hebbRNN_run_model(x0, net, targ, 'input', inp);

%% Plot center-out reaching results
c = lines(length(inp));
figure(1)
for cond = 1:length(inp)
    h(cond) = filledCircle([targ{cond}(1,1) targ{cond}(2,1)], 0.2, 100, [0.9 0.9 0.9]);
    h(cond).EdgeColor = c(cond,:);
    hold on
end
for cond = 1:length(inp)
    plot(Z{cond}(1,:), Z{cond}(2,:), 'Color', c(cond,:), 'Linewidth', 2)
    
end
axis([-1.3 1.3 -1.3 1.3])
axis square

%% Play short movie showing trained movements for all directions
figure(2)
set(gcf, 'Color', 'white')
for cond = 1:length(inp)
    for t = 1:5:length(targetFunPassthrough.kinTimes)-1
        clf
        for cond2 = 1:length(inp)
            h(cond2) = filledCircle([targ{cond2}(1,1) targ{cond2}(2,1)], 0.2, 100, [0.9 0.9 0.9]);
            h(cond2).EdgeColor = c(cond2,:);
            hold on
        end
        
        line([kin(cond).initvals(1) kin(cond).posL1(t,1)], ...
            [kin(cond).initvals(2) kin(cond).posL1(t,2)], 'LineWidth', 8, 'Color', 'black')
        line([kin(cond).posL1(t,1) Z{cond}(1,targetFunPassthrough.kinTimes(t))], ...
            [kin(cond).posL1(t,2) Z{cond}(2,targetFunPassthrough.kinTimes(t))], 'LineWidth', 8, 'Color', 'black')
        
        axis([-1.2 4.5 kin(cond).initvals(2) 1.2])
        axis off
        drawnow
        pause(0.02)
    end
    pause(0.5)
end