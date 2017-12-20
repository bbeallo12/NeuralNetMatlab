% default layer object
% every layer has these methods:
%   init: to initialize the handles for the previous layer and next layers.
%         to initialize any other important variable like layer
%         dimensionality
%   forwardpass: Operation to forward pass data through the network
%   backpass: Operation to implement back propagation through the network
classdef layer < handle
    properties
        DimCount; % number of dimensions for layer (Example: convolution2dLayer has DimCount = 2)
        Dim; % the exact dimensions of this layer's data
        Values; % neuron values
        Next; % handle to next layer in network
        Prev; % handle to previous layer in network
        dE; % back propagation error
    end
    methods
        % determine previous layer, next layer, and number of values
        function init(nn, Prev, Next)
            nn.Prev = Prev;
            nn.Next = Next;
            nn.valCount = Prev.valCount;
        end
        function forwardPass(nn)
            nn.Next.Values = nn.Values;
        end
        % LR    Learning Rate
        % M1    Momentum 1
        % M2    Momentum 2
        % N     Current iteration
        function backPass(nn,LR,M1,M2,N)
            nn.Prev.dE = nn.dE;
        end
    end
end