classdef passLayer < layer
    properties
    end
    methods
        % instantiate layer
        function nn = passLayer()
        end
        % initialize layer
        % this is called by the neuralNet class
        function init(nn, Prev, Next)
            nn.Dim = Prev.Dim;
            nn.DimCount = Prev.DimCount;
            nn.Prev = Prev;
            nn.Next = Next;
        end
        function forwardPass(nn)
            nn.Next.Values = nn.Values;
        end
        function backPass(nn,~,~,~,~)
            nn.Prev.dE = nn.dE;
        end
    end
end