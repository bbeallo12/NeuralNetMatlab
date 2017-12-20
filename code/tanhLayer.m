% hyperbolic tangent activation function
classdef tanhLayer < layer
    properties
    end
    methods
        function init(nn, Prev, Next)
            nn.Prev = Prev;
            nn.Next = Next;
            nn.Dim = Prev.Dim;
            nn.DimCount = Prev.DimCount;
        end
        function forwardPass(nn)
            nn.Values = tanh(nn.Values);
            nn.Next.Values = nn.Values;
        end
        function backPass(nn,~,~,~,~)
            nn.dE = nn.dE .* (1-nn.Values.^2);
            nn.Prev.dE = nn.dE;
        end
    end
    methods(Static)
        
    end
end