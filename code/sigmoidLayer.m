% sigmoid activation function
classdef sigmoidLayer < layer
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
            nn.Values = 1./(1+exp(-nn.Values));
            nn.Next.Values = nn.Values;
        end
        function backPass(nn,~,~,~,~)
            nn.dE = nn.dE .* nn.Values.*(1-nn.Values);
            nn.Prev.dE = nn.dE;
        end
    end
    methods(Static)
        
    end
end