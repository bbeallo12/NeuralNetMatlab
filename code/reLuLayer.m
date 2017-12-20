% rectified linear unit activation function
classdef reLuLayer < layer
    properties
    end
    methods
        % initialize layer
        function init(nn, Prev, Next)
            nn.Prev = Prev;
            nn.Next = Next;
            nn.Dim = Prev.Dim;
            nn.DimCount = Prev.DimCount;
        end
        % activate and pass forward
        function forwardPass(nn)
            nn.Next.Values = nn.Values.*(nn.Values>=0);
        end
        % calculate derivative and pass backward
        function backPass(nn,~,~,~,~)
            nn.Prev.dE = nn.dE.*(nn.Values>=0);
        end
    end
    methods(Static)
        
    end
end