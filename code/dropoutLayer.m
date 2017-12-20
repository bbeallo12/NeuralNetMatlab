% dropOut Layer
classdef dropoutLayer < layer
    properties
        keepRate;
        initKeepRate;
        keeps;
    end
    methods
        % initialize layer
        function nn = dropoutLayer(keepRate)
            nn.initKeepRate = keepRate;
            nn.keepRate = keepRate;
        end
        function init(nn, Prev, Next)
            nn.Prev = Prev;
            nn.Next = Next;
            nn.Dim = Prev.Dim;
            nn.DimCount = Prev.DimCount;
        end
        function reset(nn)
            nn.keepRate = nn.initKeepRate;
        end
        function one(nn)
            nn.keepRate = 1.0;
        end
        % activate and pass forward
        function forwardPass(nn)
            nn.keeps = rand(size(nn.Values))<nn.keepRate;
            nn.Next.Values = nn.Values.*nn.keeps;
        end
        % calculate derivative and pass backward
        function backPass(nn,~,~,~,~)
            nn.Prev.dE = nn.dE.*nn.keeps;
        end
    end
    methods(Static)
        
    end
end