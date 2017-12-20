classdef parallelLayer < layer
    properties
        Layers;
        inType;
        Count;
    end
    methods
        % instantiate layer
        function nn = parallelLayer(Layers)
            nn.Layers = Layers;
            nn.Count = numel(Layers);
            nn.inType = 0;
        end
        % initialize layer
        % this is called by the neuralNet class
        function init(nn, Prev, Next)
            assert(isa(Next,'parallelLayer')||isa(Next,'concatLayer'),'parallelLayer must be followed by either a parallelLayer of a concatLayer')
            if(isa(Prev,'parallelLayer'))
                assert(Prev.Count == nn.Count,'Consecutive parallelLayers must have the same number of layers');
                nn.inType = true;
            else
                nn.Dim = Prev.Dim;
                nn.DimCount = Prev.DimCount;
                nn.inType = false;
            end
            
            if(nn.inType)
                for L = 1:nn.Count
                    nn.Layers{L}.init(Prev.Layers{L},Next.Layers{L});
                end
            else
                for L = 1:nn.Count
                    nn.Layers{L}.init(nn,Next.Layers{L});
                end
            end
            nn.Prev = Prev;
            nn.Next = Next;
        end
        function forwardPass(nn)
            % apply weights to data
            % if data dimension is higher than 0, reduce dimensionality
            for L = 1:nn.Count
                if(~nn.inType)
                    nn.Layers{L}.Values = nn.Values;
                end
                nn.Layers{L}.forwardPass();
            end
        end
        function backPass(nn,LR,M1,M2,N)
            % calculate gradient of forward pass,
            % increment weights,
            % combine it with the back prop error accumulated from next layer,
            % and calculate new back prop error to pass to previous layer.
            
            % if input dimensionality was not 0
            % reshape back prop error to original dimensionality
            % reshape back prop error to original dimensionality
            dE = nan;
            for L = 1:nn.Count
                nn.Layers{L}.backPass(LR,M1,M2,N);
                if(~nn.inType)
                    if(isnan(dE))
                        dE = nn.dE;
                    else
                        dE = dE + nn.dE;
                    end
                end
            end
            
            if(~nn.inType)
                nn.dE = dE/nn.Count;
                nn.Prev.dE = nn.dE;
            end

        end
    end
end