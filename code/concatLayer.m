classdef concatLayer < layer
    properties
        axis;
        Count;
        Layers;
        concatValues;
    end
    methods
        % instantiate layer
        function nn = concatLayer(Count,axis)
            nn.DimCount = 0;
            nn.Dim = 0;
            nn.Count = Count;
            nn.Layers = cell(nn.Count,1);
            nn.axis = axis;
            for i = 1:nn.Count
                nn.Layers{i} = passLayer();
            end
        end
        % initialize layer
        % this is called by the neuralNet class
        function init(nn, Prev, Next)
            assert(isa(Prev,'parallelLayer'),'concatLayer must be preceded by a parallelLayer');
            assert(Prev.Count == nn.Count,'Consecutive parallelLayers must have the same number of layers');
            
            for L = 1:nn.Count
                nn.Dim = nn.Dim + Prev.Layers{L}.Dim;
                nn.Layers{L}.init(Prev.Layers{L},nn);
            end
            nn.Prev = Prev;
            nn.Next = Next;
        end
        function forwardPass(nn)
            % apply weights to data
            % if data dimension is higher than 0, reduce dimensionality
            nn.Layers{1}.forwardPass();
            nn.concatValues = nn.Values;
            for L = 2:nn.Count
                nn.Layers{L}.forwardPass();
                nn.concatValues = cat(nn.axis,nn.concatValues, nn.Values);
            end
            
            nn.Values = nn.concatValues;
            
            nn.Next.Values = nn.Values;
        end
        function backPass(nn,LR,M1,M2,N)
            % calculate gradient of forward pass,
            % increment weights,
            % combine it with the back prop error accumulated from next layer,
            % and calculate new back prop error to pass to previous layer.
            
            % if input dimensionality was not 0
            % reshape back prop error to original dimensionality
            dim = size(nn.dE);
            dE = mat2cell(nn.dE,ones(nn.Count,1)*dim(1)/nn.Count,dim(2));
            
            for L = 1:nn.Count
                nn.Layers{L}.dE = dE{L};
                nn.Layers{L}.backPass(LR,M1,M2,N);
            end
        end
    end
end