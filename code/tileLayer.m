classdef tileLayer < layer
    properties
        tileMask;
        tiles;
        epoche;
    end
    methods
        % instantiate layer
        function nn = tileLayer(tiles)
            nn.DimCount = 0;
            nn.tiles = tiles;
            nn.epoche = 0;
        end
        % initialize layer
        % this is called by the neuralNet class
        function init(nn, Prev, Next)
            nn.Dim = Prev.Dim;
            nn.Prev = Prev;
            nn.Next = Next;
        end
        function forwardPass(nn)
            % apply weights to data
            % if data dimension is higher than 0, reduce dimensionality
            if(nn.epoche ~= size(nn.Values,2))
                nn.epoche = size(nn.Values,2);
                nn.tileMask = sparse(repmat(eye(nn.epoche),[1,nn.tiles]));
            end
            nn.Values = nn.Values * nn.tileMask;
            nn.Next.Values = nn.Values;
        end
        function backPass(nn,~,~,~,~)
            % calculate gradient of forward pass,
            % increment weights,
            % combine it with the back prop error accumulated from next layer,
            % and calculate new back prop error to pass to previous layer.
            
            % if input dimensionality was not 0
            % reshape back prop error to original dimensionality
            nn.dE = nn.dE * nn.tileMask';
            nn.Prev.dE = nn.dE;
        end
    end
end