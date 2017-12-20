classdef fullyConnectedLayer < layer
    properties
        W;
        M;
        V;
    end
    methods
        % instantiate layer
        function nn = fullyConnectedLayer(Dim)
            nn.Dim = Dim;
            nn.DimCount = 0;
        end
        % initialize layer
        % this is called by the neuralNet class
        function init(nn, Prev, Next)
            nn.Prev = Prev;
            nn.Next = Next;
            if(nn.Prev.DimCount == 0)
                nn.W = normrnd(0,1/sqrt(nn.Prev.Dim(1)+1),nn.Dim,nn.Prev.Dim(1)+1);
                nn.M = zeros(nn.Dim,nn.Prev.Dim(1)+1);
                nn.V = zeros(nn.Dim,nn.Prev.Dim(1)+1);
            elseif(nn.Prev.DimCount == 2)
                nn.W = normrnd(0,1/sqrt(prod(nn.Prev.Dim(1:3))+1),nn.Dim,prod(nn.Prev.Dim(1:3))+1);
                nn.M = zeros(nn.Dim,prod(nn.Prev.Dim(1:3))+1);
                nn.V = zeros(nn.Dim,prod(nn.Prev.Dim(1:3))+1);
            end
        end
        function forwardPass(nn)
            % apply weights to data
            % if data dimension is higher than 0, reduce dimensionality
            if(nn.Prev.DimCount == 0)
                nn.Values = nn.W * [ones(1,size(nn.Values,2));nn.Values];
            elseif(nn.Prev.DimCount == 2)
                nn.Values = reshape(nn.Values,[prod(nn.Prev.Dim(1:3)),size(nn.Prev.Values,4)]);
                nn.Values = nn.W * [ones(1,size(nn.Values,2));nn.Values];
            end
            nn.Next.Values = nn.Values;
        end
        function backPass(nn,LR,M1,M2,N)
            % calculate gradient of forward pass,
            % increment weights,
            % combine it with the back prop error accumulated from next layer,
            % and calculate new back prop error to pass to previous layer.
            
            % if input dimensionality was not 0
            % reshape back prop error to original dimensionality
            if(nn.Prev.DimCount == 2)
                PrevValues = reshape(nn.Prev.Values,[nn.Ins,size(nn.Prev.Values,4)]);
            else
                PrevValues = nn.Prev.Values;
            end
            
            G = nn.dE*[ones(1,size(PrevValues,2));PrevValues]';
            nn.M = M1.*nn.M + (1-M1).*G;
            nn.V = M2.*nn.V + (1-M2).*G.^2;

            LR2 = LR * sqrt(1-M2^N)/(1-M1^N);

            nn.dE = (nn.W(:,2:end))'*nn.dE;
            
            if(nn.Prev.DimCount == 2)
                nn.dE = reshape(nn.dE,[nn.Prev.Dim(1:3),size(nn.Prev.Values,4)]);
            end

            nn.Prev.dE = nn.dE;
            nn.W = nn.W - LR2 * nn.M./(sqrt(nn.V)+1e-8);
        end
    end
end