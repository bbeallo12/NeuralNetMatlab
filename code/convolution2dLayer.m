classdef convolution2dLayer < layer
    properties
        W;
        M;
        V;
        Depth;
        Size;
        Stride;
        Padding;
        convSp;
        convMtx;
    end
    methods
        % instantiate layer
        function nn = convolution2dLayer(Size, Depth, Stride, Padding)
            nn.Depth = Depth;
            nn.Size = Size;
            nn.Stride = Stride;
            nn.Padding = Padding;
            nn.DimCount = 2;
        end
        % initialize layer
        % this is called by the neuralNet class
        function init(nn, Prev, Next)
            nn.Prev = Prev;
            nn.Next = Next;
            
            if(nn.Prev.DimCount ~= nn.DimCount)
                error('Requires 2 Dimensional Input (convolution2dLayer)');
            end
            nn.Dim = (nn.Prev.Dim(1:2)+2*nn.Padding-nn.Size)./nn.Stride + 1;
            if(mod(nn.Dim(1),1)~=0 || mod(nn.Dim(2),1)~=0)
                error('Bad Dimensions (convolution2dLayer)');
            end
            nn.Dim = [nn.Dim,nn.Depth];

            strideI = nn.Stride(1);
            strideJ = nn.Stride(2);
            sizeI = nn.Size(1);
            sizeJ = nn.Size(2);
            maxI = Prev.Dim(1)+2*nn.Padding(1)-nn.Size(1);
            maxJ = Prev.Dim(2)+2*nn.Padding(2)-nn.Size(2);
            dimI = nn.Dim(1);
            dimJ = nn.Dim(2);
            depth = nn.Prev.Dim(3);
            
            % using the filter dimensions, create a sparse matrix to
            % map input data to the proper filters and speed up convolution
            
            convSpT = zeros(dimJ*dimI*sizeI*sizeJ*depth,3);
            map = reshape(1:prod(Prev.Dim(1:3)+[2*nn.Padding,0]),Prev.Dim(1:3)+[2*nn.Padding,0]);
            L = 0;
            for j = 1:strideJ:maxJ+1
                for i = 1:strideI:maxI+1
                    temp = map(1+i-1:sizeI+i-1,1+j-1:sizeJ+j-1,:);
                    for k = 1:sizeI*sizeJ*depth
                        L = L + 1;
                        convSpT(L,:) = [L,temp(k),1];
                    end
                end
            end
            nn.convSp = sparse(convSpT(:,1),convSpT(:,2),convSpT(:,3),dimJ*dimI*sizeI*sizeJ*depth,prod(Prev.Dim(1:3)+[2*nn.Padding,0])); 
            
            % initialize weights and momentum
            nn.W = randn(prod(nn.Size)*Prev.Dim(3)+1,nn.Dim(3))/sqrt(prod(nn.Size)*nn.Prev.Dim(3)+1);
            nn.M = zeros(prod(nn.Size)*Prev.Dim(3)+1,nn.Dim(3));
            nn.V = zeros(prod(nn.Size)*Prev.Dim(3)+1,nn.Dim(3));
        end
        function forwardPass(nn)
            % apply 2d convolution to data
            nn.Values = padarray(nn.Values,nn.Padding);
            
            rows = size(nn.Values,1);
            cols = size(nn.Values,2);
            depth = size(nn.Values,3);
            images = size(nn.Values,4);
            
            nn.Values = reshape(nn.Values,[rows*cols*depth,images]);
            
            nn.convMtx = nn.convSp * nn.Values;
            nn.convMtx = reshape(nn.convMtx,[prod(nn.Size(1:2))*depth,prod(nn.Dim(1:2)),images]);
            nn.convMtx = permute(nn.convMtx,[2,3,1]);
            nn.convMtx = reshape(nn.convMtx,[prod(nn.Dim(1:2))*images,prod(nn.Size(1:2))*depth]);
            nn.convMtx = [ones(size(nn.convMtx,1),1),nn.convMtx];
            
            nn.Values = nn.convMtx * nn.W;
            nn.Values = reshape(nn.Values,[nn.Dim(1:2),images,nn.Dim(3)]);
            nn.Values = permute(nn.Values,[1 2 4 3]);
            
            nn.Next.Values = nn.Values;
        end
        function backPass(nn,LR,M1,M2,N)
            % calculate gradient of forward pass,
            % increment filter weights,
            % combine it with the back prop error accumulated from next layer,
            % and calculate new back prop error to pass to previous layer.
            Dim = nn.Prev.Dim;
            
            nn.dE = permute(nn.dE,[1 2 4 3]);
            nn.dE = reshape(nn.dE,[size(nn.dE,1)*size(nn.dE,2)*size(nn.dE,3),size(nn.dE,4)]);
            G = nn.convMtx' * nn.dE;

            nn.M = M1.*nn.M + (1-M1).*G;
            nn.V = M2.*nn.V + (1-M2).*G.^2;
            
            LR2 = LR * sqrt(1-M2^N)/(1-M1^N);
            
            nn.dE = nn.dE*nn.W(2:end,:)';
            
            nn.dE = reshape(nn.dE,[prod(nn.Dim(1:2)),size(nn.Prev.Values,4),size(nn.dE,2)]);
            nn.dE = permute(nn.dE,[1,3,2]);
            nn.dE = reshape(nn.dE,[prod(nn.Dim(1:2))*size(nn.dE,2),size(nn.Prev.Values,4)]);
            nn.dE = nn.convSp' * nn.dE;
            nn.dE = reshape(nn.dE,[Dim(1:2)+2*nn.Padding,Dim(3),size(nn.Prev.Values,4)]);
            nn.dE = nn.dE(1+nn.Padding(1):end-nn.Padding(1),1+nn.Padding(2):end-nn.Padding(2),:,:);
            nn.Prev.dE = nn.dE;
                       
            nn.W = nn.W - LR2 * nn.M./(sqrt(nn.V)+1e-8);
            
        end
    end
end