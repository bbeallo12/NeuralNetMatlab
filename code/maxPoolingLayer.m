classdef maxPoolingLayer < layer
    properties
        Max;
        Size;
        Stride;
        Padding;
        convSp;
    end
    methods
        % instantiate layer
        function nn = maxPoolingLayer(Size, Stride, Padding)
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
                error('Requires 2 Dimensional Input (maxPoolingLayer)');
            end
            nn.Dim = (nn.Prev.Dim(1:2)+2*nn.Padding-nn.Size)./nn.Stride + 1;
            if(mod(nn.Dim(1),1)~=0 || mod(nn.Dim(2),1)~=0)
                error('Bad Dimensions (convolution2dLayer)');
            end
            nn.Dim = [nn.Dim,nn.Prev.Dim(3)];
            
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
            for k = 1:depth
                for j = 1:strideJ:maxJ+1
                    for i = 1:strideI:maxI+1
                        for m = 1:sizeI*sizeJ
                            temp = map(1+i-1:sizeI+i-1,1+j-1:sizeJ+j-1,k);
                            L = L + 1;
                            convSpT(L,:) = [L,temp(m),1];
                        end
                    end
                end
            end
            nn.convSp = sparse(convSpT(:,1),convSpT(:,2),convSpT(:,3),dimJ*dimI*sizeI*sizeJ*depth,prod(Prev.Dim(1:3)+[2*nn.Padding,0]));
        end
        function forwardPass(nn)
            % apply 2d max pooling to data
            nn.Values = padarray(nn.Values,nn.Padding);
            
            rows = size(nn.Values,1);
            cols = size(nn.Values,2);
            depth = size(nn.Values,3);
            images = size(nn.Values,4);
            
            nn.Values = reshape(nn.Values,[rows*cols*depth,images]);
            nn.Values = nn.convSp * nn.Values;
            nn.Values = reshape(nn.Values,[prod(nn.Size),prod(nn.Dim),images]);
            nn.Values = permute(nn.Values,[2,3,1]);
            nn.Values = reshape(nn.Values,[size(nn.Values,1)*size(nn.Values,2),size(nn.Values,3)]);
            [nn.Values,nn.Max] = max(nn.Values,[],2);
            nn.Values = reshape(nn.Values,[nn.Dim,images]);
            
            nn.Next.Values = nn.Values;
        end
        function backPass(nn,~,~,~,~)
            % calculate gradient of forward pass,
            % reroute errors to previous max values,
            % combine it with the back prop error accumulated from next layer,
            % and calculate new back prop error to pass to previous layer.
            rows = size(nn.dE,1);
            cols = size(nn.dE,2);
            depth = size(nn.dE,3);
            images = size(nn.dE,4);
            
            nn.dE = reshape(nn.dE,[numel(nn.dE),1]);
            
            dE = zeros(numel(nn.dE),prod(nn.Size));
            dE(sub2ind(size(dE),1:size(dE,1),nn.Max')) = nn.dE;
            nn.dE = dE;
            nn.dE = reshape(nn.dE,[rows*cols*depth,images,size(nn.dE,2)]);
            nn.dE = permute(nn.dE,[3,1,2]);
            nn.dE = reshape(nn.dE,[size(nn.dE,1)*size(nn.dE,2),size(nn.dE,3)]);
            nn.dE = nn.convSp' * nn.dE;
            nn.dE = reshape(nn.dE,[size(nn.Prev.Values,1)+2*nn.Padding(1),size(nn.Prev.Values,2)+2*nn.Padding(1),size(nn.Prev.Values,3),images]);
            nn.dE = nn.dE(1+nn.Padding(1):end-nn.Padding(1),1+nn.Padding(2):end-nn.Padding(2),:,:);
            
            nn.Prev.dE = nn.dE;
        end
    end
end
