classdef geneticLayer < layer
    properties
        Gene = nan;
        Mask;
        Layers = 1;
        Hiddens = 0;
        Acts;
        
        Ins;
        PrevDimCount;
        PrevDim;
        A;
        W;
        M;
        V;
        
        mx = 1;
        
        Dist;
        dDist;
        Params;
    end
    methods
        % instantiate layer
        function nn = geneticLayer(Dim)
            nn.Dim = Dim;

            nn.Layers = 1;
            nn.Hiddens = 0;
            
            nn.DimCount = 0;
        end
        % initialize layer
        % this is called by the neuralNet class
        function init(nn, Prev, Next)
            nn.Prev = Prev;
            nn.Next = Next;
            
            if(isa(Prev,'evolveLayer'))
                if(nn.Prev.Prev.DimCount == 2)
                    nn.Ins = prod(nn.Prev.Prev.Dim(1:3));
                    nn.PrevDim = nn.Prev.Prev.Dim;
                else
                    nn.Ins = prod(nn.Prev.Prev.Dim(1));
                end
                nn.PrevDimCount = nn.Prev.Prev.DimCount;
            else
                if(nn.Prev.DimCount == 2)
                    nn.Ins = prod(nn.Prev.Dim(1:3));
                    nn.PrevDim = nn.Prev.Dim;
                else
                    nn.Ins = prod(nn.Prev.Dim(1));
                end
                nn.PrevDimCount = nn.Prev.DimCount;
            end
            
            if(isnan(nn.Gene))
                nn.generateGenes()
            end

            nn.Dist = 0;
            nn.dDist = 0;
            nn.A = cell(nn.Layers+1,1);
            if(size(nn.Gene,1)~=nn.Ins+nn.Dim+nn.Hiddens)
               disp('test')
            end
            nn.Mask = [ones(nn.Ins+nn.Dim+nn.Hiddens,1),nn.Gene];
            nn.W = nn.Mask.*normrnd(0,1/sqrt(nn.Ins+nn.Dim+nn.Hiddens+1),nn.Ins+nn.Dim+nn.Hiddens,nn.Ins+nn.Dim+nn.Hiddens+1);
            nn.M = sparse(nn.Ins+nn.Dim+nn.Hiddens,nn.Ins+nn.Dim+nn.Hiddens+1);
            nn.V = sparse(nn.Ins+nn.Dim+nn.Hiddens,nn.Ins+nn.Dim+nn.Hiddens+1);
            nn.Params = sum(nn.Gene(:));
        end
        function generateGenes(nn)
            nn.Acts = randi([0,3],[nn.Ins+nn.Dim+nn.Hiddens, 1]);
            nn.Gene = sparse(rand(nn.Ins+nn.Dim+nn.Hiddens)>0.5);
        end
        function forwardPass(nn)
            % apply weights to data
            % if data dimension is higher than 0, reduce dimensionality
            if(nn.PrevDimCount == 2)
                nn.Values = reshape(nn.Values,[nn.Ins,size(nn.Values,4)]);
            end

            B = size(nn.Values,2);
            nn.A{1} = sparse([nn.Values;zeros(nn.Dim+nn.Hiddens,B)]);
            for i = 1:nn.Layers
                nn.A{i+1} = nn.act(nn.W*[ones(1,B);nn.A{i}],nn.Acts);
                
                nn.A{i+1}(isinf(nn.A{i+1})) = 0;
                nn.A{i+1}(isnan(nn.A{i+1})) = 0;
                if(sum(sum(isnan(nn.A{i+1})))>0)
                    disp('geneticLayer 92')
                elseif(sum(sum(isinf(nn.A{i+1})))>0)
                    disp('geneticLayer 94')
                end 
            end
            nn.Values = full(nn.A{end}(nn.Ins+1:nn.Ins+nn.Dim,:));
            
            nn.Next.Values = nn.Values;
        end
        function backPass(nn,LR,M1,M2,N)
            % calculate gradient of forward pass,
            % increment weights,
            % combine it with the back prop error accumulated from next layer,
            % and calculate new back prop error to pass to previous layer.
            
            % if input dimensionality was not 0
            % reshape back prop error to original dimensionality

            B = size(nn.dE,2);
            E = 1E-8;
                       
            dE = nn.dE;
            
            if (nn.mx < max(dE(:)))
                nn.mx = max(abs(dE(:)));
            end
            
            if(sum(dE(:)==0)>0)
                disp('zeros')
            end
            dE(dE==0) = nn.mx;
            
            ddDist = sqrt(sum(dE(:).^2))/numel(dE) - nn.dDist;
            nn.dDist = sqrt(sum(dE(:).^2))/numel(dE);
            nn.Dist = 0.1*nn.Dist + 0.9.*nn.dDist + 0.1.*ddDist;
            
            nn.dE = sparse([zeros(nn.Ins,B);nn.dE;zeros(nn.Hiddens,B)]);
            
            if(sum(isnan(nn.dE(:)))>0)
                disp('geneticLayer 131')
            elseif(sum(isinf(nn.dE(:)))>0)
                disp('geneticLayer 133')
            end     
            nn.dE = nn.dE .* nn.dAct(nn.A{end},nn.Acts);
            if(sum(isnan(nn.dE(:)))>0)
                disp('geneticLayer 137')
            elseif(sum(isinf(nn.dE(:)))>0)
                disp('geneticLayer 139')
            end  
            for i = nn.Layers:-1:1
                G = nn.dE * [ones(1,B);nn.A{i}]';
                if(sum(sum(isnan(G)))>0)
                    disp('geneticLayer 144')
                elseif(sum(sum(isinf(G)))>0)
                    disp('geneticLayer 146')
                end  
                nn.M = M1.*nn.M + (1-M1).*G;
                nn.V = M2.*nn.V + (1-M2).*G.^2;

                LR2 = LR * sqrt(1-M2^N)/(1-M1^N);
                nn.W = nn.W - nn.Mask.*LR2.* nn.M./(sqrt(nn.V)+E);
                
                nn.W(isnan(nn.W)) = 0;
                nn.W(isinf(nn.W)) = 0;
                                
                if(i > 1)
                    nn.dE = (nn.W(:,2:end)' * nn.dE) .* nn.dAct(nn.A{i},nn.Acts);
                end
            end
 
            nn.dE = full(nn.dE(1:nn.Ins,:));
            
            if(nn.PrevDimCount == 2)
                nn.dE = reshape(nn.dE,[nn.PrevDim,size(nn.dE,2)]);
            end
            
            nn.Prev.dE = nn.dE;
            
        end
        function [nn3, nn4] = Mate(nn1,nn2,mutRateG1,mutRateG2,mutRateG3,mutRateHid)
            if(nn1.Hiddens ~= nn2.Hiddens || nn1.Layers ~= nn2.Layers)
                error('Genome Mismatch');
            end
            nn3 = geneticLayer(nn1.Dim);
            nn4 = geneticLayer(nn1.Dim);
            nn3.Hiddens = nn1.Hiddens;
            nn4.Hiddens = nn1.Hiddens;
            nn3.Layers = nn1.Layers;
            nn4.Layers = nn1.Layers;
            nn3.Ins = nn1.Ins;
            nn4.Ins = nn1.Ins;
            
            dims = size(nn1.Gene);
            
            % Gene 1
            Gene1_nn1 = nn1.Gene(:);
            Gene1_nn2 = nn2.Gene(:);
            geneCount = numel(Gene1_nn1);
            
            crossPoint = randi([1,geneCount]);
            
            Gene1_nn3 = Gene1_nn1;
            Gene1_nn4 = Gene1_nn2;
            Gene1_nn3(1:crossPoint) = Gene1_nn2(1:crossPoint);
            Gene1_nn4(1:crossPoint) = Gene1_nn1(1:crossPoint);
            Gene1_nn3 = sparse(xor(Gene1_nn3, rand(geneCount,1)<mutRateG1));
            Gene1_nn4 = sparse(xor(Gene1_nn4, rand(geneCount,1)<mutRateG1));
            nn3.Gene = reshape(Gene1_nn3,dims);
            nn4.Gene = reshape(Gene1_nn4,dims);
            
            % Gene 2
            Gene2_nn1 = nn1.Acts;
            Gene2_nn2 = nn2.Acts;
            geneCount = numel(Gene2_nn1);
            
            crossPoint = randi([1,geneCount]);
            
            Gene2_nn3 = Gene2_nn1;
            Gene2_nn4 = Gene2_nn2;
            Gene2_nn3(1:crossPoint) = Gene2_nn2(1:crossPoint);
            Gene2_nn4(1:crossPoint) = Gene2_nn1(1:crossPoint);
            RN = rand(geneCount,1) < mutRateG2;
            Gene2_nn3(RN) = randi([0,3],sum(RN),1);
            RN = rand(geneCount,1) < mutRateG2;
            Gene2_nn4(RN) = randi([0,3],sum(RN),1);
            nn3.Acts = Gene2_nn3;
            nn4.Acts = Gene2_nn4;
            
            
            % Gene 3
            if(rand() < mutRateG3)
                if(nn3.Layers > 1)
                    if(rand()<0.5)
                        nn3.Layers = nn3.Layers + 1;
                    else
                        nn3.Layers = nn3.Layers - 1;
                    end
                else
                    nn3.Layers = nn3.Layers + 1;
                end
            end
            if(rand() < mutRateG3)
                if(nn4.Layers > 1)
                    if(rand()<0.5)
                        nn4.Layers = nn4.Layers + 1;
                    else
                        nn4.Layers = nn4.Layers - 1;
                    end
                else
                    nn4.Layers = nn4.Layers + 1;
                end
            end
            
            if(rand() < mutRateHid)
%                 dims = size(nn3.Gene);
%                 nn3.Gene = [[nn3.Gene;rand(1,dims(2))<mutRateG1],rand(dims(1)+1,1)<mutRateG1];
%                 nn3.Acts = [nn3.Acts;randi([0,3])];
%                 nn3.Hiddens = nn3.Hiddens + 1;
                if(nn3.Hiddens > 0)
                    if(rand()<0.5)
                        dims = size(nn3.Gene);
                        nn3.Gene = [[nn3.Gene;rand(1,dims(2))<mutRateG1],rand(dims(1)+1,1)<mutRateG1];
                        %nn3.Acts = [nn3.Acts;2];
                        nn3.Acts = [nn3.Acts;randi([0,3])];
                        nn3.Hiddens = nn3.Hiddens + 1;
                    else
                        H = randi([1,nn3.Hiddens]);
                        nn3.Gene(nn3.Ins+nn3.Dim+H,:) = [];
                        nn3.Gene(:,nn3.Ins+nn3.Dim+H) = [];
                        nn3.Acts(nn3.Ins+nn3.Dim+H) = [];
                        nn3.Hiddens = nn3.Hiddens - 1;
                    end
                else
                    dims = size(nn3.Gene);
                    nn3.Gene = [[nn3.Gene;rand(1,dims(2))<mutRateG1],rand(dims(1)+1,1)<mutRateG1];
                    %nn3.Acts = [nn3.Acts;2];
                    nn3.Acts = [nn3.Acts;randi([0,3])];
                    nn3.Hiddens = nn3.Hiddens + 1;
                end
            end
            if(rand() < mutRateHid)
%                 dims = size(nn4.Gene);
%                 nn4.Gene = [[nn4.Gene;rand(1,dims(2))<mutRateG1],rand(dims(1)+1,1)<mutRateG1];
%                 nn4.Acts = [nn4.Acts;randi([0,3])];
%                 nn4.Hiddens = nn4.Hiddens + 1;                
                if(nn4.Hiddens > 0)
                    if(rand()<0.5)
                        dims = size(nn4.Gene);
                        nn4.Gene = [[nn4.Gene;rand(1,dims(2))<mutRateG1],rand(dims(1)+1,1)<mutRateG1];
                        %nn4.Acts = [nn4.Acts;2];
                        nn4.Acts = [nn4.Acts;randi([0,3])];
                        nn4.Hiddens = nn4.Hiddens + 1;
                    else
                        H = randi([1,nn4.Hiddens]);
                        nn4.Gene(nn4.Ins+nn4.Dim+H,:) = [];
                        nn4.Gene(:,nn4.Ins+nn4.Dim+H) = [];
                        nn4.Acts(nn4.Ins+nn4.Dim+H) = [];
                        nn4.Hiddens = nn4.Hiddens - 1;
                    end
                else
                    dims = size(nn4.Gene);
                    nn4.Gene = [[nn4.Gene;rand(1,dims(2))<mutRateG1],rand(dims(1)+1,1)<mutRateG1];
                    %nn4.Acts = [nn4.Acts;2];
                    nn4.Acts = [nn4.Acts;randi([0,3])];
                    nn4.Hiddens = nn4.Hiddens + 1;
                end
            end
            if(size(nn3.Gene,1)~=nn1.Ins+nn1.Dim+nn3.Hiddens)
               disp('test')
            end
            if(size(nn4.Gene,1)~=nn1.Ins+nn1.Dim+nn4.Hiddens)
               disp('test')
            end
            nn3.init(nn1.Prev,nn1.Next);
            nn4.init(nn1.Prev,nn1.Next);
        end
    end
    methods(Static)
        function Y = act(X,A)
            Y = zeros(size(X));
            for i = 1:numel(A)
                switch A(i)
                    case {'linear',0}
                        Y(i,:) = X(i,:);
                    case {'softplus',1}
                        Y(i,:) = log(1+exp(X(i,:)));
                    case {'tanh',2}
                        Y(i,:) = tanh(X(i,:));
                    case {'sigmoid',3}
                        Y(i,:) = 1./(1+exp(-X(i,:)));
                    otherwise
                        Y(i,:) = X(i,:);
                end

                
            end
            Y = sparse(Y);
        end
        function Y = dAct(X,A)
            if(size(X,1) ~= numel(A))
                disp('help')
            end
            Y = zeros(size(X));
            for i = 1:numel(A)
                switch A(i)
                    case {'linear',0}
                        Y(i,:) = ones(size(X(i,:)));
                    case {'softplus',1}
                        Y(i,:) = 1./(1+exp(-X(i,:)));
                    case {'tanh',2}
                        Y(i,:) = 1 - tanh(X(i,:)).^2;
                    case {'sigmoid',3}
                        Y(i,:) = exp(-X(i,:))./(1+exp(-X(i,:)));
                    otherwise
                        Y(i,:) = ones(size(X(i,:)));
                end
            end
            Y = sparse(Y);
        end
    end
end