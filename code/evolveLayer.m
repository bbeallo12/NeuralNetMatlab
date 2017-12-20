classdef evolveLayer < layer
    properties
        Pop;
        mutRateG1;
        mutRateG2;
        mutRateG3;
        mutRateHid;
        Outputs;
        Species;
        ValuesIn;
        E;
        ValuesOut;
        dEOut;
        Best;
        Cycles;
        Dist;
        Info;
        hidRange;
        layRange;
        Buckets;
        BuckSize;
        plotNum;
        WS1 = nan;
    end
    methods
        % instantiate layer
        function nn = evolveLayer(Dim, Pop, mutRateG1, mutRateG2, mutRateG3, mutRateHid, Cycles, hidRange, layRange, plotNum)
            nn.hidRange = hidRange;
            nn.layRange = layRange;
            nn.plotNum = plotNum;
            nn.mutRateG1 = mutRateG1;
            nn.mutRateG2 = mutRateG2;
            nn.mutRateG3 = mutRateG3;
            nn.mutRateHid = mutRateHid;
            nn.Pop = Pop;
            nn.Best = 1;
            nn.DimCount = 0;
            nn.Dim = Dim;
            nn.Cycles = Cycles;
            nn.Dist = zeros(Pop,1);
            nn.Info = zeros(6,Pop);
        end
        % initialize layer
        % this is called by the neuralNet class
        function init(nn, Prev, Next)
            nn.Prev = Prev;
            nn.Next = Next;
            nn.Species = cell(nn.hidRange(2)+1,nn.layRange(2));
            nn.ValuesOut = cell(1,nn.Pop);
            for i = 1:nn.Pop
                child = geneticLayer(nn.Dim);
                child.Hiddens = randi(nn.hidRange);
                child.Layers = randi(nn.layRange);
                child.init(nn,nn);
                nn.Species{child.Hiddens+1, child.Layers}(end+1) = child;
            end
            
            disp(nn.Species)
        end
        function forwardPass(nn)
            % apply weights to data
            % if data dimension is higher than 0, reduce dimensionality
            nn.ValuesIn = nn.Values;
            
            dims = size(nn.Species);
            count = 1;
            for row = 1:dims(1)
                for col = 1:dims(2)
                    for net = 1:numel(nn.Species{row,col})
                        nn.Species{row,col}(net).Values = nn.ValuesIn;
                        nn.Species{row,col}(net).forwardPass();
                        nn.ValuesOut{count} = nn.Values;

                        count = count + 1;
                    end
                end
            end
            
            output = cell2mat(nn.ValuesOut);
            shape = size(output);
            output = reshape(output,[shape(1)*shape(2)/nn.Pop, nn.Pop]);

            output(:,[1,nn.Best]) = output(:,[nn.Best,1]);
            nn.Values = reshape(output,shape);
            
            if(sum(sum(isnan(nn.Values)))>0)
                disp('why')
            elseif(sum(sum(isinf(nn.Values)))>0)
                disp('why')
            end              
            
            nn.Next.Values = nn.Values;
        end
        function backPass(nn,LR,M1,M2,N)
            
            if(sum(sum(isnan(nn.dE)))>0)
                disp('evolveLayer 96')
            elseif(sum(sum(isinf(nn.dE)))>0)
                disp('evolveLayer 98')
            end   
            shape = size(nn.dE);
            nn.dE = reshape(nn.dE,[shape(1)*shape(2)/nn.Pop, nn.Pop]);
            nn.dE(:,[1,nn.Best]) = nn.dE(:,[nn.Best,1]);
            nn.dE = reshape(nn.dE,shape);
            dE = mat2cell(nn.dE,shape(1),ones(1,nn.Pop).*shape(2)/nn.Pop);
            
            if(sum(nn.dE(:)==0)>0)
                disp('zeros')
            end
            
            M = mod(N-1,nn.Cycles)+1;
            
            dims = size(nn.Species);
            count = 1;
            for row = 1:dims(1)
                for col = 1:dims(2)
                    for net = 1:numel(nn.Species{row,col})
                        nn.Species{row,col}(net).dE = dE{count};
                        nn.Species{row,col}(net).backPass(LR,M1,M2,M);
                        nn.Dist(count) = nn.Species{row,col}(net).Dist;
                        nn.Info(:,count) = [row;...
                                            col;...
                                            net;...
                                            nn.Species{row,col}(net).Params;...
                                            nn.Species{row,col}(net).Layers;...
                                            nn.Species{row,col}(net).Hiddens];
                        if(count == 1)
                            nn.dEOut = nn.dE;
                        else
                            nn.dEOut = nn.dEOut + nn.dE;
                        end
                        count = count + 1;
                    end
                end
            end
            nn.Prev.dE = nn.dEOut/nn.Pop;
            
            figure(nn.plotNum*2)
            subplot(4,1,1)
            It = nn.bestFit(1);
            D1 = nn.Dist(It);
            bar(D1,'b');
            xlim([0,nn.Pop+1])
            title({['Generation: ',num2str(floor(N/nn.Cycles)),'     Max D1: ',num2str(max(D1)),'  Min D1: ',num2str(min(D1)),'  Avg D1: ',num2str(sum(D1)/numel(D1))]})
            
            subplot(4,1,2)
            params = nn.Info(4,It);
            bar(params,'r');
            xlim([0,nn.Pop+1])
            title({['Max params: ',num2str(max(params)),'  Min params: ',num2str(min(params)),'  Avg params: ',num2str(sum(params)/numel(params))]})
            
            subplot(4,1,3)
            layers = nn.Info(5,It);
            bar(layers,'r');
            xlim([0,nn.Pop+1])
            title({['Max layers: ',num2str(max(layers)),'  Min layers: ',num2str(min(layers)),'  Avg layers: ',num2str(sum(layers)/numel(layers))]})
            
            subplot(4,1,4)
            hiddens = nn.Info(6,It);
            bar(hiddens,'r');
            xlim([0,nn.Pop+1])
            title({['Max hiddens: ',num2str(max(hiddens)),'  Min WS2: ',num2str(min(hiddens)),'  Avg WS2: ',num2str(sum(hiddens)/numel(hiddens))]})
        
            drawnow limitrate   
            
            if(M == nn.Cycles)
                nn.WS1 = nan;
                nn.evolve();
            end
        end
        function setE(nn,E)
            nn.E = E;
        end
        function best = getBest(nn,n)
            I1 = nn.bestFit(nn.plotNum);
            row = nn.Info(1,I1(n));
            col = nn.Info(2,I1(n));
            net = nn.Info(3,I1(n));
            best = nn.Species{row,col}(net);
        end
        function y = getProbs(nn,a,c)
            y = linspace(0,1,nn.Pop);
            y = 1./(1+exp(a.*(-y+c)));
            y = (y - min(y))./(max(y) - min(y));
        end
        function I1 = bestFit(nn,p)
            %nn.Dist(nn.Dist==0) = max(nn.Dist);
            [Dst,I1] = sort(nn.Dist);
            if p == 1
                a = 0.1;
                nn.BuckSize = find(Dst<(1-a)*min(Dst)+a*sum(Dst)/numel(Dst),1,'last');
                nn.Buckets = ceil(nn.Pop/nn.BuckSize);
                for i = 1:nn.Buckets
                    if(i == nn.Buckets || i == nn.Pop)
                        It = I1((i-1)*nn.BuckSize+1:end);
                        [~,I2] = sort(nn.Info(4,It));
                        I2 = It(I2);
                        I1((i-1)*nn.BuckSize+1:end) = I2;
                        if(i == nn.Pop)
                            break;
                        end
                    else
                        It = I1((i-1)*nn.BuckSize+1:i*nn.BuckSize);
                        [~,I2] = sort(nn.Info(4,It));
                        I2 = It(I2);
                        I1((i-1)*nn.BuckSize+1:i*nn.BuckSize) = I2;
                    end
                end
            end
        end

        function evolve(nn)
            I1 = nn.bestFit(1);
            I2 = zeros(size(I1));
            probs = nn.getProbs(10,0.5);
            alive = 0;
            for i = 1:nn.Pop
                if(rand()<probs(i))
                    row = nn.Info(1,I1(i));
                    col = nn.Info(2,I1(i));
                    net = nn.Info(3,I1(i));

                    nn.Info(3,sum([nn.Info(1:2,:)==[row;col];nn.Info(3,:)>net],1)==3) =  nn.Info(3,sum([nn.Info(1:2,:)==[row;col];nn.Info(3,:)>net],1)==3) - 1;

                    nn.Species{row,col}(net) = [];
                else
                    alive = alive + 1;
                    I2(alive) = I1(i);
                end
            end
            
            It = I2(1:alive);
            [~,It2] = min(nn.Dist(It));
            nn.Best = It(It2);
            
            children = 0;

            Species2 = nn.Species;

            dims = size(Species2);
            for row = 1:dims(1)
                for col = 1:dims(2)
                    for net = 1:numel(Species2{row,col})
                        Species2{row,col}(net).init(nn,nn);
                    end
                end
            end

            while(alive + children < nn.Pop)
                select = randi([1,alive]);
                row = nn.Info(1,I2(select));
                col = nn.Info(2,I2(select));
                net1 = nn.Info(3,I2(select));
                net2 = randi([1,numel(nn.Species{row,col})]);
                mate1 = nn.Species{row, col}(net1);
                mate2 = nn.Species{row, col}(net2);

                [child1, child2] = mate1.Mate(mate2,nn.mutRateG1,nn.mutRateG2,nn.mutRateG3,nn.mutRateHid);
                dims = size(Species2);
                if(child1.Hiddens+1>dims(1) || child1.Layers>dims(2))
                    Species2{child1.Hiddens+1, child1.Layers} = child1;
                else
                    Species2{child1.Hiddens+1, child1.Layers}(end+1) = child1;
                end

                children = children + 1;
                if(alive + children < nn.Pop)
                    dims = size(Species2);
                    if(child2.Hiddens+1>dims(1) || child2.Layers>dims(2))
                        Species2{child2.Hiddens+1, child2.Layers} = child2;
                    else
                        Species2{child2.Hiddens+1, child2.Layers}(end+1) = child2;
                    end
                    children = children + 1;
                end
            end
            nn.Species = Species2;
            disp(nn.Species)
        end
    end
end