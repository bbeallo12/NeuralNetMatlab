% softmax activation function
classdef softmaxLayer < layer
    properties
    end
    methods
        function init(nn, Prev, Next)
            nn.Prev = Prev;
            nn.Next = Next;
            nn.Dim = Prev.Dim;
            nn.DimCount = Prev.DimCount;
        end
        function forwardPass(nn)
            if(sum(sum(isnan(nn.Values)))>0)
                disp('softmax 14')
            elseif(sum(sum(isinf(nn.Values)))>0)
                disp('softmax 16')
            end  
            
            Values = zeros(size(nn.Values));
            
            nn.Values(nn.Values>37) = 37;
            nn.Values(nn.Values<-37) = -37;
            D = sum(exp(nn.Values),1);
            
            for i = 1:nn.Dim
                Values(i,:) = exp(nn.Values(i,:))./D;
                Values(i,isnan(Values(i,:))) = 0;
                Values(i,isinf(Values(i,:))) = 0;
            end  
            
            nn.Values = Values;           
            nn.Next.Values = nn.Values;
        end
        function backPass(nn,~,~,~,~)
            if(sum(sum(isnan(nn.dE)))>0)
                disp('softmax 36')
            elseif(sum(sum(isinf(nn.dE)))>0)
                disp('softmax 38')
            end
            for k = 1:size(nn.Values,2)
                M = zeros(size(nn.Values,1));
                for i = 1:size(nn.Values,1)
                    for j = 1:size(nn.Values,1)
                        M(i,j) = nn.Values(i,k)*((i==j)-nn.Values(j,k));
                        if(sum(sum(isnan(M)))>0)
                            disp('softmax 46')
                        elseif(sum(sum(isinf(M)))>0)
                            disp('softmax 48')
                        end           
                    end
                end
                if(sum(sum(isnan(M*nn.dE(:,k))))>0)
                    disp('softmax 53')
                elseif(sum(sum(isinf(M*nn.dE(:,k))))>0)
                    disp('softmax 55')
                end                
                nn.dE(:,k) = M*nn.dE(:,k);

            end

            nn.Prev.dE = nn.dE;
        end
    end
    methods(Static)
        
    end
end