% classification output layer
% must have softmaxLayer preceding it
classdef classificationOutputLayer < layer
    properties
        Targets;
        E;
        dup;
    end
    methods
        % initialize function
        function nn = classificationOutputLayer(dup)
            nn.dup = dup;
        end
        function init(nn, Prev, ~)
            if(~isa(Prev,'softmaxLayer'))
                error('classificationOutputLayer must be preceded by softmaxLayer');
            end
            nn.Prev = Prev;
            nn.Dim = Prev.Dim;
        end
        function setTargets(nn,Targets)
            nn.Targets = Targets;
        end
        % calcError: calculates the error of output
        function E = calcError(nn,Targets)
            nn.Targets = repmat(Targets,[1,nn.dup]);
            sh = size(Targets);
            Values = nn.Values(:,1:sh(2));
            
            if(sum(Values(:)==1)>0)
                Values(Values ==1) = 1-1e-16;
            end
            if(sum(Values(:)==0)>0)
                Values(Values ==0) = 1e-16;
            end
            
            E = ((Targets .* log(Targets) + (1-Targets).*log(1 - Targets))-(Targets .* log(Values) + (1-Targets).*log(1 - Values)))./numel(Targets);           
            E = sum(E(:));
        end
        % get outputs
        function Output = Output(nn)
            Output = nn.Values(:,1:end/nn.dup);
        end
        % calculated error derivative and passes to previous layer
        function backPass(nn,~,~,~,~)
            Values = nn.Values;
            
            if(sum(Values(:)==1)>0)
                Values(Values ==1) = 1-1e-16;
            end
            if(sum(Values(:)==0)>0)
                Values(Values ==0) = 1e-16;
            end
            
            nn.dE = (Values-nn.Targets)./(Values.*(1-Values));

            nn.dE(isinf(nn.dE)) = 0;
            nn.dE(isnan(nn.dE)) = 0;
            nn.Prev.dE = nn.dE;
        end
        
    end
end