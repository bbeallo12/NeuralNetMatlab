%layer for inputting and preparing vector data
classdef vectorInputLayer < layer
   properties
   end
   methods
       function nn = vectorInputLayer(Dim)
           nn.Dim = Dim;
           nn.DimCount = 0;
       end
       function init(nn, ~, Next)
           nn.Next = Next;
       end
       function setInputs(nn,Inputs)
           nn.Values = Inputs;
       end
       function forwardPass(nn)
           nn.Next.Values = nn.Values;
       end
   end
   methods(Static)
       
   end
end