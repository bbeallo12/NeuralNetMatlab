%layer for inputting and preparing image data
classdef imageInputLayer < layer
   properties
   end
   methods
       function nn = imageInputLayer(Dim)
           nn.Dim = Dim;
           nn.DimCount = 2;
       end
       function init(nn, ~, Next)
           nn.Next = Next;
       end
       function setInputs(nn,Inputs)
           Dim = [size(Inputs,1),size(Inputs,2),size(Inputs,3),size(Inputs,4)];
           if(Dim(1:3)~=nn.Dim)
               error('Wrong Input Dimensions (imageInputLayer)');
           end
           nn.Values = Inputs;
       end
       function forwardPass(nn)
           nn.Next.Values = nn.Values;
       end
   end
   methods(Static)
       
   end
end