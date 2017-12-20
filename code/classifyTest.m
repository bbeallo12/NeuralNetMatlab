% test classification of trained network
clc;

images = zeros(28,28,3,10);
for i = 0:9
    images(:,:,:,i+1) = double(1.0*imread(['CustomTestDigits\testimage',num2str(i),'.png'],'png'))/255;
end

[Output, M] = NN.feedForward(images(:,:,1,:), 0);
disp([num2cell(Output);num2cell(M)]');

I = 0; % digit 4

% figure(1)
% for i = 1:30
% subplot(6,5,i)
% imshow(NN.Layers{4}.Values(:,:,i,I+1));
% end
% figure(2)
% for i = 1:30
% subplot(6,5,i)
% imshow(NN.Layers{7}.Values(:,:,i,I+1));
% end
% figure(3)
% for i = 1:30
% subplot(6,5,i)
% imshow(NN.Layers{9}.Values(:,:,i,I+1));
% end
% 
% figure(4)
% subplot(1,1,1)
% imshow(reshape(NN.Layers{11}.Values(:,I+1),[10,10]),'InitialMagnification','fit');
% 
% figure(5)
% for i = 1:10
% subplot(2,5,i)
% imshow(NN.Layers{13}.Values(i,I+1));
% end