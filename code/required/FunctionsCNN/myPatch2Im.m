function [finalImg]= Patch2Im(blkMatrix)

global param;

param.cropSizeTraining= -param.mBorder;
height = ceil((param.height-param.patchSize)/param.stride)*param.stride+ param.patchSize;
width = ceil((param.width-param.patchSize)/param.stride)*param.stride+ param.patchSize;
patchSize = param.patchSize;
stride = param.stride;

numPatchesX = ceil((width-patchSize)/stride)+1;
numPatchesY = ceil((height-patchSize)/stride)+1;
numPatches = numPatchesY * numPatchesX;

if(numPatches~=size(blkMatrix,4))
    disp('wrong input patch number');
    return;
end

cropBorder= 0;
cropPatchSize= size(blkMatrix,1);
WU= ones(cropPatchSize,cropPatchSize);

ImgOut= zeros(height+2*cropBorder,width+2*cropBorder);
Mout= zeros(height+2*cropBorder,width+2*cropBorder);
b= gather(blkMatrix);
count = 0;

for iX = cropBorder+1 : stride : width - patchSize + cropBorder + 1
	for iY = cropBorder+1 : stride : height - patchSize + cropBorder+ 1
        count= count+1;
        if(count>numPatches)
            break;
        end
        ImgOut(iY:iY+cropPatchSize-1, iX:iX+cropPatchSize-1, :)= ...
            ImgOut(iY:iY+cropPatchSize-1, iX:iX+cropPatchSize-1, :)+ b(:, :, :,count);
        Mout(iY:iY+cropPatchSize-1, iX:iX+cropPatchSize-1, :)= ...
            Mout(iY:iY+cropPatchSize-1, iX:iX+cropPatchSize-1, :)+ WU;
        
    end
end

finalImg= (ImgOut./Mout);
finalImg= finalImg(1:param.height,1:param.width);

end




% function patches = GetPatches(input, patchSize, stride)
% 
% [height, width, depth] = size(input);
% 
% numPatches = (floor((width-patchSize)/stride)+1)*(floor((height-patchSize)/stride)+1);
% patches = zeros(patchSize, patchSize, depth, numPatches);
% 
% count = 0;
% for iX = 1 : stride : width - patchSize + 1
%     for iY = 1 : stride : height - patchSize + 1
%         count = count + 1;
%         patches(:, :, :, count) = input(iY:iY+patchSize-1, iX:iX+patchSize-1, :);
%     end
% end