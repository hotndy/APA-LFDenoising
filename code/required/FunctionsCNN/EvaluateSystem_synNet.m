% function [finalImg, depthRes, colorRes] = EvaluateSystem(depthNet, colorNet, images, refPos, isTraining, depthFeatures, reference, isTestDuringTraining)
function [dispRes] = EvaluateSystem(dispNet,inAll,inEW,inNS,gtIm,isTraining,isTestDuringTraining)

global param;

if (~exist('isTraining', 'var') || isempty(isTraining))
    isTraining = false;
end

if (~exist('isTestDuringTraining', 'var') || isempty(isTestDuringTraining))
    isTestDuringTraining = false;
end

inFeatures(:,:,1:8,:)      = inEW- repmat(inAll,1,1,size(inEW,3),1);
inFeatures(:,:,9:16,:)     = inNS- repmat(inAll,1,1,size(inEW,3),1);

if(isTraining)   
    if (param.useGPU)
        gtImRes= gpuArray(CropImg(gtIm- repmat(inAll,1,1,size(gtIm,3),1),param.mBorder));
    else
        gtImRes= CropImg(gtIm- repmat(inAll,1,1,size(gtIm,3),1),param.mBorder);
    end
end

if (param.useGPU)
	inFeatures = gpuArray(inFeatures);
end

dispRes = EvaluateNet(dispNet, inFeatures, [], true);
finalOut= dispRes(end).x;

%% Backpropagation    
if(isTraining && ~isTestDuringTraining)   
    dzdx = vl_nnpdist(finalOut, gtImRes, 2, 1, 'noRoot', true, 'aggregate', true); % /length(find(inSpMask));
%     dzdx = vl_nnpdist(finalOut.*outRainMask.*outSpMask, ...
%                     gtImSpMask.*outRainMask.*outSpMask, 2, 1, ...
%                     'noRoot', true, 'aggregate', true);% /length(find(outSpMask.*outRainMask));
    dispRes(end).dzdx = dzdx;
    dispRes = EvaluateNet(dispNet,inFeatures, dispRes, false);    
end

%{
figure;
for n=1:param.batchSize
    numIm= 4;
    subplot(numIm,param.batchSize,n);imshow(inImgs(:,:,:,n)/255);
	subplot(numIm,param.batchSize,n+1*param.batchSize);imshow(finalOut(:,:,1,n));  
    subplot(numIm,param.batchSize,n+2*param.batchSize);imshow(finalImg(:,:,1,n));
    subplot(numIm,param.batchSize,n+3*param.batchSize);imshow(bgrf(:,:,:,n));
end
%}
