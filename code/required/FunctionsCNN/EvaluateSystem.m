function [viewRes] = EvaluateSystem(viewNet,inAll,gtImSAI,synOutSAI,svSpSAI,isTraining,isTestDuringTraining)

global param;

if (~exist('isTraining', 'var') || isempty(isTraining))
    isTraining = false;
end

if (~exist('isTestDuringTraining', 'var') || isempty(isTestDuringTraining))
    isTestDuringTraining = false;
end

% inFeatures(:,:,1:64,:)      = synOut- repmat(inAll,1,1,size(synOut,3),1);
% inFeatures(:,:,1:64,:)      = synOut;

if(isTraining)   
    gtImRes= gpuArray(CropImg(gtImSAI-inAll,param.mBorder));
end
       
if (param.useGPU)
    inFeatures(:,:,1,:) = gpuArray(synOutSAI);
    inFeatures(:,:,2,:) = gpuArray(svSpSAI-inAll);
end

viewRes = EvaluateNet(viewNet, inFeatures, [], true);
finalOut= viewRes(end).x;



%% Backpropagation    
if(isTraining && ~isTestDuringTraining)   
    dzdx = vl_nnpdist(viewRes(end).x, gtImRes, 2, 1, 'noRoot', true, 'aggregate', true); % /length(find(inSpMask));
    viewRes(end).dzdx = dzdx; % assign last layer errors
    viewRes = EvaluateNet(viewNet,inFeatures, viewRes, false); % calculate gradients
end

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
