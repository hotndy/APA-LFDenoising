% function [pInImgs, pInFeat, pRef, refPos] = ComputeTrainingExamples(curFullLF, curInputLF)
function [pDpMSP,pConf,pCV,pCE,pInDisparity] = ComputeTrainingExamples(...
            disparityMSP,confidence,centralView,centralEdge,gtDisparity)
global param;
% global inputView;
padSize = param.mBorder;
% numRefs = param.numRefs;
patchSize = param.patchSize;
stride = param.stride;
% origAngRes = param.origAngRes;

[height, width, ~,] = size(centralView);
%%%%%%%%%%%%% preparing input MSP %%%%%%%%%%%%%%%
% disparityMSP = CropImg(disparityMSP, cropSize);
disparityMSP = PadImg(disparityMSP, padSize);
pDpMSP = GetPatches(disparityMSP, patchSize, stride);

%%%%%%%%%%%%% preparing input confidence  %%%%%%%%%%%%%%%
confidence = PadImg(confidence, padSize);
pConf = GetPatches(confidence, patchSize, stride);

%%%%%%%%%%%%% preparing input central View  %%%%%%%%%%%%%%%
centralView = PadImg(centralView, padSize);
pCV = GetPatches(centralView, patchSize, stride);

%%%%%%%%%%%%% preparing input central Edge  %%%%%%%%%%%%%%%
centralEdge = PadImg(centralEdge, padSize);
pCE = GetPatches(centralEdge, patchSize, stride);

%%%%%%%%%%%%% preparing ground truth disparity  %%%%%%%%%%%%%%%
gtDisparity = PadImg(gtDisparity, padSize);
pInDisparity = GetPatches(gtDisparity, patchSize, stride);

%%%%%%%%%%%%% selecting random references %%%%%%%%%%
% numSeq = randperm(origAngRes^2);
% refInds = numSeq(1:numRefs);

%%%%%%%%%%%%% initializing the arrays %%%%%%%%%%%%%%
% numPatches = GetNumPatches();
% pInFeat = zeros(patchSize, patchSize, param.numDepthFeatureChannels, numPatches * numRefs);
% pRef = zeros(patchSize, patchSize, 3, numPatches * numRefs);
% refPos = zeros(2, numPatches * numRefs);
% pInFeat = zeros(patchSize, patchSize, 3, numPatches);
% pRef = zeros(patchSize, patchSize, 3, numPatches * numRefs);
% refPos = zeros(2, numPatches * numRefs);
% 
% 
% for ri = 1 : numRefs
%     
%     fprintf('Working on random reference %d of %d: ', ri, numRefs);
%     
%     [curRefInd.Y, curRefInd.X] = ind2sub([origAngRes, origAngRes], refInds(ri));
%     curRefPos.Y = GetImgPos(curRefInd.Y); curRefPos.X = GetImgPos(curRefInd.X);
%     
%     wInds = (ri-1) * numPatches + 1 : ri * numPatches;
%     
%     %%%%%%%%%%%%%%%%%%%%% preparing reference %%%%%%%%%%%%%%%%%%%%%%%%%%%
%     ref = curFullLF(:, :, :, curRefInd.Y, curRefInd.X);
%     ref = CropImg(ref, cropSize);
%     pRef(:, :, :, wInds) = GetPatches(ref, patchSize, stride);
%     
%     
%     %%%%%%%%%%%%%%%%%%%%% preparing features %%%%%%%%%%%%%%%%%%%%%%%%%
%     deltaViewY = inputView.Y - curRefPos.Y; 
%     deltaViewX = inputView.X - curRefPos.X;
% 
%     inFeat = PrepareDepthFeatures(curInputLF, deltaViewY, deltaViewX);
%     inFeat = CropImg(inFeat, cropSize);
%     pInFeat(:, :, :, wInds) = GetPatches(inFeat, patchSize, stride);
%     
%     
%     %%%%%%%%%%%%%%%%%%%%%% preparing ref positions %%%%%%%%%%%%%%%%%%%
%     refPos(1, wInds) = repmat(curRefPos.Y, [1, numPatches]);
%     refPos(2, wInds) = repmat(curRefPos.X, [1, numPatches]);
%     
%     fprintf(repmat('\b', 1, 5));
%     fprintf('Done\n');
% end



