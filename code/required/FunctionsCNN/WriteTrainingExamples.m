function WriteTrainingExamples(pInImgs,pInDp,pEdgeIn,pEdgeBkg,...
        pEdgeRfc, outputDir, writeOrder, startInd, createFlag, arraySize)

chunkSize = 400; %1000;
fileName = sprintf('%s/training.h5', outputDir);

% [~, numElements] = size(refPos);
[~,~,~,numElements] = size(pInImgs);

for k=1;numElements
    
j = k + startInd - 1

curInImgs= pInImgs(:,:,:,k);
curInDp= pInDp(:,:,:,k);
curInEdge= pEdgeIn(:,:,:,k);
curRefBkg= pEdgeBkg(:,:,:,k);
curRefRfc= pEdgeRfc(:,:,:,k);

SaveHDF(fileName, '/INIMG', single(curInImgs), PadWithOne(size(curInImgs), 4), [1, 1, 1, writeOrder(j)], chunkSize, createFlag, arraySize);
SaveHDF(fileName, '/INDP',  single(curInDp),   PadWithOne(size(curInDp), 4),   [1, 1, 1, writeOrder(j)], chunkSize, createFlag, arraySize);
SaveHDF(fileName, '/INEDG', single(curInEdge), PadWithOne(size(curInEdge), 4), [1, 1, 1, writeOrder(j)], chunkSize, createFlag, arraySize);
SaveHDF(fileName, '/GTBKG', single(curRefBkg), PadWithOne(size(curRefBkg), 4), [1, 1, 1, writeOrder(j)], chunkSize, createFlag, arraySize);
SaveHDF(fileName, '/GTRFC', single(curRefRfc), PadWithOne(size(curRefRfc), 4), [1, 1, 1, writeOrder(j)], chunkSize, createFlag, arraySize);

end

% for k = 1 : numElements
%     
%     j = k + startInd - 1
%     
%     curInImgs = inImgs(:, :, :, k);
%     curInFeat = inFeat(:, :, :, k);
%     curRef = ref(:, :, :, k);
%     curRefPos = refPos(:, k);
%     
%     SaveHDF(fileName, '/IN', single(curInImgs), PadWithOne(size(curInImgs), 4), [1, 1, 1, writeOrder(j)], chunkSize, createFlag, arraySize);
%     SaveHDF(fileName, '/FT', single(curInFeat), PadWithOne(size(curInFeat), 4), [1, 1, 1, writeOrder(j)], chunkSize, createFlag, arraySize);
%     SaveHDF(fileName, '/GT', single(curRef), PadWithOne(size(curRef), 4), [1, 1, 1, writeOrder(j)], chunkSize, createFlag, arraySize);
%     SaveHDF(fileName, '/RP', single(curRefPos), size(curRefPos), [1, writeOrder(j)], chunkSize, createFlag, arraySize);
%     
%     createFlag = false;
% end
% 


