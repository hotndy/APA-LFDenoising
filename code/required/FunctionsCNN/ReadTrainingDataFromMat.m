% function [images, features, reference, refPos] = ReadTrainingData(fileName, isTraining, it)
function [inImgs,inDP,inEdges,bgEdges,rfEdges] = ReadTrainingDataFromMat(fileName, isTraining, it)

global param;
batchSize = param.batchSize;
% depthBorder = param.depthBorder;
% colorBorder = param.colorBorder;
mBorder = param.mBorder;
useGPU = param.useGPU;

if (~exist('isTraining', 'var') || isempty(isTraining))
    isTraining = true;
end

fileInfo = h5info(fileName);
numItems = length(fileInfo.Datasets);
maxNumPatches = fileInfo.Datasets(1).Dataspace.Size(end);
numImages = floor(maxNumPatches / batchSize) * batchSize;

if (isTraining)
    startInd = mod((it-1) * batchSize, numImages) + 1;
else
    startInd = 1;
    batchSize = 1;
end

features = []; reference = []; images = []; refPos = [];

for i = 1 : numItems
    
    dataName = fileInfo.Datasets(i).Name;
    
    switch dataName
        
        case 'GTBKG'
            s = fileInfo.Datasets(i).Dataspace.Size;
            bgEdges = h5read(fileName, '/GTBKG', [1, 1, 1, startInd], [s(1), s(2), s(3), batchSize]);
            % bgEdges = single(bgEdges);
            bgEdges = single(CropImg(bgEdges, mBorder));
            if (useGPU)
                bgEdges = gpuArray(bgEdges);
            end
        
        case 'GTRFC'
            s = fileInfo.Datasets(i).Dataspace.Size;
            rfEdges = h5read(fileName, '/GTRFC', [1, 1, 1, startInd], [s(1), s(2), s(3), batchSize]);
            rfEdges = single(CropImg(rfEdges, mBorder));
            if (useGPU)
                rfEdges = gpuArray(rfEdges);
            end
            
        case 'INDP'
            s = fileInfo.Datasets(i).Dataspace.Size;
            inDP = h5read(fileName, '/INDP', [1, 1, 1, startInd], [s(1), s(2), s(3), batchSize]);
            inDP = single(inDP);
            if (useGPU)
                inDP = gpuArray(inDP);
            end
            
        case 'INEDG'
            s = fileInfo.Datasets(i).Dataspace.Size;
            inEdges = h5read(fileName, '/INEDG', [1, 1, 1, startInd], [s(1), s(2), s(3), batchSize]);
            inEdges = single(inEdges);
            if (useGPU)
                inEdges= gpuArray(inEdges);
            end
            
        case 'INIMG'
            s = fileInfo.Datasets(i).Dataspace.Size;
            inImgs = h5read(fileName, '/INIMG',  [1, 1, 1, startInd], [s(1), s(2), s(3), batchSize]);
            inImgs = single(inImgs);
            if (useGPU)
                inImgs = gpuArray(inImgs);
            end
    end
end

%  switch dataName
% 
%         case 'FT'
%             s = fileInfo.Datasets(i).Dataspace.Size;
%             features = h5read(fileName, '/FT', [1, 1, 1, startInd], [s(1), s(2), s(3), batchSize]);
%             features = single(features);
%             if (useGPU)
%                 features = gpuArray(features);
%             end
% 
%         case 'GT'
%             s = fileInfo.Datasets(i).Dataspace.Size;
%             reference = h5read(fileName, '/GT', [1, 1, 1, startInd], [s(1), s(2), s(3), batchSize]);
%             reference = single(CropImg(reference, depthBorder+colorBorder));
%             if (useGPU)
%                 reference = gpuArray(reference);
%             end
% 
%         case 'IN'
%             s = fileInfo.Datasets(i).Dataspace.Size;
%             images = h5read(fileName, '/IN', [1, 1, 1, startInd], [s(1), s(2), s(3), batchSize]);
%             if (useGPU)
%                 images = gpuArray(images);
%             end
% 
%         case 'RP'
%             refPos = h5read(fileName, '/RP', [1, startInd], [2, batchSize]);
%             refPos = single(refPos);
%             if (useGPU)
%                 refPos = gpuArray(refPos);
%             end
% 
%     end
    
