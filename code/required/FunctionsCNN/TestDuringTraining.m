function error = TestDuringTraining(viewNet)

global param;
global IN;

% sceneNames = param.testNames;
% fid = fopen([param.trainNet, '/error.txt'], 'at');

% numScenes = length(sceneNames);
error = 0;

IN.currImageBatches=0;

maxNumPatches = size(IN.testData.GT,4);
IN.currImageBatches = floor(maxNumPatches / param.batchSize);
        

for it = 1 : IN.currImageBatches
    
    % startInd= mod((it-1) * param.batchSize, IN.currImageBatches) + 1;
    startInd= (it-1) * param.batchSize + 1;
    
    inAll=          im2single(IN.testData.INALL(:,:,:,startInd:startInd+ param.batchSize-1));
    gtIm=           im2single(IN.testData.GT(:,:,:,startInd:startInd+ param.batchSize-1));
    synOut=         im2single(IN.testData.SYNOUT(:,:,:,startInd:startInd+ param.batchSize-1));
    svSp=           im2single(IN.testData.SVSP(:,:,:,startInd:startInd+ param.batchSize-1));

    inAll=          padarray(inAll,[param.mBorder,param.mBorder],'both');      
    gtIm=           padarray(gtIm,[param.mBorder,param.mBorder],'both');
    synOut=       	padarray(synOut,[param.mBorder,param.mBorder],'both');
    svSp=       	padarray(svSp,[param.mBorder,param.mBorder],'both');

    % figure;imshow([showLFIM(squeeze(inT1SpMask(:,:,:,1))),squeeze(inRainMask(:,:,:,1)),squeeze(gtImSpMask(:,:,:,1))])

    if (param.useGPU)
        inAll       = gpuArray(inAll);           
        gtIm        = gpuArray(gtIm);
        synOut      = gpuArray(synOut);
        svSp        = gpuArray(svSp);
    end
    
  
    viewIdx= [1,8,57,64,12,53,23,32];
    for nnv= 1:length(viewIdx)
        nv= viewIdx(nnv);
        [viewRes] = EvaluateSystem(viewNet,inAll,...
            gtIm(:,:,nv,:),synOut(:,:,nv,:),svSp(:,:,nv,:),true,true);
        finalOut(:,:,nv,:)= viewRes(end).x;
    end
            
    % curError = ComputePSNR(dispRes(end).x.*inSpMask, gtImSpMask(:,:,1,:));
    gtImRes= gpuArray(CropImg(gtIm- repmat(inAll,1,1,size(gtIm,3),1),param.mBorder));
    % gtImRes= gpuArray(CropImg(gtIm- synOut- repmat(inAll,1,1,size(gtIm,3),1),param.mBorder));
    curError = ComputePSNR(finalOut(:,:,viewIdx,:), gtImRes(:,:,viewIdx,:));
            
    error = error + curError / IN.currImageBatches;
end

%fprintf(fid, '%f\n', error);
%fclose(fid);

error = gather(error);
