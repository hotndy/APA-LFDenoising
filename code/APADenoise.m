clearvars; clearvars -global;
warning off;

addpath(genpath('required/'));
vl_setupnn();
InitSystemParam();
global param;

% load trained viewNet and synNet
while true 
noiseLevel= input('Please chose noise level. [10, 20, 50]:'); 
param.noiseLevel= noiseLevel;
SynNetList= dir(sprintf('trainedNets\\SynNet-N%d*.mat',noiseLevel));
if(length(SynNetList)==0)
    disp('Please choose available noise level only.');
    continue;
end
load(['trainedNets\',SynNetList(end).name]);
param.synNet= dispNet;
clear dispNet testError totalTrainedBatch SynNetList;

ViewNetList= dir(sprintf('trainedNets\\ViewNet-N%d*.mat',noiseLevel));
if(length(ViewNetList)==0)
    disp('Please choose available noise level only.');
    continue;
end
load(['trainedNets\',ViewNetList(end).name]);
param.viewNet= viewNet;
clear viewNet testError totalTrainedBatch ViewNetList;

break;

end


IN.fileFolder= ['dataLF\'];
IN.list = dir([IN.fileFolder,'\*_eslf.png']);
% choose sequence
for nf=1:length(IN.list)
    fprintf(['%d. ',IN.list(nf).name,'\n'],nf);
end 
numF= input('Please chose rain sequence: ');  

for nn= 1:length(numF)
    
SV=[];
SVG= [];

% Load ESLF file and trim angular to 8x8
IN.currFileName= IN.list(numF(nn)).name;   
LF= imread([IN.fileFolder,IN.currFileName]);    

tmpVal= 14;
totalNum= 8;
nStart= floor(tmpVal/2)-floor((totalNum+1)/2)+1;    
for nr=1:totalNum
    for nc= 1:totalNum
        SV(:,:,:,nr,nc)= im2double(LF(nStart+nr-1:tmpVal:end,nc+nStart-1:tmpVal:end,:));
        SVG(:,:,nr,nc)= rgb2gray(SV(:,:,:,nr,nc));
    end
end
clear LF SVN;

[mRow,mCol,nCC,nInput.angDim,~]= size(SVG);

% add noise
sigma= param.noiseLevel;
SVGN= [];
for nr=1:totalNum
    for nc= 1:totalNum
        SVGN(:,:,nr,nc)= SVG(:,:,nr,nc)+ randn(size(SVG,1),size(SVG,2))*sigma/255;
    end
end    

% calculate central view PSNR
nInput.angDim= totalNum;
nInput.sigma= sigma;
  
% pad the borders
padRow= (ceil((mRow- param.patchSize)/param.stride)*param.stride+ param.patchSize)-mRow;
padCol= (ceil((mCol- param.patchSize)/param.stride)*param.stride+ param.patchSize)-mCol;
SVGNex= padarray(SVGN,[padRow padCol],0,'post');
SVGex= padarray(SVG,[padRow padCol],0,'post');
SVGNvec= reshape(SVGNex,mRow+padRow,mCol+padCol,nInput.angDim^2);

count= 0;

nInput.AvgAll=  mean(mean(SVGNex,3),4); 
nInput.AvgEW=   squeeze(mean(SVGNex,4));
nInput.AvgNS=   squeeze(mean(SVGNex,3));
%{
figure;imshow(nInput.AvgAll);
figure;imshow(nInput.AvgEW);
figure;imshow(nInput.AvgNS);
%}

% prepare data for the current image 
INALL=[]; INEW =[]; INNS =[]; GT	=[]; gfiltEW=[]; gfiltNS=[];
for nv=1:nInput.angDim
    gfiltEW(:,:,nv)= guidedfilter(nInput.AvgEW(:,:,nv), nInput.AvgAll, 9, 10^-6);
    gfiltNS(:,:,nv)= guidedfilter(nInput.AvgNS(:,:,nv), nInput.AvgAll, 9, 10^-6);
end
    
INALL=  im2single(GetPatches(nInput.AvgAll, param.patchSize, param.stride));
INEW =  im2single(GetPatches(gfiltEW, param.patchSize, param.stride));
INNS =  im2single(GetPatches(gfiltNS, param.patchSize, param.stride));

GT=  im2single(GetPatches(SVGNex, param.patchSize, param.stride));

% prepare data for the current image                
maxNumPatches = size(INALL,4);
IN.currImageBatches = ceil(maxNumPatches / param.batchSize);

% viewNet components
spFilt= []; SVSP= [];
for nr=1:8
    for nc=1:8
    % [spPSNR(nr,nc),SVSP(:,:,nr,nc)]= BM3D(SVG(:,:,nr,nc), SVGN(:,:,nr,nc), noiseLevel, 'np', 0);
        spFilt(:,:,nr,nc) = imgaussfilt(SVGNex(:,:,nr,nc),noiseLevel/10);
    end
end
SVSP =  im2single(GetPatches(spFilt, param.patchSize, param.stride));
    
startBatch=1;

for it=1:IN.currImageBatches

  %% main optimization
    startInd= (it-1) * param.batchSize +1;
    endInd= min(startInd+ param.batchSize-1,maxNumPatches);
    inAll=          INALL(:,:,:,startInd:endInd);
    inEW=           INEW(:,:,:,startInd:endInd);
    inNS=           INNS(:,:,:,startInd:endInd);
    gtIm=           GT(:,:,:,startInd:endInd);

    inAll=          padarray(inAll,[param.mBorder,param.mBorder],'both');
    inEW=           padarray(inEW,[param.mBorder,param.mBorder],'both');
    inNS=           padarray(inNS,[param.mBorder,param.mBorder],'both');
    gtIm=           padarray(gtIm,[param.mBorder,param.mBorder],'both');

    if (param.useGPU)
        inAll       = gpuArray(inAll);
        inEW        = gpuArray(inEW);
        inNS        = gpuArray(inNS);
        gtIm        = gpuArray(gtIm);
    end

    synOut= [];
    [synRes] = EvaluateSystem_synNet(param.synNet,inAll,inEW,inNS,gtIm,false,false);        
    synOut= synRes(end).x;
    synOut=       	padarray(synOut,[param.mBorder,param.mBorder],'both');
    
    % end of synNet
    
    %% start of viewNet
    svSp=   SVSP(:,:,:,startInd:endInd);
    svSp= 	padarray(svSp,[param.mBorder,param.mBorder],'both');
    
    if (param.useGPU)
        svSp        = gpuArray(svSp);
    end
    
    for nv= 1:size(gtIm,3)
        [viewRes] = EvaluateSystem(param.viewNet,inAll,...
            gtIm(:,:,nv,:),synOut(:,:,nv,:),svSp(:,:,nv,:),false,false);
        % record current compensated view
        finalOut(:,:,nv,startInd:endInd)= viewRes(end).x;
    end
    
end

param.height= mRow;
param.width= mCol;
SVOut= myPatch2LF(finalOut+ repmat(INALL,1,1,size(GT,3),1));
SVOut= reshape(SVOut,size(SVOut,1),size(SVOut,2),nInput.angDim,nInput.angDim);   

AvgAll= mean(mean(SVGN,3),4);
for nr=1:nInput.angDim
    for nc= 1:nInput.angDim
     	viewPSNRCV(nr,nc)= psnr(im2double(AvgAll),SVG(:,:,nr,nc));
        viewPSNREW(nr,nc)= psnr(im2double(nInput.AvgEW(1:mRow,1:mCol,nr)),SVG(:,:,nr,nc));
        viewPSNRNS(nr,nc)= psnr(im2double(nInput.AvgEW(1:mRow,1:mCol,nc)),SVG(:,:,nr,nc));
        viewPSNR(nr,nc)=   psnr(im2double(SVOut(:,:,nr,nc)),SVG(:,:,nr,nc));
    end
end
% fprintf('Direct Avg mean: %.1f dB\t CNN: %.1f dB\n',mean(viewPSNRCV(:)),mean(viewPSNR(:)));
fprintf([IN.currFileName,' \tnoise %d: \tAvg: \t%.2f \tviewNet:\t%.2f\n'],noiseLevel,mean(viewPSNRCV(:)), mean(viewPSNR(:)));

% disp(viewPSNR);
% disp(viewPSNRCV);
% disp(viewPSNREW);
% disp(viewPSNRNS);

% without four coners
% sum(viewPSNR(:))/60- (viewPSNR(1,1)+ viewPSNR(8,1)+ viewPSNR(1,8)+viewPSNR(8,8))/60
outputFolder= [IN.fileFolder,'\Denoise\viewNet\'];
if(~isdir(outputFolder))
    mkdir(outputFolder);
end

% save([outputFolder,IN.currFileName,sprintf('_n%d_CNN.mat',noiseLevel)],'SVOut','viewPSNR','viewPSNRCV');

end

% imwrite(SVGN(:,:,4,4),'noisyCentral2.png','png')
% imwrite(SVOut(:,:,4,4),'denoisedCentral2.png','png')




