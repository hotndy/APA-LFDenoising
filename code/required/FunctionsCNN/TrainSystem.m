function TrainSystem(viewNet, synNet)

global param;

count = 0;

% load validation data from mat
global IN;
IN.listTraining = dir([param.trainingData,'\*n',sprintf('%d.mat',param.noiseLevel)]);
IN.numImages= length(IN.listTraining);
IN.currImageBatches= 0;
param.currEpoch= param.startEpoch;
% validation data assignment
IN.listTesting = dir([param.testData,'\*n',sprintf('%d.mat',param.noiseLevel)]);

IN.testSetIdx= [1:length(IN.listTesting)];
IN.trainSetIdx= [1:length(IN.listTraining)];

% load validation data
count=0;
IN.testData.INALL= [];
IN.testData.INEW= []; 
IN.testData.INNS= [];
IN.testData.GT= [];

startInd= 1;
for nd= 1:length(IN.testSetIdx)
    fprintf(repmat('\b', [1, count]));
    count= fprintf('loading validation data %d/%d\n', nd,length(IN.testSetIdx));
    ni= IN.testSetIdx(nd);	
    load([param.testData,IN.listTesting(ni).name]);
    
    for nr=1:8
        for nc=1:8
        % [spPSNR(nr,nc),SVSP(:,:,nr,nc)]= BM3D(SVG(:,:,nr,nc), SVGN(:,:,nr,nc), noiseLevel, 'np', 0);
            spFilt(:,:,nr,nc) = imgaussfilt(nInput.SVGN(:,:,nr,nc),param.noiseLevel/10);
        end
    end
    
    SYNOUT= stepSynNet(nInput,synNet); 
	% SYNUT= GetPatches(nInput.synOut, param.patchSize, param.stride);
    
    INALL=  GetPatches(nInput.AvgAll, param.patchSize, param.stride);
	GT	 =  GetPatches(nInput.gtLFIM, param.patchSize, param.stride);    
    SVSP	=  GetPatches(spFilt, param.patchSize, param.stride); 
   
    endInd= startInd+ size(SYNOUT,4)-1;
    IN.testData.INALL(:,:,:,startInd:endInd)= im2single(INALL);
    IN.testData.GT(:,:,:,startInd:endInd)= im2single(GT);
    IN.testData.SVSP(:,:,:,startInd:endInd)= im2single(SVSP);
    IN.testData.SYNOUT(:,:,:,startInd:endInd)= im2single(SYNOUT);
    startInd= endInd+1;
    clear INALL GT SYNOUT SVSP spFilt;
end
fprintf(repmat('\b', [1, count]));
fprintf('Validation data loaded.\n');
count=0;

testError=IN.error;
totalTrainedBatch= IN.totalTrainedBatch;

% load([param.trainingData,'training.mat']);
% maxNumPatches = size(INCONF,4);
% numImages = floor(maxNumPatches / param.batchSize) * param.batchSize;

startFlag= 1;

while (1)    % epoch loop
 

    % set a different permutation of data index for each epoch
    IN.trainSetIdx= randi(length(IN.listTraining),[1,length(IN.listTraining)]);
    startData=1; % always start from the first data of current epoch
        
    for nd = startData:length(IN.trainSetIdx)
                
        ni= IN.trainSetIdx(nd);
        
        % train using current sequqnce
        load([param.trainingData,IN.listTraining(ni).name]);
        
        % prepare data for the current image 
        INALL=[]; GT	=[];
        SVSP= []; SYNOUT= []; spFilt= [];
        
        for nr=1:8
            for nc=1:8
            % [spPSNR(nr,nc),SVSP(:,:,nr,nc)]= BM3D(SVG(:,:,nr,nc), SVGN(:,:,nr,nc), noiseLevel, 'np', 0);
                spFilt(:,:,nr,nc) = imgaussfilt(nInput.SVGN(:,:,nr,nc),param.noiseLevel/10);
            end
        end
        
        
        SYNOUT= im2single(stepSynNet(nInput,synNet)); 
        % SYNOUT= im2single(GetPatches(nInput.synOut, param.patchSize, param.stride));       
    
    	INALL=  im2single(GetPatches(nInput.AvgAll, param.patchSize, param.stride));
        GT	 =  im2single(GetPatches(nInput.gtLFIM, param.patchSize, param.stride));
        SVSP =  im2single(GetPatches(spFilt, param.patchSize, param.stride)); 
        
        % prepare data for the current image                
        maxNumPatches = size(GT,4);
        IN.currImageBatches = floor(maxNumPatches / param.batchSize);
        
        if(startFlag==1)
            startBatch= param.startBatch;
        else
            startBatch=1;
        end
        
        for it=startBatch:IN.currImageBatches
            
            startFlag= 0;

            totalTrainedBatch= totalTrainedBatch+1;
            
            if (mod(it, param.printInfoIter) == 0)
                fprintf(repmat('\b', [1, count]));
                count = fprintf('Current batch %d (total %d) / data (%d/%d) epoch %d\n', it, totalTrainedBatch, ni, length(IN.trainSetIdx),param.currEpoch);
            end

          %% main optimization
            % startInd= mod((it-1) * param.batchSize, IN.currImageBatches) + 1;
            startInd= (it-1) * param.batchSize + 1;
            inAll=          INALL(:,:,:,startInd:startInd+ param.batchSize-1);
            gtIm=           GT(:,:,:,startInd:startInd+ param.batchSize-1);
            synOut=         SYNOUT(:,:,:,startInd:startInd+ param.batchSize-1);
            svSp=           SVSP(:,:,:,startInd:startInd+ param.batchSize-1);

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

            param.viewTrainNum= 5;
            viewIdx= randi(64,param.viewTrainNum,1);
            for nnv= 1:param.viewTrainNum
                nv= viewIdx(nnv);
                [viewRes] = EvaluateSystem(viewNet,inAll,...
                    gtIm(:,:,nv,:),synOut(:,:,nv,:),svSp(:,:,nv,:),true,false);
                % network update for each view
                viewNet = UpdateNet(viewNet, viewRes, ceil(param.currEpoch/30));
            end
            
            % if (mod(it, param.testNetIter) == 0)
            if (mod(totalTrainedBatch, param.testNetIter) == 0)
                % 1. perform validation
                % countTest = fprintf('\nStarting the validation process / test data %d\n', IN.testSetIdx);
                countTest = fprintf('\n Validating trained network\n');

                curError = TestDuringTraining(viewNet);
                testError = [testError; curError];
                plot(1:length(testError), testError);
                title(sprintf('ViewNet Current RMSE: %f', curError));
                drawnow;

                fprintf(repmat('\b', [1, countTest]));                
                % 2. save network
                [~, curNetName, ~] = GetFolderContent(param.trainNet, '.mat');
                fileName = sprintf('/Net-e%04d-d%04d-b%06d.mat', param.currEpoch, ni,it);
                save([param.trainNet, fileName], 'viewNet','testError','totalTrainedBatch');

                if (~isempty(curNetName))
                    curNetName = curNetName{1};
                    delete(curNetName);
                end
            end 
            
        end
    %     it = it + 1;   
    end
    
    param.currEpoch= param.currEpoch +1;
    
end
