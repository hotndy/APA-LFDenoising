function TrainSystem(dispNet)

global param;

% testError = GetTestError(param.trainNet);
count = 0;
% it = param.startIter + 1;

% load validation data from mat
global IN;
IN.listTraining = dir([param.trainingData,'\*n20.mat']);
IN.numImages= length(IN.listTraining);
IN.currImageBatches= 0;
currEpoch= param.startEpoch;
% validation data assignment
IN.listTesting = dir([param.testData,'\*n20.mat']);

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
    ni= IN.trainSetIdx(nd);	
    load([param.testData,IN.listTesting(ni).name]);
    
    for nn=1:nInput.angDim
        gfiltEW(:,:,nn)= guidedfilter(nInput.AvgEW(:,:,nn), nInput.AvgAll, 9, 10^-6);
        gfiltNS(:,:,nn)= guidedfilter(nInput.AvgNS(:,:,nn), nInput.AvgAll, 9, 10^-6);
    end
    
    INALL=  GetPatches(nInput.AvgAll, param.patchSize, param.stride);
    INEW =  GetPatches(gfiltEW, param.patchSize, param.stride);
    INNS =  GetPatches(gfiltNS, param.patchSize, param.stride);
	GT	 =  GetPatches(nInput.gtLFIM, param.patchSize, param.stride);
      
    endInd= startInd+ size(GT,4)-1;
    IN.testData.INALL(:,:,:,startInd:endInd)= im2single(INALL);
    IN.testData.INEW(:,:,:,startInd:endInd)= im2single(INEW);
    IN.testData.INNS(:,:,:,startInd:endInd)= im2single(INNS);
    IN.testData.GT(:,:,:,startInd:endInd)= im2single(GT);
    startInd= endInd+1;
    clear INALL INEW INNS GT gfiltEW gfiltNS;
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
    % for ni=1:IN.numImages
%     if(startFlag==1)
%         startData= find(param.startData == IN.trainSetIdx);
%         if(isempty(startData)) % new added data
%             startData=1;
%         end
%     else
%         startData=1;
%     end  

    % set a different permutation of data index for each epoch
    IN.trainSetIdx= randi(length(IN.listTraining),[1,length(IN.listTraining)]);
    startData=1; % always start from the first data of current epoch
        
    for nd = startData:length(IN.trainSetIdx)
                
        ni= IN.trainSetIdx(nd);
        
        % train using current sequqnce
        load([param.trainingData,IN.listTraining(ni).name]);
        
        % prepare data for the current image 
        INALL=[]; INEW =[]; INNS =[];
        INNW =[]; INNE =[]; GT	=[];
        gfiltEW=[];  gfiltNS=[];
        
        for nn=1:nInput.angDim
            gfiltEW(:,:,nn)= guidedfilter(nInput.AvgEW(:,:,nn), nInput.AvgAll, 9, 10^-6);
            gfiltNS(:,:,nn)= guidedfilter(nInput.AvgNS(:,:,nn), nInput.AvgAll, 9, 10^-6);
        end
    
    	INALL=  im2single(GetPatches(nInput.AvgAll, param.patchSize, param.stride));
        INEW =  im2single(GetPatches(gfiltEW, param.patchSize, param.stride));
        INNS =  im2single(GetPatches(gfiltNS, param.patchSize, param.stride));
        GT	 =  im2single(GetPatches(nInput.gtLFIM, param.patchSize, param.stride));
            
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
                count = fprintf('Current batch %d (total %d) / data (%d/%d) epoch %d\n', it, totalTrainedBatch, ni, length(IN.trainSetIdx),currEpoch);
            end

          %% main optimization
            startInd= mod((it-1) * param.batchSize, IN.currImageBatches) + 1;
            inAll=          INALL(:,:,:,startInd:startInd+ param.batchSize-1);
            inEW=           INEW(:,:,:,startInd:startInd+ param.batchSize-1);
            inNS=           INNS(:,:,:,startInd:startInd+ param.batchSize-1);
            gtIm=           GT(:,:,:,startInd:startInd+ param.batchSize-1);

            inAll=          padarray(inAll,[param.mBorder,param.mBorder],'both');
            inEW=           padarray(inEW,[param.mBorder,param.mBorder],'both');
            inNS=           padarray(inNS,[param.mBorder,param.mBorder],'both');
            gtIm=           padarray(gtIm,[param.mBorder,param.mBorder],'both');
                   	            
            % figure;imshow([showLFIM(squeeze(inT1SpMask(:,:,:,1))),squeeze(inRainMask(:,:,:,1)),squeeze(gtImSpMask(:,:,:,1))])
            
            if (param.useGPU)
                inAll       = gpuArray(inAll);
                inEW        = gpuArray(inEW);
                inNS        = gpuArray(inNS);            
                gtIm        = gpuArray(gtIm);
            end

            [dispRes] = EvaluateSystem(dispNet,inAll,inEW,inNS,gtIm,true,false);
            %{
                tmpOutput= dispRes(end).x+ repmat(CropImg(inAll,param.mBorder),1,1,size(gtIm,3),1);
                pNo= 35;
                figure;imshow(tmpOutput(:,:,1,pNo));
                figure;imshow(tmpOutput(:,:,8,pNo));
            %}
            
            % dispNet = UpdateNet(dispNet, dispRes, it);
            % dispNet = UpdateNet(dispNet, dispRes, totalTrainedBatch);
            dispNet = UpdateNet(dispNet, dispRes, ceil(currEpoch/30));

            % if (mod(it, param.testNetIter) == 0)
            if (mod(totalTrainedBatch, param.testNetIter) == 0)
                % 1. perform validation
                % countTest = fprintf('\nStarting the validation process / test data %d\n', IN.testSetIdx);
                countTest = fprintf('\n Validating trained network\n');

                curError = TestDuringTraining(dispNet);
                testError = [testError; curError];
                plot(1:length(testError), testError);
                title(sprintf('SPAC Shuffle Avg T0T1 Current RMSE: %f', curError));
                drawnow;

                fprintf(repmat('\b', [1, countTest]));                
                % 2. save network
                [~, curNetName, ~] = GetFolderContent(param.trainNet, '.mat');
                fileName = sprintf('/Net-e%04d-d%04d-b%06d.mat', currEpoch, ni,it);
                save([param.trainNet, fileName], 'dispNet','testError','totalTrainedBatch');

                if (~isempty(curNetName))
                    curNetName = curNetName{1};
                    delete(curNetName);
                end
            end 
            
        end
    %     it = it + 1;   
    end
    
    currEpoch= currEpoch +1;
    
end
