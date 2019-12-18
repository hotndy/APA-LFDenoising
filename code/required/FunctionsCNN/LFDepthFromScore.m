function [disparity, Ce]= LFDepthFromScore(SV,dp,mode)

SV= double(SV);
% W= double(W);

if(ndims(SV)==5)
    SVt= reshape(SV,size(SV,1),size(SV,2),size(SV,3),size(SV,4)*size(SV,5));
    clear SV;
    for v=1:size(SVt,4);
        SV(:,:,v)= 255*rgb2gray(SVt(:,:,:,v)/255);
    end
    clear SVt;
end

viewDimWid= size(SV,1);
viewDimLen= size(SV,2);
viewTotNum= size(SV,3);
angDim= sqrt(viewTotNum);

IndxArray=reshape(1:angDim^2, angDim, angDim);
ctrRC= round((angDim+1)/2);
activeIdx= IndxArray(ctrRC-2:ctrRC+2,ctrRC-2:ctrRC+2);

% balancing vignetting
ctrIdx= IndxArray(round((angDim+1)/2),round((angDim+1)/2));
IRt= SV./repmat(SV(:,:,ctrIdx),1,1,angDim^2);
IR= median(reshape(IRt,[],angDim^2),1);
IRt= max(reshape(SV,[],angDim^2));
IR= IRt/IRt(ctrIdx);
% normalize
%{
IR(find(IR>0.9))=1;
%}
% figure;imshow(reshape(IR==1,15,15));
SVt= SV./permute(repmat(IR',1,viewDimWid,viewDimLen),[2,3,1]);
SVt= SVt(:,:,activeIdx(:));

viewTotNum= size(SVt,3);
angDim= sqrt(viewTotNum);
IndxArray=reshape(1:angDim^2, angDim, angDim);
% caculate disparity using SX slices, i.e.,using horizontal slices
fprintf('Calculating LF disparity.');
if(strcmp(mode,'horizontal')| strcmp(mode,'both')) 
    for y=1:viewDimWid
        TN= (angDim+1)/2; 
        epiSX= permute(SVt(y,:,IndxArray(TN, :)),[3,2,1]);

        [disparitySX(y,:), CeSX(y,:)]= disparityVoting(epiSX,dp);
        %imshow(epiSX(:,:,:,t)/255);
        if((double(y)/50)==floor(double(y)/50))
            % str= sprintf('%.2f%% completed', 100*y/viewDimWid);
            % disp(str);
            fprintf('.');
        end
    end
    disparity= disparitySX;
    Ce= CeSX;
end

% caculate disparity using TY slices, i.e.,using vertical slices
if(strcmp(mode,'vertical')| strcmp(mode,'both'))
    for x=1:viewDimLen
        TN= (angDim+1)/2; 
        epiTY= permute(SVt(:,x,IndxArray(:, TN)),[3,1,2]);

        [disparityTY(:,x), CeTY(:,x)]= disparityVoting(epiTY,dp);
        %imshow(epiSX(:,:,:,t)/255);
        if((double(x)/50)==floor(double(x)/50))
            % str= sprintf('%.2f%% completed', 100*x/viewDimLen);          
            % disp(str);  
             fprintf('.');
        end
    end
    disparity= disparitySX;
    Ce= CeSX;    
end
            
if(strcmp(mode,'both'))
    disparity= disparitySX.*(CeSX>=CeTY)+ disparityTY.*(CeSX<CeTY);
    Ce= CeSX.*(CeSX>=CeTY)+ CeTY.*(CeSX<CeTY);
end

fprintf('\n');

end

function [disparity, Ce]= disparityVoting(E,dp)

viewNum= size(E,1);
pixNum= size(E,2);

% Calculate confidence window
WS= 9; % confidence window size
Sm= (1+viewNum)/2;
Ex= padarray(E, [0 (WS-1)/2],'symmetric','both');
Ce= zeros(1,pixNum);

for u=1:pixNum
    X= repmat(E(Sm,u),1,WS);
    X1= Ex(Sm,u:(u+WS-1));
    Ce(u)= sqrt(sum((X-X1).*(X-X1)))/255; % neighbourhood variation
end

% calculate SX slicings
% dp= -3:0.1:3; % disparity range
R= zeros(viewNum,pixNum,length(dp));

% form the set
uVec=1:pixNum;
for d=1:length(dp) 
    for s=1:viewNum
        ut= uVec+(Sm-s)*dp(d);
        iu= find(floor(ut)>0 & ceil(ut)<=pixNum);
        ou= find(floor(ut)<=0 | ceil(ut)>pixNum);
        EH= E(s,ceil(ut(iu)));
        EL= E(s,floor(ut(iu)));
        R(s,iu,d)= EL+ (EH- EL).*(ut(iu)- floor(ut(iu)));
        R(s,ou,d)= -1;
    end

    indx= (R(:,:,d)>=0);

    rm= repmat(E(Sm,:),viewNum,1);   

    Kr= (1-(R(:,:,d)-rm).^2/(255^2)).*indx; % fault found Kr= (1-(Rt-rm).^2.*indx/(Hbp^2));
    Kr(find(Kr<0))=0;

    S(:,d)= sum(Kr)./(sum(indx,1)); %average score RGB
    
end

% calculate combined confidence
[Smax, I] = max(S');
Smean= mean(S');
disparity= dp(I);
% Cd= Ce.*abs(Smax-Smean);
end