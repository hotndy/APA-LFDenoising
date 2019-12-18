function showb(in)

in= gather(in);
[row,col,nc,batchSize]= size(in);
% if(nc~=1 & nc~= 3)
%     disp('wrong input');
%     return;
% end

shownIm= zeros(row*nc,col*batchSize);

for ni=1:nc
    shownIm((ni-1)*row+1:ni*row,:)= reshape(in(:,:,ni,:),row,col*batchSize);
end

figure;imshow(shownIm);


end