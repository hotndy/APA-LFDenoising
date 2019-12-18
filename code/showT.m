function showT(T)

nd= ndims(T);

if(nd==4) % color
    [row,col,cc,num]=size(T);
    T= permute(T,[1,2,4,3]);
    T= reshape(T,row,col*num,cc);
    figure;imshow(T);
elseif(nd==3) % gray
    [row,col,num]=size(T);
    T= reshape(T,row,col*num);
    figure;imshow(T);
    
end

end