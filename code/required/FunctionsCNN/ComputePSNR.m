function errEst = ComputePSNR(input, ref)

input = im2single(input);
ref = im2single(ref);


numPixels = numel(input);
rmsqrdErr = sqrt(sum((input(:) - ref(:)).^2)/ numPixels);
% errEst = 10 * log10(1/sqrdErr);

errEst= rmsqrdErr;