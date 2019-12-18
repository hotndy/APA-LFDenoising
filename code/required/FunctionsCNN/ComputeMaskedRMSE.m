function errEst = ComputePSNR(input, ref, mask)

input = im2single(input);
ref = im2single(ref);

validateIdx= find(mask);

numPixels = length(validateIdx);
rmsqrdErr = sqrt(sum((input(validateIdx) - ref(validateIdx)).^2)/ numPixels);
% errEst = 10 * log10(1/sqrdErr);

errEst= rmsqrdErr;