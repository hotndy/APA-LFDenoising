function output = PadImg(input, pad)

% output = input(pad+1:end-pad, pad+1:end-pad, :, :, :, :, :, :);
output = padarray(input,[pad,pad],'symmetric','both');