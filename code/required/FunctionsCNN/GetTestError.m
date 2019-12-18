function testError = GetTestError(errorFolder)

global param;

if (param.continue)
    fid = fopen([errorFolder, '/error.txt'], 'rt');
    prevError = cell2mat(textscan(fid, '%f\n'));
    fclose(fid);
    testError = prevError;
else
    fid = fopen([errorFolder, '/error.txt'], 'wt');
    fclose(fid);
    testError = [];
end