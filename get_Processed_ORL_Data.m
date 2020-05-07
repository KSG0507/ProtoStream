function [Y,originalData] = get_Processed_ORL_Data(DatabasePath)
    %Y = zeros(112*92,400);
    Y = [];
    originalData = zeros(112*92,400);
    imgSets = imageSet(DatabasePath, 'recursive');
    imgNumber = 0;
    for i = 1:length(imgSets)
        srcFiles=dir(strcat(DatabasePath,imgSets(i).Description,'\*.pgm'));
        for j = 1:length(srcFiles)
             imageFileName = strcat(DatabasePath,imgSets(i).Description,'\',srcFiles(j).name);
             imageVal = imread(imageFileName);
             I = double(imageVal);
             HOGfeatures = extractHOGFeatures(imageVal);
             normalizedImage = (I-min(I(:)))/(max(I(:))-min(I(:)));
             imgNumber = imgNumber+1;
             originalData(:,imgNumber) = normalizedImage(:);
             %Y = horzcat(Y,double(HOGfeatures(:)));
             Y = horzcat(Y,normalizedImage(:));
        end
    end
end