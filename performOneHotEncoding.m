function [encodedData,uniqueLabels] = performOneHotEncoding(Z)
    d = size(Z,1);
    startPos = 0;
    encodedData = [];
    uniqueLabels = cell(d,1);
    numTrueUsages = 0;
    fprintf('Dimensionality of original features: %d\n',d);
    for feat = 1:d
        values  = Z(feat,:);
        labels = unique(values);
        uniqueLabels{feat} = labels;
        numLabels = length(labels);
        nSamples = length(values);
        if(numLabels <= 100 || (numLabels/nSamples) <=0.1)
            %% Perfrom one hot encoding
            codedMatrix = zeros(numLabels,nSamples);
            for i = 1:numLabels
                codedMatrix(i,values==labels(i)) = 1;
            end
            encodedData(startPos+1:startPos+numLabels,:) = codedMatrix;
            startPos = startPos+numLabels;
        else
            %% Use the true value as the encoded value
            numTrueUsages = numTrueUsages+1;
            fprintf('Using the true value for the %d time\n',numTrueUsages);
            encodedData(startPos+1,:) = (values-min(values))/(max(values)-min(values));
            startPos = startPos+1;
        end
    end
end