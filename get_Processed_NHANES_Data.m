function [encodedData,originalData] = get_Processed_NHANES_Data(fileName)
    oneHotEncode = true;
    rawData = xptread(fileName);
    originalData = table2array(rawData)';
    %The first row is just sequence numbers. Remove them
    originalData(1,:) = [];
    %Replace all nan's with zeros
    originalData(isnan(originalData)) = 31864; %some random number
    if(oneHotEncode)
        [encodedData,~] = performOneHotEncoding(originalData);
    else
        encodedData = originalData;
    end
end