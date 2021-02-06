function [train, labels_train, test, labels_test] = readUSPS
    mainPath = strcat('.',filesep,'USPS',filesep);
    path = strcat(mainPath,'usps.h5');
    train = double(h5read(path,'/train/data'));
    labels_train = h5read(path,'/train/target');
    test = double(h5read(path,'/test/data'));
    labels_test = h5read(path,'/test/target');
end