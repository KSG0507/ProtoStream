function [data, labels] = readUCILetters_2
    mainPath = strcat('.',filesep,'UCI_Letters',filesep);
    path = strcat(mainPath,'letter-recognition.data');
    A = importdata(path);
    data = A.data';
    labels = cell2mat(A.textdata);
end