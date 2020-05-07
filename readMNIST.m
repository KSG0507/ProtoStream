function [I,labels,I_test,labels_test] = readMNIST(num)
    mainPath = strcat('.',filesep,'MNIST',filesep);
    path = strcat(mainPath,'train-images.idx3-ubyte');
    fid = fopen(path,'r','b');  %big-endian
    magicNum = fread(fid,1,'int32');   
    if(magicNum~=2051) 
        display('Error: cant find magic number');
        return;
    end
    imgNum = fread(fid,1,'int32');  
    rowSz = fread(fid,1,'int32'); 
    colSz = fread(fid,1,'int32');   
    if(num<imgNum) 
        imgNum=num; 
    end
    I = zeros(rowSz*colSz,imgNum);
    for k = 1:imgNum
        A = double(uint8(fread(fid,[rowSz colSz],'uchar')));
        A = (A-min(A(:)))/(max(A(:))-min(A(:)));
        I(:,k) = A(:);
    end
    fclose(fid);
%%
    path = strcat(mainPath,'train-labels.idx1-ubyte');
    fid = fopen(path,'r','b');  % big-endian
    magicNum = fread(fid,1,'int32');
    if(magicNum~=2049) 
        display('Error: cant find magic number');
        return;
    end
    itmNum = fread(fid,1,'int32');

    if(num<itmNum) 
        itmNum=num; 
    end

    labels = double(uint8(fread(fid,itmNum,'uint8')));
    fclose(fid);
%%
    path = strcat(mainPath,'t10k-images.idx3-ubyte');
    fid = fopen(path,'r','b');  % big-endian
    magicNum = fread(fid,1,'int32');
    if(magicNum~=2051) 
        display('Error: cant find magic number');
        return;
    end
    imgNum = fread(fid,1,'int32');
    rowSz = fread(fid,1,'int32');
    colSz = fread(fid,1,'int32');
%     if(num<imgNum) 
%         imgNum=num; 
%     end
    I_test = zeros(rowSz*colSz,imgNum);
    for k = 1:imgNum
        A = double(uint8(fread(fid,[rowSz colSz],'uchar')));
        A = (A-min(A(:)))/(max(A(:))-min(A(:)));
        I_test(:,k) = A(:);
    end
    fclose(fid);
%%
    path = strcat(mainPath,'t10k-labels.idx1-ubyte');
    fid = fopen(path,'r','b');  % big-endian
    magicNum = fread(fid,1,'int32');
    if(magicNum~=2049) 
        display('Error: cant find magic number');
        return;
    end
    itmNum = fread(fid,1,'int32'); 
%     if(num<itmNum) 
%         itmNum=num; 
%     end
    labels_test = double(uint8(fread(fid,itmNum,'uint8')));
    fclose(fid);
end