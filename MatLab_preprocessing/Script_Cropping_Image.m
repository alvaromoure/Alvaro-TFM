sDirOrigi = '/COVIDNet/Data/OrgImages/';
sDirSegmented = '/COVIDNet/Data/OrgImagesSegmented/';
sDirOut1 = '/COVIDNet/Data/CroppedImages/';
sDirOut2 = '/COVIDNet/Data/CroppedSegmentedImages/';
if ~exist(sDirOut1,'dir')
    mkdir(sDirOut1);
end
if ~exist(sDirOut2,'dir')
    mkdir(sDirOut2);
end

eImages = dir(sDirOrigi);%There are different data formats
iNImages = length(eImages);



for i = 3:iNImages
    sFileOrigen = strcat(sDirOrigi,eImages(i).name);
    mImageOrg=imread( sFileOrigen );
    [h,w,c] = size(mImageOrg);
    if c>1
        mImageOrg = rgb2gray(mImageOrg);
    end
    info = imfinfo( sFileOrigen );
    
    if info.BitDepth > 16
        disp( 'Fichero con bitdepth > 16 bits ' );
    elseif  info.BitDepth < 16
        disp( 'Fichero con bitdepth < 16 bits ' );
    end
    mImageOrg = im2uint16(mImageOrg);
    
    %---------------------------------------------------------
    %---------- Load segmented image ------------------
    try
        sFileSegmented = strcat(sDirSegmented,eImages(i).name);
        mImageSeg=imread( sFileSegmented );
        mImageSeg = imresize(mImageSeg,[h, w]);
        [~,~,cs] = size(mImageSeg);
        if cs > 1
            mImageSeg = rgb2gray(mImageSeg);
        end
        mImageSeg = im2uint16(mImageSeg);
        %----------- Apply mask to the original images --------
        se = offsetstrel('ball',5,5);
        mErodedImageSeg = imerode(mImageSeg,se);%erosion
        mErodedImageSeg(mErodedImageSeg~=0)=1;
        mImageNew2 = mImageOrg.*mErodedImageSeg; %CroppedSegmented
        mImageNew1 = mImageOrg;%cropped
        %----------- Boundaries ----------------------------
        vRows = sum(mErodedImageSeg)/h;
        vColumns = sum(mErodedImageSeg,2)/w;
        vIndW = find(vRows>0);
        vIndH = find(vColumns>0);
        %----------- Build cropped and cropped-segmented images -----
        mImageNew1 = mImageNew1(vIndH(1):vIndH(end),vIndW(1):vIndW(end));
        mImageNew1 = uint16(65536 * mat2gray( mImageNew1 ));
        
        mImageNew2 = mImageNew2(vIndH(1):vIndH(end),vIndW(1):vIndW(end));
        mImageNew2 = uint16(65536 * mat2gray( mImageNew2 ));
        
        [~,sName,~]=fileparts(eImages(i).name);
        
        sFilePNGDestino1 = strcat(sDirOut1,sName,'.png');
        imwrite(mImageNew1, sFilePNGDestino1, 'png' );
        
        sFilePNGDestino2 = strcat(sDirOut2,sName,'.png');
        imwrite(mImageNew2, sFilePNGDestino2, 'png' );
    catch
        error('Not processed image');
    end
end
