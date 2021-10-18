function [BVP_F,BVP, PR] = ippg_face(VidObj, Fs_video, StartTime, Duration,flag_display)
    f_scheme=2;
    addpath(genpath('tools'))
        %% Parameters
    LPF = 0.7; %low cutoff frequency (Hz) - 0.7 Hz in reference
    HPF = 2.5; %high cutoff frequency (Hz) - 4.0 Hz in reference

    %% Load Video:

    FramesToRead       = ceil(Duration*Fs_video); %video may be encoded at slightly different frame rate
    RGB = zeros(FramesToRead,3);%initialize color signal
    FN  = 0;
    bbox        = [330 200 200 180];
    readFrame(VidObj);
    VidObj.CurrentTime
   
   %------------------------------------------------------------
   t_list = [];
   while hasFrame(VidObj)
       VidFrame      = readFrame(VidObj);
       t_list        = [t_list,VidObj.CurrentTime];     
   end
   tt  = t_list-StartTime;
   ind = find(tt>0);
   VidObj.CurrentTime=t_list(ind(1))
   %------------------------------------------------------------
   

   N_threshold =Duration*Fs_video;
   while hasFrame(VidObj) && FN< N_threshold

        FN        = FN+1;    
        disp(FN);

        VidFrame      = readFrame(VidObj);
        VidFrame_gray = rgb2gray(VidFrame);
        points      = detectMinEigenFeatures(VidFrame_gray, 'ROI', bbox);
        % Re-initialize the point tracker.初始化
        xyPoints    = points.Location;
%        numPts      = size(xyPoints,1);release(pointTracker);initialize(pointTracker, xyPoints, VidFrame_gray);
        oldPoints   = xyPoints;    %保存点的副本
        bboxPoints  = bbox2points(bbox);%

        bboxPolygon = reshape(bboxPoints', 1, []);%将框角转换为插入形状所需的[x1y1x2y2x3y3x4y4]格式。

        if (flag_display==2)
            figure(3)
            subplot(1,2,1)

            % Display a bounding box around the detected face显示边界框
            VidFrame_flag_display = insertShape(VidFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);

            % Display detected corners显示检测到的点
            VidFrame_flag_display = insertMarker(VidFrame_flag_display, xyPoints, '+', 'Color', 'white');

            imshow(VidFrame_flag_display)
        end
        VidROI    = VidFrame_gray(bbox(2):bbox(2)+bbox(4), bbox(1):bbox(1)+bbox(3), :);
        
        if f_scheme~=1                    
           %% SKIN DETECTION
            img               = double(VidFrame)/255;
            img_roi           = img(bbox(2):bbox(2)+bbox(4), bbox(1):bbox(1)+bbox(3), :);
            img_roi_ycbcr     = rgb2ycbcr(img_roi);%img_roi_ycbcr = imgaussfilt(img_roi_ycbcr, 3);  % optional

            img_roi_thresh_Y  = img_roi_ycbcr(:,:,1) > 80/255;
            img_roi_thresh_Cb = img_roi_ycbcr(:,:,2) > 77/255 & img_roi_ycbcr(:,:,2) < 127/255;
            img_roi_thresh_Cr = img_roi_ycbcr(:,:,3) > 133/255 & img_roi_ycbcr(:,:,3) < 173/255;    
            img_roi_skin      = img_roi_thresh_Y & img_roi_thresh_Cb & img_roi_thresh_Cr;


            %% COLORSPACE CONVERSION
            img_roi_luv = rgb2xyz(img_roi);
            cform       = makecform('xyz2uvl');
            img_roi_luv = applycform(img_roi_luv, cform);

            if (flag_display==2)
                subplot(1,2,2)
                imshow(img_roi_skin)
            end

            %% IMAGE -> SIGNAL
             BVP_I(FN) = sum(sum(img_roi_luv(:,:,1).*img_roi_skin))/sum(sum(img_roi_skin));
        else        
            RGB(FN,:) = sum(sum(VidROI));    
        end
   end

    %% Filter, Normalize
     NyquistF  = 1/2*Fs_video;
    [B,A] = butter(3,[LPF/NyquistF HPF/NyquistF]);%Butterworth 3rd order filter
    BVP_I(isnan(BVP_I))=nanmean(BVP_I);%BVP_I(isnan(BVP_I))=0;

    BVP_F = filtfilt(B,A,double(BVP_I));
    iPPG_signal  = BVP_F.';
    
    % cwt filtering
    wavelet_type      = 'amor';
    wt                = cwt(iPPG_signal, wavelet_type, Fs_video, 'VoicesPerOctave', 32, 'FrequencyLimits', [0.8 2.5]);
    wt_energy         = sum(abs(wt'));
    wt_filt           = wt .* repmat(wt_energy, size(wt, 2), 1)';
    BVP = icwt(wt_filt, 'amor');%使用解析小波对cwt求逆
    PR    = prpsd(BVP,Fs_video,40,240,true);


end%end function
