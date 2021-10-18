function [BVP,PR_psd] = ippg_face_dynamic1(VidObj, Fs_video, StartTime, Duration,display)
% function [BVP,PR_psd] = ippg_face_dynamic1(VidObj, Fs_video, StartTime, Duration,display)
    addpath(genpath('tools'))
        %% Parameters
    LPF = 0.5; %low cutoff frequency (Hz) - 0.7 Hz in reference
    HPF = 2; %high cutoff frequency (Hz) - 4.0 Hz in reference
    f_bandpass_hr_findpeak    = [0.5 2];
    iPPG_time    = 0:1/Fs_video:(Duration-1/Fs_video);
    %% Load Video:

    FramesToRead       = ceil(Duration*Fs_video); %video may be encoded at slightly different frame rate
    RGB = zeros(FramesToRead,3);%initialize color signal
    FN  = 0;
    faceDetector = vision.CascadeObjectDetector();
    pointTracker = vision.PointTracker('MaxBidirectionalError', 2);
    numPts       = 0;


   %------------------------------------------------------------
%     n_movmean    = 6;
    n_movmean    = 6;
    dim          = 2;
   N_threshold =Duration*Fs_video;
   while hasFrame(VidObj) && FN< N_threshold

        FN        = FN+1;    
%         disp(FN);
        VidFrame      = readFrame(VidObj);   
        VidFrame      = double(VidFrame)/255;
        VidFrame_gray = rgb2gray(VidFrame);
       if numPts < 10
        % Detection mode
          bbox = faceDetector.step(VidFrame_gray);
        if ~isempty(bbox)
            % Find corner points inside the detected region.
            points = detectMinEigenFeatures(VidFrame_gray, 'ROI', bbox(1, :));
            
            % Re-initialize the point tracker.
            xyPoints    = points.Location;
            numPts      = size(xyPoints,1);release(pointTracker);initialize(pointTracker, xyPoints, VidFrame_gray);
            oldPoints   = xyPoints;     
            bboxPoints  = bbox2points(bbox(1, :));
            bboxPolygon = reshape(bboxPoints', 1, []);
            
            if (display==2)
                figure(1)
                subplot(1,2,1)
                
                % Display a bounding box around the detected face
                img_display = insertShape( VidFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
                
                % Display detected corners
                img_display = insertMarker(img_display, xyPoints, '+', 'Color', 'white');
                
                imshow(img_display)
            end
        end
        
    else
        % Tracking mode
        [xyPoints, isFound] = step(pointTracker,  VidFrame_gray);
        visiblePoints       = xyPoints(isFound, :);
        oldInliers          = oldPoints(isFound, :);
        
        numPts = size(visiblePoints, 1);
        if numPts >= 10
            % Estimate the geometric transformation between the old points and the new points.
            [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);
            
            % Apply the transformation to the bounding box.
            bboxPoints  = transformPointsForward(xform, bboxPoints);
            bboxPolygon = reshape(bboxPoints', 1, []);     
            oldPoints   = visiblePoints;setPoints(pointTracker, oldPoints);
            bbox        = round([bboxPolygon(1) bboxPolygon(2) bboxPolygon(3)-bboxPolygon(1) bboxPolygon(6)-bboxPolygon(2)]);
            
            if (display==2)
                figure(1)
                subplot(1,2,1)
                
                % Display a bounding box around the face being tracked
                img_display = insertShape( VidFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
                
                % Display tracked points
                img_display = insertMarker(img_display, visiblePoints, '+', 'Color', 'white');
                
                imshow(img_display)
            end
        end
    end
    img_roi = VidFrame(bbox(2):bbox(2)+bbox(4), bbox(1):bbox(1)+bbox(3), :);
                          
           %% SKIN DETECTION

        img_roi           = removeImpulseNoise(img_roi);
%         Lambda    = 100;
%         img_roi   = spdetrend(img_roi,Lambda); 
        img_roi_ycbcr     = rgb2ycbcr(img_roi);%img_roi_ycbcr = imgaussfilt(img_roi_ycbcr, 3);  % optional
        img_roi_thresh_Y  = img_roi_ycbcr(:,:,1) > 80/255;
        img_roi_thresh_Y = movmean(real(img_roi_thresh_Y),n_movmean,dim)+j*movmean(imag(img_roi_thresh_Y),n_movmean,dim);
        img_roi_thresh_Cb = img_roi_ycbcr(:,:,2) > 77/255 & img_roi_ycbcr(:,:,2) < 127/255;
        img_roi_thresh_Cb = movmean(real(img_roi_thresh_Cb),n_movmean,dim)+j*movmean(imag(img_roi_thresh_Cb),n_movmean,dim);
        img_roi_thresh_Cr = img_roi_ycbcr(:,:,3) > 133/255 & img_roi_ycbcr(:,:,3) < 173/255;
        img_roi_thresh_Cr = movmean(real(img_roi_thresh_Cr),n_movmean,dim)+j*movmean(imag(img_roi_thresh_Cr),n_movmean,dim);
        img_roi_skin      = img_roi_thresh_Y & img_roi_thresh_Cb & img_roi_thresh_Cr;   

        %% COLORSPACE CONVERSION
        img_roi_luv = rgb2xyz(img_roi);
        cform       = makecform('xyz2uvl');
        img_roi_luv = applycform(img_roi_luv, cform);
        img_roi_luv = movmean(real(img_roi_luv),n_movmean,dim)+j*movmean(imag(img_roi_luv),n_movmean,dim);
        
        if (display==2)
            subplot(1,2,2)
            imshow(img_roi_skin)
        end

            %% IMAGE -> SIGNAL
             BVP_I(FN) = sum(sum(img_roi_luv(:,:,1).*img_roi_skin))/sum(sum(img_roi_skin));
             
   end

    NyquistF  = 1/2*Fs_video;
    [B,A] = butter(3,[LPF/NyquistF HPF/NyquistF]);%Butterworth 3rd order filter
    BVP_I(isnan(BVP_I))=nanmean(BVP_I);%BVP_I(isnan(BVP_I))=0;

    BVP_F = filtfilt(B,A,double(BVP_I));
    iPPG_signal  = BVP_F.';

    iPPG_signal  = gradient(iPPG_signal);
    %Step1  Baseline cancelation 去基线去趋势操作
    iPPG_signal= detrend(iPPG_signal,'constant');
    %Step2 Spike smoothing 尖峰平滑
    M=Fs_video;
    iPPG_signal=spike_smooth(iPPG_signal,M);
    %Step3  Five-point cubic smoothing 五三平滑
    iPPG_signal=mean5_3(iPPG_signal,M);
    % cwt filtering
    wavelet_type      = 'amor';
    wt                = cwt(iPPG_signal, wavelet_type, Fs_video, 'VoicesPerOctave', 32, 'FrequencyLimits', f_bandpass_hr_findpeak);
    wt_energy         = sum(abs(wt'));
    wt_filt           = wt .* repmat(wt_energy, size(wt, 2), 1)';
    BVP = icwt(wt_filt, 'amor');%使用解析小波对cwt求逆
%     BVP = removeImpulseNoise(BVP);
%%PSD
    PR_psd    = prpsd(BVP,Fs_video,40,240,true);



end%end function
