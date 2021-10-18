function [iPPG_time30, BVP, PR] = ippg_dynamic_tracking(f_VideoFile,Fs_video,StartTime,Duration,display)

% INPUTS: 
%	file: source folder path (.png images) or video path/filename.
%   display: 0 = no display, 1 = display signals only, 2 = display signals and face tracking.
%
% OUTPUTS:
%   iPPG_time30, iPPG_signal30filt: iPPG signal and time vectors (u* channel filtered using its CWT representation).
%   iPR_time, iPR: instantaneous (beat-to-beat) pulse rate.
%   iBR_time, iBR: instantaneous (beat-to-beat) breathing rate.
%
% Reference: Frederic Bousefsaf, Alain Pruski, Choubeila Maaoui, Continuous wavelet filtering on webcam photoplethysmographic signals to remotely assess the instantaneous heart rate, Biomedical Signal Processing and Control, vol. 8, n? 6, pp. 568?574 (2013)


%% PREPARE IMAGES / VIDEO LOADING
    LPF = 0.8; %low cutoff frequency (Hz) - 0.7 Hz in reference
    HPF = 2.5; %high cutoff frequency (Hz) - 4.0 Hz in reference

VidObj       = VideoReader(f_VideoFile);
% length_vid   = floor(VidObj.FrameRate*VidObj.Duration);   %30fps
start_vid    = floor(StartTime*VidObj.FrameRate);
length_vid   = floor(VidObj.FrameRate*Duration);
end_vid      = start_vid+length_vid;
iPPG_time    = 0:1/VidObj.FrameRate:(Duration-1/VidObj.FrameRate);
iPPG_signal  = zeros(1,length_vid);

%% INIT FACE DETECTION AND TRACKING
faceDetector = vision.CascadeObjectDetector();
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);
numPts       = 0;

vidObj       = VideoReader(f_VideoFile);
% for i = 1:length_vid
for i =start_vid :end_vid
    disp(i);
    %% LOAD IMAGES
    img      = readFrame(VidObj);   
    img      = double(img)/255;
    img_gray = rgb2gray(img);
    
    %% FACE DETECTION AND TRACKING
    if numPts < 10
        % Detection mode
        bbox = faceDetector.step(img_gray);
        if ~isempty(bbox)
            % Find corner points inside the detected region.
            points = detectMinEigenFeatures(img_gray, 'ROI', bbox(1, :));
            
            % Re-initialize the point tracker.
            xyPoints    = points.Location;
            numPts      = size(xyPoints,1);release(pointTracker);initialize(pointTracker, xyPoints, img_gray);
            oldPoints   = xyPoints;     
            bboxPoints  = bbox2points(bbox(1, :));
            bboxPolygon = reshape(bboxPoints', 1, []);
            
            if (display==2)
                figure(1)
                subplot(1,2,1)
                
                % Display a bounding box around the detected face
                img_display = insertShape(img, 'Polygon', bboxPolygon, 'LineWidth', 3);
                
                % Display detected corners
                img_display = insertMarker(img_display, xyPoints, '+', 'Color', 'white');
                
                imshow(img_display)
            end
        end
        
    else
        % Tracking mode
        [xyPoints, isFound] = step(pointTracker, img_gray);
        visiblePoints       = xyPoints(isFound, :);
        oldInliers          = oldPoints(isFound, :);
        
        numPts = size(visiblePoints, 1);
        if numPts >= 10
            % Estimate the geometric transformation between the old points and the new points.
            [xform, ~, visiblePoints] = estimateGeometricTransform(...
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
                img_display = insertShape(img, 'Polygon', bboxPolygon, 'LineWidth', 3);
                
                % Display tracked points
                img_display = insertMarker(img_display, visiblePoints, '+', 'Color', 'white');
                
                imshow(img_display)
            end
        end
    end
    img_roi = img(bbox(2):bbox(2)+bbox(4), bbox(1):bbox(1)+bbox(3), :);
    %% SKIN DETECTION
    img_roi_ycbcr     = rgb2ycbcr(img_roi);
    %img_roi_ycbcr = imgaussfilt(img_roi_ycbcr, 3);  % optional
    
    img_roi_thresh_Y  = img_roi_ycbcr(:,:,1) > 80/255;
    img_roi_thresh_Cb = img_roi_ycbcr(:,:,2) > 77/255 & img_roi_ycbcr(:,:,2) < 127/255;
    img_roi_thresh_Cr = img_roi_ycbcr(:,:,3) > 133/255 & img_roi_ycbcr(:,:,3) < 173/255;    
    img_roi_skin      = img_roi_thresh_Y & img_roi_thresh_Cb & img_roi_thresh_Cr;
    %% COLORSPACE CONVERSION
    img_roi_luv = rgb2xyz(img_roi);
    cform       = makecform('xyz2uvl');
    img_roi_luv = applycform(img_roi_luv, cform);
    
    if (display==2)
        subplot(1,2,2)
        imshow(img_roi_skin)
    end
    %% IMAGE -> SIGNAL
    iPPG_signal(i) = sum(sum(img_roi_luv(:,:,1).*img_roi_skin))/sum(sum(img_roi_skin));

%% FILTERING USING CWT
if Fs_video<30%if Fs_video<30 Hz, resample to 30 Hz
    iPPG_time30   = iPPG_time(1):1/30:iPPG_time(end);
    iPPG_signal30 = interp1(iPPG_time, iPPG_signal, iPPG_time30, 'pchip'); %线性插值
else
    iPPG_time30   = iPPG_time;
    iPPG_signal30 = iPPG_signal;    
end

    NyquistF  = 1/2*Fs_video;
    
    
    
    [B,A] = butter(3,[LPF/NyquistF HPF/NyquistF]);%Butterworth 3rd order filter
     iPPG_signal30(isnan(iPPG_signal30))=nanmean(iPPG_signal30);%BVP_I(isnan(BVP_I))=0; 给Nan的点赋值
     BVP_F = filtfilt(B,A,double(iPPG_signal30));  %零相位数字滤波(butter 滤波器)
     iPPG_signal30  = BVP_F.';
% cwt filtering
wavelet_type      = 'amor';
wt                = cwt(iPPG_signal30, wavelet_type, Fs_video, 'VoicesPerOctave', 32, 'FrequencyLimits', [0.8 2.5]);    % 0.65 - 3 Hz in the original paper
wt_energy         = sum(abs(wt'));
wt_filt           = wt .* repmat(wt_energy, size(wt, 2), 1)';
BVP = icwt(wt_filt, 'amor');
    PR    = prpsd(BVP,Fs_video,40,240,true);%寻峰计数
      
end

