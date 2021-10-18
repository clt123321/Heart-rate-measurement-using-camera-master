function [BVP_F,BVP, PR] = ippg_face_dynamic(VidObj, Fs_video, StartTime, Duration,display)
    addpath(genpath('tools'))
        %% Parameters
    LPF = 0.7; %low cutoff frequency (Hz) - 0.7 Hz in reference
    HPF = 2.5; %high cutoff frequency (Hz) - 4.0 Hz in reference

    %% Load Video:

    FramesToRead       = ceil(Duration*Fs_video); %video may be encoded at slightly different frame rate
    RGB = zeros(FramesToRead,3);%initialize color signal
    FN  = 0;
    faceDetector = vision.CascadeObjectDetector();
    pointTracker = vision.PointTracker('MaxBidirectionalError', 2);
    numPts       = 0;

   %------------------------------------------------------------
%    t_list = [];
%    while hasFrame(VidObj)
%        VidFrame      = readFrame(VidObj);
%        t_list        = [t_list,VidObj.CurrentTime];     
%    end
%    tt  = t_list-StartTime;
%    ind = find(tt>0);
%    VidObj.CurrentTime=t_list(ind(1))
   %------------------------------------------------------------
   

   N_threshold =Duration*Fs_video;
   while hasFrame(VidObj) && FN< N_threshold
       FN = FN+1;    
       if mod(FN,200) == 0  %每200帧打印一次
           fprintf('Have handled %d frames\n',FN);
       end
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
            
            if (display==2)  %如果等于2 的时候会显示框， 可视化效果
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
       
        img_roi_ycbcr     = rgb2ycbcr(img_roi);%img_roi_ycbcr = imgaussfilt(img_roi_ycbcr, 3);  % optional
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
             BVP_I(FN) = sum(sum(img_roi_luv(:,:,1).*img_roi_skin))/sum(sum(img_roi_skin));
   end

 %% FILTERING USING CWT
    NyquistF  = 1/2*Fs_video;
    [B,A] = butter(3,[LPF/NyquistF HPF/NyquistF]);%Butterworth 3rd order filter
    BVP_I(isnan(BVP_I))=nanmean(BVP_I);%BVP_I(isnan(BVP_I))=0;

    save('signal1.mat','BVP_I');
    BVP_F = filtfilt(B,A,double(BVP_I));
    iPPG_signal  = BVP_F.';
    
    % cwt filtering
    wavelet_type      = 'amor';
    wt                = cwt(iPPG_signal, wavelet_type, Fs_video, 'VoicesPerOctave', 32, 'FrequencyLimits', [0.8 2.5]);
    wt_energy         = sum(abs(wt'));
    wt_filt           = wt .* repmat(wt_energy, size(wt, 2), 1)';
    BVP = icwt(wt_filt, 'amor');%使用解析小波对cwt求逆
    save('signal2.mat','BVP');
    PR    = prpsd(BVP,Fs_video,40,240,true);


end%end function
