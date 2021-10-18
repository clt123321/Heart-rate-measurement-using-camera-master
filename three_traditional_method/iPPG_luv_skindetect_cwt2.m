function [iPPG_time30, iPPG_signal30filt, iPR_time, iPR, iBR_time, iBR] = ippg_luv_skindetect_cwt2(f_VideoFile,Fs_video,display)
   
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
VidObj       = VideoReader(f_VideoFile);
length_vid   = floor(VidObj.FrameRate*VidObj.Duration);   %fr=30fps 帧率乘时长等于总帧数
iPPG_time    = 0:1/VidObj.FrameRate:(VidObj.Duration-1/VidObj.FrameRate);%0：一帧的用时：总时长-一帧的时长
iPPG_signal  = zeros(1,length_vid);

%% INIT FACE DETECTION AND TRACKING
faceDetector = vision.CascadeObjectDetector();%使用Viola-Jones算法创建一个检测器来检测对象，找到需要识别的脸
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);%创建一个点的踪迹将错误进行双向限制，以使它即使在噪声存在时也能正常表示 
numPts       = 0;

vidObj       = VideoReader(f_VideoFile);%读取一段录像
for i = 1:length_vid
    %% LOAD IMAGES
    img      = readFrame(VidObj);   %读取帧
    img      = double(img)/255; %读取图像，三维彩图
    img_gray = rgb2gray(img);  %灰度图像，二维
    
    %% FACE DETECTION AND TRACKING 脸部识别和追踪
    if numPts < 10
        % Detection mode检测方式
        bbox = faceDetector.step(img_gray);
        
        if ~isempty(bbox)%判断是否为空
            % Find corner points inside the detected region.
            points = detectMinEigenFeatures(img_gray, 'ROI', bbox(1, :)); %最小特征值算法检测角点，ROI默认为[1,1,size(I,1),size(1)]，表示进行角点检测的图像区域；bbox(1, :)是ROI的值
            
            % Re-initialize the point tracker.初始化
            xyPoints    = points.Location;
            numPts      = size(xyPoints,1);release(pointTracker);initialize(pointTracker, xyPoints, img_gray);
            oldPoints   = xyPoints;    %保存点的副本 
            bboxPoints  = bbox2points(bbox(1, :));%将表示为[x，y，w，h]的矩形转换为四个角的[x，y]坐标的M-by-2矩阵。 这需要能够转换边框来显示人脸的方向。
            bboxPolygon = reshape(bboxPoints', 1, []);%将框角转换为插入形状所需的[x1y1x2y2x3y3x4y4]格式。
            
            if (display==2)
                figure(1)
                subplot(1,2,1)
                
                % Display a bounding box around the detected face显示边界框
                img_display = insertShape(img, 'Polygon', bboxPolygon, 'LineWidth', 3);
                
                % Display detected corners显示检测到的点
                img_display = insertMarker(img_display, xyPoints, '+', 'Color', 'white');
                
                imshow(img_display)
            end
        end
        
    else
        % Tracking mode 跟踪模式
        [xyPoints, isFound] = step(pointTracker, img_gray);
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
                img_display = insertShape(img, 'Polygon', bboxPolygon, 'LineWidth', 3);
                
                % Display tracked points
                img_display = insertMarker(img_display, visiblePoints, '+', 'Color', 'white');
                
                imshow(img_display)
            end
        end
    end
    
    img_roi = img(bbox(2):bbox(2)+bbox(4), bbox(1):bbox(1)+bbox(3), :);
    
    
    %% SKIN DETECTION 转换成ycbcr色彩空间
    img_roi_ycbcr     = rgb2ycbcr(img_roi);%img_roi_ycbcr = imgaussfilt(img_roi_ycbcr, 3);  % optional
   
    img_roi_thresh_Y  = img_roi_ycbcr(:,:,1) > 80/255;
    img_roi_thresh_Cb = img_roi_ycbcr(:,:,2) > 77/255 & img_roi_ycbcr(:,:,2) < 127/255;
    img_roi_thresh_Cr = img_roi_ycbcr(:,:,3) > 133/255 & img_roi_ycbcr(:,:,3) < 173/255;    
    img_roi_skin      = img_roi_thresh_Y & img_roi_thresh_Cb & img_roi_thresh_Cr;
    
    
    %% COLORSPACE CONVERSION
    img_roi_luv = rgb2xyz(img_roi);%转成xyz色彩空间
    cform       = makecform('xyz2uvl');
    img_roi_luv = applycform(img_roi_luv, cform);
    
    if (display==2)
        subplot(1,2,2)
        imshow(img_roi_skin)
    end
    
    
    %% IMAGE -> SIGNAL
    iPPG_signal(i) = sum(sum(img_roi_luv(:,:,1).*img_roi_skin))/sum(sum(img_roi_skin));
    
    
    %% DISPLAY INFORMATIONS (STATE OF ADVANCEMENT)
    if (mod(i, 10)==0)
        disp([int2str(i) ' over ' int2str(length(iPPG_time)) ' frames have been processed'])
    end
end


%% FILTERING USING CWT 使用CWT（连续小波变换）过滤
if Fs_video<30%if Fs_video<30 Hz, resample to 30 Hz
    iPPG_time30   = iPPG_time(1):1/30:iPPG_time(end);
    iPPG_signal30 = interp1(iPPG_time, iPPG_signal, iPPG_time30, 'pchip');
else
    iPPG_time30   = iPPG_time;
    iPPG_signal30 = iPPG_signal;    
end

% cwt filtering
wavelet_type      = 'amor';
wt                = cwt(iPPG_signal30, wavelet_type, Fs_video, 'VoicesPerOctave', 32, 'FrequencyLimits', [0.667 4]);    % 0.65 - 3 Hz in the original paper
wt_energy         = sum(abs(wt'));
wt_filt           = wt .* repmat(wt_energy, size(wt, 2), 1)';
iPPG_signal30filt = icwt(wt_filt, 'amor');


%% INSTANTANEOUS (BEAT-TO-BEAT) PULSE RATE
fs256                 = 256;
iPPG_time256          = iPPG_time(1):1/fs256:iPPG_time(end);
iPPG_signal256filt    = interp1(iPPG_time30, iPPG_signal30filt, iPPG_time256, 'spline');

[pks_iPPG, locs_iPPG] = findpeaks(iPPG_signal256filt*-1, 'MinPeakHeight', 0, 'MinPeakDistance', 256/4);
pks_iPPG              = pks_iPPG * -1;

iPR_time              = iPPG_time256(locs_iPPG);
iPR                   = gradient(iPPG_time256(locs_iPPG));
iPR                   = 60./iPR;

% PR    = prpsd(iPR,Fs_video,40,240,true);
%% INSTANTANEOUS (BEAT-TO-BEAT) BREATHING RATE
%resample to 30 Hz
iPR_time30            = iPR_time(1):1/30:iPR_time(end);
iPR30                 = interp1(iPR_time, iPR, iPR_time30, 'pchip');    % linear in the article

wt                    = cwt(iPR30, wavelet_type, Fs_video, 'VoicesPerOctave', 32, 'FrequencyLimits', [0.15 0.4]);
wt_energy             = sum(abs(wt'));
wt_filt               = wt .* repmat(wt_energy, size(wt, 2), 1)';
iPR30filt             = icwt(wt_filt, 'amor');

[pks_iPR, locs_iPR]   = findpeaks(iPR30filt, 'MinPeakHeight', 0, 'MinPeakDistance', 30/0.4);
pks_iPR               = pks_iPR * -1;

iBR_time              = iPR_time30(locs_iPR);
iBR                   = gradient(iPR_time30(locs_iPR));
iBR                   = 1./iBR;

% BR    = prpsd(iBR,Fs_video,40,240,true);

%% DISPLAY RESULTS
if (display >= 1)
    figure(2)
    subplot(3,1,1);plot(iPPG_time,               (iPPG_signal-mean(iPPG_signal))/std(iPPG_signal), ...
                        iPPG_time256,            iPPG_signal256filt/std(iPPG_signal256filt),...
                        iPPG_time256(locs_iPPG), iPPG_signal256filt(locs_iPPG)/std(iPPG_signal256filt), '*r');
                   legend('raw', 'filtered', 'min');title('iPPG signals');ylabel('a.u.');   
    subplot(3,1,2);stairs(iPR_time, iPR);title('beat-to-beat PR');ylabel('beats per minute');
    subplot(3,1,3);stairs(iBR_time, iBR);title('beat-to-beat BR');xlabel('Time (s)');ylabel('breaths per minute');
end


end

