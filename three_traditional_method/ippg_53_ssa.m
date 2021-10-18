function [iPPG_time30, BVP, PR] = ippg_53_ssa(VidObj,Fs_video,StartTime,Duration,display)

% INPUTS: 
%	file: source folder path (.png images) or video path/filename.
%   display: 0 = no display, 1 = display signals only, 2 = display signals and face tracking.
%
% OUTPUTS:
%   iPPG_time30, iPPG_signal30filt: iPPG signal and time vectors (u* channel filtered using its CWT representation).
%   iPR_time, iPR: instantaneous (beat-to-beat) pulse rate.
%   iBR_time, iBR: instantaneous (beat-to-beat) breathing rate.
%
% References: face tracking and colour space parts: Frederic Bousefsaf, Alain Pruski, Choubeila Maaoui, Continuous wavelet filtering on webcam photoplethysmographic signals to remotely assess the instantaneous heart rate, Biomedical Signal Processing and Control, vol. 8, n? 6, pp. 568?574 (2013)
%            signal processing part:（2020）Detail-preserving pulse wave extraction from facial videos using consumer-level camera
   
%% PREPARE IMAGES / VIDEO LOADING
    LPF = 0.8; %low cutoff frequency (Hz) - 0.7 Hz in reference
    HPF = 2.5; %high cutoff frequency (Hz) - 4.0 Hz in reference

%VidObj       = VideoReader(f_VideoFile);
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

%% Resample when fs less than 30
if Fs_video<30%if Fs_video<30 Hz, resample to 30 Hz
    iPPG_time30   = iPPG_time(1):1/30:iPPG_time(end);
    signal = interp1(iPPG_time, iPPG_signal, iPPG_time30, 'pchip'); %线性插值
else
    iPPG_time30   = iPPG_time;
    signal = iPPG_signal;    
end
    %NyquistF  = 1/2*Fs_video;
    M=Fs_video; % sampling rate =61
    signal(isnan(signal))=nanmean(signal);%BVP_I(isnan(BVP_I))=0；给Nan的点赋值
    
%% Signal processing use method in 2020
%Step1  Baseline cancelation 去基线去趋势操作
%removes just the mean value from the vector X, or the mean value from each column, if X is a matrix.
signal= detrend(signal,'constant');

%Step2 Spike smoothing 尖峰平滑
signal=spike_smooth(signal,M);

%Step3  Five-point cubic smoothing 五三平滑
signal=mean5_3(signal,5);

%Step4 trajectory matrix construction迹矩阵构建
N=length(signal);
L=ceil(N/2);
K=N-L+1;
X=rajectory_matrix(signal,N,L,K);  % size(X)= L*K

%Step5 Singular value decomposition奇异值分解   
P=X*X';
[U,autoval]=eig(P);%eig返回矩阵的特征值和特征向量，U是特征向量，autoval是特征值
[~,d]=sort(diag(autoval),'descend');  %降序
U=U(:,d); %d是一个索引
V=(X')*U;
%[U,~,V] = svd(X);
%Step6 Self-adaptive components selection based on EALM  拉格朗日乘数法的自适应成分选取
S=inexact_alm_rpca(X);

%Step7 Signal reconstruction信号重构
r=rank(S);
VT=V';
RCA=U(:,1:r) * VT(1:r,:);   
%U的前r列  *  V的前r行    
%size(RCA)= L*K
BVP=rebuild(RCA,L,K,N);


%% 寻峰计数
PR    = prpsd(BVP,Fs_video,40,240,true);
      
end

