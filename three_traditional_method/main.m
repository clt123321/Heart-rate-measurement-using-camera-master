clc;clear all;close all;
addpath(genpath([cd '\tools\']));

t_start = clock;  

input_filefolder      = 'C:\study\demo1.0\Script\iPPG\dataset\'; %输入video所在文件夹  单一video测试
%input_filefolder      = 'D:\testdata_29\'; %输入video所在文件夹

output_filefolder     = strrep(pwd(),  'scripts','processed\');
if ~exist(output_filefolder)
     mkdir(output_filefolder);
end
File         = dir(fullfile(input_filefolder,'*.avi')); 
FileNames    = {File.name}';
Length_files = size(FileNames,1);
disp(Length_files);
for p = 1:Length_files  
    %________________ Step1: Load data ________________________________________
    %**(1.1)parameter setting ---------------
    Fs                = 256;           %Hz,Sampling Rate
    disp('number of video:');
    disp(p);
    name_ippg             = FileNames(p);
    
    disp('name of video:');
    disp(name_ippg);
    name_ippg             = name_ippg{1,1};
    ss_name               = erase(name_ippg,'.avi');
    
    f_VideoFile             = strcat(input_filefolder, name_ippg);%输入video路径
    f_iPPG_average         = [output_filefolder,'f_iPPG_good.xls']; %输出excel路径
    
    %________________ Step4: iPPG _____________________________________________
    %(4.1)load video
    disp(f_VideoFile)
    VidObj         = VideoReader(f_VideoFile);
    Fs_video       = floor(VidObj.FrameRate)
%     StartTime      = 0;                       % Timepoint at which to start process    (default = 0 seconds).
%     Duration       = floor(VidObj.Duration);  % Duration of the time window to process (default = 60 seconds).
%     Fs_video       = 61;
    StartTime      = 5;  
    Duration       =30;
    %----------------------------------------

    %(4.2)iPPG计算，--------------
% display: 0 = no display, 1 = display signals only, 2 = display signals and face tracking.
    %flag_display   = 0;
    %[data,data_1, PR_1] = ippg_face        (VidObj, Fs_video, StartTime, Duration,flag_display);    %1.0


    display   = 0; 
    %[data,data_1, PR_1] = ippg_face_dynamic       (VidObj, Fs_video, StartTime, Duration,display);     %2.0
    
    [data,data_1, PR_1] = ippg_53_ssa       (VidObj, Fs_video, StartTime, Duration,display);     %3.0
    % %(4.3)time，Cases 1-6--------------------
    T_iPPG_1       = (1:length(data_1))/(Fs_video*60);  
    T_list         = [max(T_iPPG_1)];
    t_xlim         = [0 0.5]
    % %----------------------------------------

    %输出到excel文件
    [tmp1,tmp2,tmpRaw]=xlsread(f_iPPG_average);
    if size(tmp1,1)==0&&size(tmp2,1)==0%是否是空文档
       mRowRange='1';
    else
       mRowRange=num2str(size(tmpRaw,1)+1);
    end
    xlswrite(f_iPPG_average,FileNames(p),'sheet1',['A' mRowRange]); 
    xlswrite(f_iPPG_average,PR_1,'sheet1',['B' mRowRange]); 

    %(5.2)显示结果---------------------------
    fprintf("__________________________________________________________________")
    fprintf("\n");
    fprintf('ippg,         hr_average is %.2f\n',PR_1);
    fprintf("__________________________________________________________________");
    %----------------------------------------
    
end

