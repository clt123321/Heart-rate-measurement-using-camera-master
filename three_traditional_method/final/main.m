clc;clear all;close all;
addpath(genpath([cd '\tools\']));

t_start = clock;  

input_filefolder      = 'C:\study\demo4.0\Heart-rate-measurement-using-camera-master\test videos\'; 
output_filefolder     = strrep(pwd(),  'scripts','processed\');
if ~exist(output_filefolder)
     mkdir(output_filefolder);
end
File         = dir(fullfile(input_filefolder,'*.avi')); 
FileNames    = {File.name}';
Length_files = size(FileNames,1);
disp(Length_files);
for p = 1:Length_files
% for p = 290:306

    %________________ Step1: Load data ________________________________________
    %**(1.1)parameter setting ---------------
    Fs                = 256;           %Hz,Sampling Rate
    disp(p);
    name_ippg             = FileNames(p);
    disp(name_ippg);
    name_ippg             = name_ippg{1,1};
    ss_name               = erase(name_ippg,'.avi');
    f_VideoFile             = strcat(input_filefolder, name_ippg);
    f_PPG_average         = [output_filefolder,'f_all_0326.xlsx'];
%     f_PPG                 = [output_filefolder,strcat(ss_name,'-PPG.csv')];
    
    %________________ Step4: iPPG _____________________________________________
    %(4.1)load video
    VidObj         = VideoReader(f_VideoFile);
    Fs_video       = floor(VidObj.FrameRate);
%     StartTime      = 0;                       % Timepoint at which to start process    (default = 0 seconds).
%     Duration       = floor(VidObj.Duration);  % Duration of the time window to process (default = 60 seconds).
%     Fs_video       = 61;
    StartTime      = 5;  
    Duration       =30;
    %----------------------------------------

    %(4.2)iPPG计算，--------------

    display   = 0;
    [data, PR_psd] = ippg_face_dynamic1        (VidObj, Fs_video, StartTime, Duration,display);

    %输出到excel文件
   if~exist(f_PPG_average,'file')
        xlswrite(f_PPG_average,FileNames(p),'sheet1','A');
        xlswrite(f_PPG_average,PR_psd,'sheet1','B');
        xlswrite(f_PPG_average,data,'sheet1','C');
   else
    [tmp1,tmp2,tmpRaw]=xlsread(f_PPG_average);
    if size(tmp1,1)==0&&size(tmp2,1)==0%是否是空文档
       mRowRange='1';
    else
       mRowRange=num2str(size(tmpRaw,1)+1);
    end
    xlswrite(f_PPG_average,FileNames(p),'sheet1',['A' mRowRange]); 
    xlswrite(f_PPG_average,PR_psd,'sheet1',['B' mRowRange]); 
    xlswrite(f_PPG_average,data,'sheet1',['C' mRowRange]); 
   end

    %(5.2)显示结果---------------------------
    fprintf("__________________________________________________________________")
    fprintf("\n");
    fprintf('ippg_psd,                   hr_average is %.2f\n',PR_psd);

    fprintf("__________________________________________________________________");
    %----------------------------------------

end

