function data_Ph = removeImpulseNoise(data_Ph)  
    N_data = length(data_Ph);
    x1=1:N_data-2; y1 = data_Ph(x1);
    xx=2:N_data-1; yy = data_Ph(xx);
    x2=3:N_data;   y2 = data_Ph(x2);
    backward = (yy-y1)./abs(y1);
    forward  = (yy-y2)./abs(y2);
    threshod = 1.5;
    ind_1    = find(backward> threshod & forward> threshod);
    ind_2    = find(backward<-threshod & forward<-threshod);
    ind      = [ind_1,ind_2]; 
    NN       = length(ind);
    if NN>=1
        ind_data = ind+1;
        data_Ph(ind_data)=y1(ind)+(y2(ind)-y1(ind))/2;
    end
end