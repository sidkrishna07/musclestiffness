load("26April/Sub4_2/sub4_postural_0.mat")
Fs=540;
f0=[];
pass=50;
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,0];
    f0=[f0;a];

end


load("26April/Sub4_2/sub4_postural_10.mat")
f10=[];
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,10];
    f10=[f10;a];

end


load("26April/Sub4_2/sub4_postural_20.mat")
f20=[];
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,20];
    f20=[f20;a];

end


load("26April/Sub4_2/sub4_postural_30.mat")
f30=[];
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,30];
    f30=[f30;a];

end


load("26April/Sub4_2/sub4_postural_40.mat")
f40=[];
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,40];
    f40=[f40;a];

end


load("26April/Sub4_2/sub4_postural_50.mat")
f50=[];
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,50];
    f50=[f50;a];

end


f_total_s4=[f0(:,1:end-1);f10(:,1:end-1);f20(:,1:end-1);f30(:,1:end-1);f40(:,1:end-1);f50(:,1:end-1)];
f_total_s4=normalize(f_total_s4,'range');
l_s4=[f0(:,end);f10(:,end);f20(:,end);f30(:,end);f40(:,end);f50(:,end)];

save('norm_sub4_26.mat','f_total_s4','l_s4')


%==================================================
%==================================================
%==================================================

load("20April/Sub1_2/sub1_postural_0.mat")
Fs=540;
f0=[];
pass=50;
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,0];
    f0=[f0;a];

end

load("20April/Sub1_2/sub1_postural_10.mat")
f10=[];
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,10];
    f10=[f10;a];

end


load("20April/Sub1_2/sub1_postural_20.mat")
f20=[];
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,20];
    f20=[f20;a];

end


load("20April/Sub1_2/sub1_postural_30.mat")
f30=[];
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,30];
    f30=[f30;a];

end


load("20April/Sub1_2/sub1_postural_40.mat")
f40=[];
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,40];
    f40=[f40;a];

end


load("20April/Sub1_2/sub1_postural_50.mat")
f50=[];
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,50];
    f50=[f50;a];

end


f_total_s1=[f0(:,1:end-1);f10(:,1:end-1);f20(:,1:end-1);f30(:,1:end-1);f40(:,1:end-1);f50(:,1:end-1)];
f_total_s1=normalize(f_total_s1,'range');
l_s1=[f0(:,end);f10(:,end);f20(:,end);f30(:,end);f40(:,end);f50(:,end)];

save('norm_sub1_20.mat','f_total_s1','l_s1')


%==================================================
%==================================================
%==================================================

load("26April/Sub2_2/sub2_postural_0.mat")
Fs=540;
f0=[];
pass=50;
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,0];
    f0=[f0;a];

end


load("26April/Sub2_2/sub2_postural_10.mat")
f10=[];
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,10];
    f10=[f10;a];

end


load("26April/Sub2_2/sub2_postural_20.mat")
f20=[];
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,20];
    f20=[f20;a];

end


load("26April/Sub2_2/sub2_postural_30.mat")
f30=[];
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,30];
    f30=[f30;a];

end


load("26April/Sub2_2/sub2_postural_40.mat")
f40=[];
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,40];
    f40=[f40;a];

end


load("26April/Sub2_2/sub2_postural_50.mat")
f50=[];
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,50];
    f50=[f50;a];

end


f_total_s2=[f0(:,1:end-1);f10(:,1:end-1);f20(:,1:end-1);f30(:,1:end-1);f40(:,1:end-1);f50(:,1:end-1)];
f_total_s2=normalize(f_total_s2,'range');
l_s2=[f0(:,end);f10(:,end);f20(:,end);f30(:,end);f40(:,end);f50(:,end)];

save('norm_sub2_26.mat','f_total_s2','l_s2')


%==================================================
%==================================================
%==================================================


load("20April/Sub3_1/sub3_postural_0.mat")
Fs=540;
f0=[];
pass=50;
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,0];
    f0=[f0;a];

end


load("20April/Sub3_1/sub3_postural_10.mat")
f10=[];
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,10];
    f10=[f10;a];

end


load("20April/Sub3_1/sub3_postural_20.mat")
f20=[];
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,20];
    f20=[f20;a];

end


load("20April/Sub3_1/sub3_postural_30.mat")
f30=[];
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,30];
    f30=[f30;a];

end


load("20April/Sub3_1/sub3_postural_40.mat")
f40=[];
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,40];
    f40=[f40;a];

end


load("20April/Sub3_1/sub3_postural_50.mat")
f50=[];
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,50];
    f50=[f50;a];

end


f_total_s3=[f0(:,1:end-1);f10(:,1:end-1);f20(:,1:end-1);f30(:,1:end-1);f40(:,1:end-1);f50(:,1:end-1)];
f_total_s3=normalize(f_total_s3,'range');
l_s3=[f0(:,end);f10(:,end);f20(:,end);f30(:,end);f40(:,end);f50(:,end)];


save('norm_sub3_20.mat','f_total_s3','l_s3')



%==================================================
%==================================================
%==================================================

load("17April/Sub1_1/sub1_postural_0.mat")
Fs=540;
f0=[];
pass=50;
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,0];
    f0=[f0;a];

end


load("17April/Sub1_1/sub1_postural_10.mat")
f10=[];
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,10];
    f10=[f10;a];

end


load("17April/Sub1_1/sub1_postural_20.mat")
f20=[];
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,20];
    f20=[f20;a];

end


load("17April/Sub1_1/sub1_postural_30.mat")
f30=[];
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,30];
    f30=[f30;a];

end


load("17April/Sub1_1/sub1_postural_40.mat")
f40=[];
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,40];
    f40=[f40;a];

end


load("17April/Sub1_1/sub1_postural_50.mat")
f50=[];
for i =1:30
    d1 = highpass(D1(i,:),pass,Fs);
    d2 = highpass(D2(i,:),pass,Fs);
    d3 = highpass(D3(i,:),pass,Fs);

    [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13]= features(d1,d2,d3,Fs);
    a=[mean(corrs),mean(lags), amp,decay_var,bandwidths,e12,e23,e13,50];
    f50=[f50;a];

end


f_total_s1=[f0(:,1:end-1);f10(:,1:end-1);f20(:,1:end-1);f30(:,1:end-1);f40(:,1:end-1);f50(:,1:end-1)];
f_total_s1=normalize(f_total_s1,'range');
l_s1=[f0(:,end);f10(:,end);f20(:,end);f30(:,end);f40(:,end);f50(:,end)];

save('norm_sub1_20.mat','f_total_s1','l_s1')


