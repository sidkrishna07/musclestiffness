fold=10;
load('norm_sub1_17.mat')
d17=f_total_s1;
l17=l_s1;
l17=l17./10;

Xs1=[d17(1:20,:);d17(31:50,:);d17(61:80,:);d17(91:110,:);d17(121:140,:);d17(151:170,:)];
Ys1=[l17(1:20);l17(31:50);l17(61:80);l17(91:110);l17(121:140);l17(151:170)];

Xte_s1=[d17(21:30,:);d17(51:60,:);d17(81:90,:);d17(111:120,:);d17(141:150,:);d17(171:180,:)];
Yte_s1=[l17(21:30);l17(51:60);l17(81:90);l17(111:120);l17(141:150);l17(171:180)];

s1_1 = fitrlinear(d17,l17,'Learner','leastsquares','KFold',fold);
pred1_1=kfoldPredict(s1_1);
mdl1_1=fitlm(Xs1,Ys1);
mse1_1 = kfoldLoss(s1_1);

load('norm_sub1_19.mat')
d17=f_total_s1;
l17=l_s1;
l17=l17./10;

Xs12=[d17(1:20,:);d17(31:50,:);d17(61:80,:);d17(91:110,:);d17(121:140,:);d17(151:170,:)];
Ys12=[l17(1:20);l17(31:50);l17(61:80);l17(91:110);l17(121:140);l17(151:170)];

Xte_s12=[d17(21:30,:);d17(51:60,:);d17(81:90,:);d17(111:120,:);d17(141:150,:);d17(171:180,:)];
Yte_s12=[l17(21:30);l17(51:60);l17(81:90);l17(111:120);l17(141:150);l17(171:180)];

s1_2 = fitrlinear(d17,l17,'Learner','leastsquares','KFold',fold);
pred1_2=kfoldPredict(s1_2);
mse1_2 = kfoldLoss(s1_2);
mdl1_2=fitlm(Xs12,Ys12);

load('norm_sub1_20.mat')
d17=f_total_s1;
l17=l_s1;
l17=l17./10;

Xs13=[d17(1:20,:);d17(31:50,:);d17(61:80,:);d17(91:110,:);d17(121:140,:);d17(151:170,:)];
Ys13=[l17(1:20);l17(31:50);l17(61:80);l17(91:110);l17(121:140);l17(151:170)];

Xte_s13=[d17(21:30,:);d17(51:60,:);d17(81:90,:);d17(111:120,:);d17(141:150,:);d17(171:180,:)];
Yte_s13=[l17(21:30);l17(51:60);l17(81:90);l17(111:120);l17(141:150);l17(171:180)];


s1_3 = fitrlinear(d17,l17,'Learner','leastsquares','KFold',fold);
pred1_3=kfoldPredict(s1_3);
mse1_3 = kfoldLoss(s1_3);
%sub1_17=[d17,l17];
mdl1_3=fitlm(Xs13,Ys13);


load('norm_sub2_19.mat')
d19=f_total_s2;
l19=l_s2;
l19=l19./10;

Xs2=[d19(1:20,:);d19(31:50,:);d19(61:80,:);d19(91:110,:);d19(121:140,:);d19(151:170,:)];
Ys2=[l19(1:20);l19(31:50);l19(61:80);l19(91:110);l19(121:140);l19(151:170)];

Xte_s2=[d19(21:30,:);d19(51:60,:);d19(81:90,:);d19(111:120,:);d19(141:150,:);d19(171:180,:)];
Yte_s2=[l19(21:30);l19(51:60);l19(81:90);l19(111:120);l19(141:150);l19(171:180)];

s2_1 = fitrlinear(d19,l19,'Learner','leastsquares','KFold',fold);
pred2_1=kfoldPredict(s2_1);
mse2_1 = kfoldLoss(s2_1);
mdl2_1=fitlm(Xs2,Ys2);

load('norm_sub2_26.mat')
d19=f_total_s2;
l19=l_s2;
l19=l19./10;

Xs22=[d19(1:20,:);d19(31:50,:);d19(61:80,:);d19(91:110,:);d19(121:140,:);d19(151:170,:)];
Ys22=[l19(1:20);l19(31:50);l19(61:80);l19(91:110);l19(121:140);l19(151:170)];

Xte_s22=[d19(21:30,:);d19(51:60,:);d19(81:90,:);d19(111:120,:);d19(141:150,:);d19(171:180,:)];
Yte_s22=[l19(21:30);l19(51:60);l19(81:90);l19(111:120);l19(141:150);l19(171:180)];

s2_2 = fitrlinear(d19,l19,'Learner','leastsquares','KFold',fold);
pred2_2=kfoldPredict(s2_2);
mse2_2= kfoldLoss(s2_2);
mdl2_2=fitlm(Xs22,Ys22);

load('norm_sub3_20.mat')
d20=f_total_s3;
l20=l_s3;
l20=l20./10;

Xs3=[d20(1:20,:);d20(31:50,:);d20(61:80,:);d20(91:110,:);d20(121:140,:);d20(151:170,:)];
Ys3=[l20(1:20);l20(31:50);l20(61:80);l20(91:110);l20(121:140);l20(151:170)];

Xte_s3=[d20(21:30,:);d20(51:60,:);d20(81:90,:);d20(111:120,:);d20(141:150,:);d20(171:180,:)];
Yte_s3=[l20(21:30);l20(51:60);l20(81:90);l20(111:120);l20(141:150);l20(171:180)];

s3_1 = fitrlinear(d20,l20,'Learner','leastsquares','KFold',fold);
pred3_1=kfoldPredict(s3_1);
mse3_1 = kfoldLoss(s3_1);
mdl3_1=fitlm(Xs3,Ys3);

load('norm_sub3_22.mat')
d20=f_total_s3;
l20=l_s3;
l20=l20./10;

Xs32=[d20(1:20,:);d20(31:50,:);d20(61:80,:);d20(91:110,:);d20(121:140,:);d20(151:170,:)];
Ys32=[l20(1:20);l20(31:50);l20(61:80);l20(91:110);l20(121:140);l20(151:170)];

Xte_s32=[d20(21:30,:);d20(51:60,:);d20(81:90,:);d20(111:120,:);d20(141:150,:);d20(171:180,:)];
Yte_s32=[l20(21:30);l20(51:60);l20(81:90);l20(111:120);l20(141:150);l20(171:180)];

s3_2 = fitrlinear(d20,l20,'Learner','leastsquares','KFold',fold);
pred3_2=kfoldPredict(s3_2);
mse3_2 = kfoldLoss(s3_2);
mdl3_2=fitlm(Xs32,Ys32);

load('norm_sub4_21.mat')
d21=f_total_s4;
l21=l_s4;
l21=l21./10;

Xs4=[d21(1:20,:);d21(31:50,:);d21(61:80,:);d21(91:110,:);d21(121:140,:);d21(151:170,:)];
Ys4=[l21(1:20);l21(31:50);l21(61:80);l21(91:110);l21(121:140);l21(151:170)];

Xte_s4=[d21(21:30,:);d21(51:60,:);d21(81:90,:);d21(111:120,:);d21(141:150,:);d21(171:180,:)];
Yte_s4=[l21(21:30);l21(51:60);l21(81:90);l21(111:120);l21(141:150);l21(171:180)];

s4_1 = fitrlinear(d21,l21,'Learner','leastsquares','KFold',fold);
pred4_1=kfoldPredict(s4_1);
mse4_1 = kfoldLoss(s4_1);
mdl4_1=fitlm(Xs4,Ys4);

load('norm_sub4_26.mat')
d21=f_total_s4;
l21=l_s4;
l21=l21./10;

Xs42=[d21(1:20,:);d21(31:50,:);d21(61:80,:);d21(91:110,:);d21(121:140,:);d21(151:170,:)];
Ys42=[l21(1:20);l21(31:50);l21(61:80);l21(91:110);l21(121:140);l21(151:170)];

Xte_s42=[d21(21:30,:);d21(51:60,:);d21(81:90,:);d21(111:120,:);d21(141:150,:);d21(171:180,:)];
Yte_s42=[l21(21:30);l21(51:60);l21(81:90);l21(111:120);l21(141:150);l21(171:180)];

s4_2 = fitrlinear(d21,l21,'Learner','leastsquares','KFold',fold);
pred4_2=kfoldPredict(s4_2);
mse4_2 = kfoldLoss(s4_2);
mdl4_2=fitlm(Xs42,Ys42);

load('norm_sub5_24.mat')
d24=f_total_s5;
l24=l_s5;
l24=l24./10;

Xs5=[d24(1:20,:);d24(31:50,:);d24(61:80,:);d24(91:110,:);d24(121:140,:);d24(151:170,:)];
Ys5=[l24(1:20);l24(31:50);l24(61:80);l24(91:110);l24(121:140);l24(151:170)];

Xte_s5=[d24(21:30,:);d24(51:60,:);d24(81:90,:);d24(111:120,:);d24(141:150,:);d24(171:180,:)];
Yte_s5=[l24(21:30);l24(51:60);l24(81:90);l24(111:120);l24(141:150);l24(171:180)];

figure;
plot(mdl1_1);
hold on;
plot(mdl1_3,'color','r');
plot(mdl2_1,'color','g');
plot(mdl2_2,'color','b');
plot(mdl3_1,'color','k');
plot(mdl3_2);
plot(mdl4_1);
plot(mdl4_2);

mdlsub1_1=regf(Xs1,Ys1,Xte_s1,Yte_s1);
mdlsub1_2=regf(Xs12,Ys12,Xte_s12,Yte_s12);
mdlsub1_3=regf(Xs13,Ys13,Xte_s13,Yte_s13);




mdlsub2_1=regf(Xs2,Ys2,Xte_s22,Yte_s22);
mdlsub2_2=regf(Xs22,Ys22,Xte_s22,Yte_s22);


mdlsub3_1=regf(Xs3,Ys3,Xte_s3,Yte_s3);
mdlsub3_2=regf(Xs32,Ys32,Xte_s32,Yte_s32);

mdlsub4_1=regf(Xs4,Ys4,Xte_s4,Yte_s4);
mdlsub4_2=regf(Xs42,Ys42,Xte_s4,Yte_s4);

mdlsub5_1=regf(Xs5,Ys5,Xte_s5,Yte_s5);
l=2;
a=[0,1,2,3,4,5];
figure;
hold on;

errorbar(a,mdlsub2_1(1,:),mdlsub2_1(2,:),'LineWidth',l);

errorbar(a,mdlsub4_2(1,:),mdlsub4_2(2,:),'LineWidth',l);

m=[mse1_1;mse1_2;mse1_3;mse2_1;mse2_2;mse3_1;mse3_2;mse4_1;mse4_2];

figure;
hold on;
pred=[pred1_1,pred1_2,pred1_3,pred2_1,pred2_2,pred3_1,pred3_2,pred4_1,pred4_2,l17];
err1_1=abs(pred(:,1)-pred(:,end));
mean11=[mean(pred1_1(1:30,:)),mean(pred1_1(31:60,:)),mean(pred1_1(61:90,:)),mean(pred1_1(91:120,:)),mean(pred1_1(121:150,:)),mean(pred1_1(151:180,:))];
std11 =[std(pred1_1(1:30,:)),std(pred1_1(31:60,:)),std(pred1_1(61:90,:)),std(pred1_1(91:120,:)),std(pred1_1(121:150,:)),std(pred1_1(151:180,:))];
% errorbar(a,mean11,std11,'LineWidth',l)
plot(a,mean11,'LineWidth',l)

err1_2=abs(pred(:,2)-pred(:,end));
mean12=[mean(pred1_2(1:30,:)),mean(pred1_2(31:60,:)),mean(pred1_2(61:90,:)),mean(pred1_2(91:120,:)),mean(pred1_2(121:150,:)),mean(pred1_2(151:180,:))];
std12 =[std(pred1_2(1:30,:)),std(pred1_2(31:60,:)),std(pred1_2(61:90,:)),std(pred1_2(91:120,:)),std(pred1_2(121:150,:)),std(pred1_2(151:180,:))];
%errorbar(a,mean12,std12)

err1_3=abs(pred(:,3)-pred(:,end));
mean13=[mean(pred1_3(1:30,:)),mean(pred1_3(31:60,:)),mean(pred1_3(61:90,:)),mean(pred1_3(91:120,:)),mean(pred1_3(121:150,:)),mean(pred1_3(151:180,:))];
std13 =[std(pred1_3(1:30,:)),std(pred1_3(31:60,:)),std(pred1_3(61:90,:)),std(pred1_3(91:120,:)),std(pred1_3(121:150,:)),std(pred1_3(151:180,:))];
errorbar(a,mean13,std13,'LineWidth',l)
plot(a,mean13,'LineWidth',l)

err2_1=abs(pred(:,4)-pred(:,end));
mean21=[mean(pred2_1(1:30,:)),mean(pred2_1(31:60,:)),mean(pred2_1(61:90,:)),mean(pred2_1(91:120,:)),mean(pred2_1(121:150,:)),mean(pred2_1(151:180,:))];
std21 =[std(pred2_1(1:30,:)),std(pred2_1(31:60,:)),std(pred2_1(61:90,:)),std(pred2_1(91:120,:)),std(pred2_1(121:150,:)),std(pred2_1(151:180,:))];
errorbar(a,mean21,std21,'LineWidth',l)
plot(a,mean21,'LineWidth',l)

err2_2=abs(pred(:,5)-pred(:,end));
mean22=[mean(pred2_2(1:30,:)),mean(pred2_2(31:60,:)),mean(pred2_2(61:90,:)),mean(pred2_2(91:120,:)),mean(pred2_2(121:150,:)),mean(pred2_2(151:180,:))];
std22 =[std(pred2_2(1:30,:)),std(pred2_2(31:60,:)),std(pred2_2(61:90,:)),std(pred2_2(91:120,:)),std(pred2_2(121:150,:)),std(pred2_2(151:180,:))];
errorbar(a,mean22,std22,'LineWidth',l)
plot(a,mean22,'LineWidth',l)

err3_1=abs(pred(:,6)-pred(:,end));
mean31=[mean(pred3_1(1:30,:)),mean(pred3_1(31:60,:)),mean(pred3_1(61:90,:)),mean(pred3_1(91:120,:)),mean(pred3_1(121:150,:)),mean(pred3_1(151:180,:))];
std31 =[std(pred3_1(1:30,:)),std(pred3_1(31:60,:)),std(pred3_1(61:90,:)),std(pred3_1(91:120,:)),std(pred3_1(121:150,:)),std(pred3_1(151:180,:))];
errorbar(a,mean31,std31,'LineWidth',l)
plot(a,mean31,'LineWidth',l)

err3_2=abs(pred(:,7)-pred(:,end));
mean32=[mean(pred3_2(1:30,:)),mean(pred3_2(31:60,:)),mean(pred3_2(61:90,:)),mean(pred3_2(91:120,:)),mean(pred3_2(121:150,:)),mean(pred3_2(151:180,:))];
std32 =[std(pred3_2(1:30,:)),std(pred3_2(31:60,:)),std(pred3_2(61:90,:)),std(pred3_2(91:120,:)),std(pred3_2(121:150,:)),std(pred3_2(151:180,:))];
errorbar(a,mean32,std32,'LineWidth',l)
plot(a,mean32,'LineWidth',l)

err4_1=abs(pred(:,8)-pred(:,end));
mean41=[mean(pred4_1(1:30,:)),mean(pred4_1(31:60,:)),mean(pred4_1(61:90,:)),mean(pred4_1(91:120,:)),mean(pred4_1(121:150,:)),mean(pred4_1(151:180,:))];
std41 =[std(pred4_1(1:30,:)),std(pred4_1(31:60,:)),std(pred4_1(61:90,:)),std(pred4_1(91:120,:)),std(pred4_1(121:150,:)),std(pred4_1(151:180,:))];
errorbar(a,mean41,std41,'LineWidth',l)
plot(a,mean41,'LineWidth',l)

err4_2=abs(pred(:,9)-pred(:,end));
cerr10=[err1_1,err1_2,err1_3,err2_1,err2_2,err3_1,err3_2,err4_1,err4_2];
mean42=[mean(pred4_2(1:30,:)),mean(pred4_2(31:60,:)),mean(pred4_2(61:90,:)),mean(pred4_2(91:120,:)),mean(pred4_2(121:150,:)),mean(pred4_2(151:180,:))];
std42 =[std(pred4_2(1:30,:)),std(pred4_2(31:60,:)),std(pred4_2(61:90,:)),std(pred4_2(91:120,:)),std(pred4_2(121:150,:)),std(pred4_2(151:180,:))];
errorbar(a,mean42,std42,'LineWidth',l)
plot(a,mean42,'LineWidth',l)

d=[-1,a,6];
plot(d,d,'LineWidth',l,'Color','k','linestyle','-');

% figure;
% hold on;
% % bar([mean(cerr3(:,1)),mean(cerr3(:,3)),mean(cerr3(:,4)),mean(cerr3(:,5)),mean(cerr3(:,6)),mean(cerr3(:,7)),mean(cerr3(:,8)),mean(cerr3(:,9));mean(cerr5(:,1)),mean(cerr5(:,3)),mean(cerr5(:,4)),mean(cerr5(:,5)),mean(cerr5(:,6)),mean(cerr5(:,7)),mean(cerr5(:,8)),mean(cerr5(:,9));   mean(cerr10(:,1)),mean(cerr10(:,3)),mean(cerr10(:,4)),mean(cerr10(:,5)),mean(cerr10(:,6)),mean(cerr10(:,7)),mean(cerr10(:,8)),mean(cerr10(:,9));mean(cerr15(:,1)),mean(cerr15(:,3)),mean(cerr15(:,4)),mean(cerr15(:,5)),mean(cerr15(:,6)),mean(cerr15(:,7)),mean(cerr15(:,8)),mean(cerr15(:,9))]);
% %errorbar([mean(err1_1),mean(err1_3),mean(err2_1),mean(err2_2),mean(err3_1),mean(err3_2),mean(err4_1),mean(err4_2)],[std(err1_1),std(err1_3),std(err2_1),std(err2_2),std(err3_1),std(err3_2),std(err4_1),std(err4_2)])
% bar([mean(err1_1),mean(err1_3),mean(err2_1),mean(err2_2),mean(err3_1),mean(err3_2),mean(err4_1),mean(err4_2)]);
% errorbar([mean(err1_1),mean(err1_3),mean(err2_1),mean(err2_2),mean(err3_1),mean(err3_2),mean(err4_1),mean(err4_2)],[std(err1_1),std(err1_3),std(err2_1),std(err2_2),std(err3_1),std(err3_2),std(err4_1),std(err4_2)])