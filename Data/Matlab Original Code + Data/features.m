function [corrs,lags, amp,decay_var,bandwidths,e12,e23,e13] = features(sig1,sig2,sig3,Fs)

[decay_var,amp] = exFitting(sig1,sig2,sig3,Fs);

energy1=sum(sig1.^2);
energy2=sum(sig2.^2);
energy3=sum(sig3.^2);

e12=energy1/energy2;
e23=energy2/energy3;
e13=energy1/energy3;

% [pxx1,w1]=pspectrum(sig1,Fs);
% [pxx2,w2]=pspectrum(sig2,Fs);
% [pxx3,w3]=pspectrum(sig3,Fs);
% 
% pw1=pxx1.*w1;
% pw2=pxx2.*w2;
% pw3=pxx3.*w3;
% 
% mpf1=sum(pw1)/sum(pxx1);
% mpf2=sum(pw2)/sum(pxx2);
% mpf3=sum(pw3)/sum(pxx3);
% 
% mpf=[mpf1,mpf2,mpf3];

lags=40;
[c1,l1] = selfCorr(sig1,lags);
[c2,l2] = selfCorr(sig2,lags);
[c3,l3] = selfCorr(sig3,lags);

corrs=[c1,c2,c3];
% m= max(corrs);

% index= find(corrs==m);

lags=[l1,l2,l3];
% lag=lags(index);

b1 = new_ECB_factor(sig1);
b2 = new_ECB_factor(sig2);
b3 = new_ECB_factor(sig3);
bandwidths = [b1(4:end),b2(4:end),b3(4:end)];
a=1;
%bandwidths=[b1(a),b2(a),b3(a),b4(a), b1(2),b2(2),b3(2),b4(2), b1(3),b2(3),b3(3),b4(3), b1(4),b2(4),b3(4),b4(4), b1(5),b2(5),b3(5),b4(5), b1(6),b2(6),b3(6),b4(6), b1(7),b2(7),b3(7),b4(7), b1(8),b2(8),b3(8),b4(8)];