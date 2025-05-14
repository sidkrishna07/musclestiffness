function [corr,lag] = selfCorr(sig,lags)

y = sig./sqrt(sum(sig.^2));
f=autocorr(y,'NumLags',lags);
[corr,lag]=max(f(2:end));