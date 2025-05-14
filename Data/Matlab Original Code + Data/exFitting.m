function [a,b] = exFitting(sig1,sig2,sig3,Fs)

dist=[1,2,3];

[f,P1]= fft_implementation(sig1,Fs,length(sig1));
[f,P2]= fft_implementation(sig2,Fs,length(sig2));
[f,P3]= fft_implementation(sig3,Fs,length(sig3));

max1=max(P1);
index1= find(P1==max1);

max2=max(P2);
index2= find(P2==max2);

max3=max(P3);
index3= find(P3==max3);


if max1>max2  && max1>max3 
    if index1-25 > 0
        start= index1-25;
    else
        start=0;
    end
    
    if index1+25 < length(P1)
        endl= index1-25;
    else
        endl=length(P1);
    end
elseif max1<max2  && max2>max3 
    if index2-25 > 0
        start= index2-25;
    else
        start=0;
    end
    
    if index2+25 < length(P2)
        endl= index2-25;
    else
        endl=length(P1);
    end
else
    if index3-25 > 0
        start= index3-25;
    else
        start=0;
    end
    
    if index3+25 < length(P3)
        endl= index3-25;
    else
        endl=length(P3);
    end
end
    
    
energy1= sum(P1(start:endl));   
energy2= sum(P2(start:endl)); 
energy3= sum(P3(start:endl));

% P=[index1,index2,index3;P1,P2,P3];

energy=[energy1,energy2,energy3];
% dist,energy
f = fit(dist',energy', 'exp1');
a=f.a;
b= f.b;


