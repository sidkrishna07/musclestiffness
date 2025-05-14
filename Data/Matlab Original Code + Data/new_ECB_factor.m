function bandwidth = new_ECB_factor(sig1)

[tmp_s, f, t] = spectrogram(sig1,128, 64,1024,6500);

%[tmp_s] = spectrogram(sig1,128, 64,1024,6500);
        basic_f = mean(abs(tmp_s), 2);
        time_dom = mean(abs(tmp_s) - basic_f,1);
        fre_dom = mean(abs(tmp_s) - basic_f,2);
        %plot(t, time_dom)
        [peak, loc] = findpeaks(time_dom);
        sig_time_piece = 0;
        if length(loc) <1
            sig_time_piece = 1;
        else
            if loc(1) > 1
                sig_time_piece = loc(1);
            else
                sig_time_piece = loc(2);
            end
        end
        final_sig_f = tmp_s(:, sig_time_piece)';
        %hold on
        %plot(f, abs(final_sig_f)- abs(basic_f)')
        tmp_ff = abs(final_sig_f)- abs(basic_f)';
        tmp_ff(1) =0;
        
        bandwidth = NaN(1,3);
        en_th_val = [0.90, 0.75, 0.50, 0.25,0.20,0.15,0.10,0.05];
for energy_th_co =1:length(en_th_val)
    energy_th = en_th_val(energy_th_co);
X = tmp_ff;
S = X.^2;
P = S ./ sum(S);
CPF=[];
for k=1:size(P,2)
    CPF(k) = sum(P(1:k));
end

stop_band = [];
energy_p =[];
for k=1:size(P,2)
    if CPF(k)> 1- energy_th
        break;
    end
    target_p = CPF(k)+ energy_th;
    [val, loc] = min(abs(CPF-target_p));
    real_gap = CPF(loc)- CPF(k);
    if abs(real_gap - energy_th)>0.01
        continue;
    end
    stop_band = [stop_band, loc-k];
end

if size(stop_band,2) <1
    energy_p =[];
for k=1:size(P,2)
    if CPF(k)> 1- energy_th
        break;
    end
    target_p = CPF(k)+ energy_th;
    [val, loc] = min(abs(CPF-target_p));
    real_gap = CPF(loc)- CPF(k);
    if abs(real_gap - energy_th)>0.03
        continue;
    end
    stop_band = [stop_band, loc-k];
end
end
tmp_bandwidth = min(stop_band);
if length(tmp_bandwidth)<1
    continue;
end
bandwidth(energy_th_co) = tmp_bandwidth;
end
end