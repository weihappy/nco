clc;
clear all;
close all;

%��ʼ������
fs = 73.728e6; % ������
flo = 4.999e6;%�����źŵ�����Ƶ��
t = 1/fs:1/fs:10e-2; %��10��������
f = 5e6; % �����ź�Ƶ��

%���ɷ�����Ҫ�������źţ������źŷ��ȼ�˥���Ҷ��㻯
bitNumInSig=16;
inSig = .9*sin(2*pi*f*t);
plot(t,inSig)
inSig(1e5:2e5) = .3*inSig(1e5:2e5);
inSig(3e5:end) = .8*inSig(3e5:end);
inSig=round(inSig*2^(bitNumInSig-1));

%���ɶ���ı����ź�
bitNumNcoSig=19;
N = 2^22;
sinwave = round(2^(bitNumNcoSig-1)*sin(linspace(0,2*pi-2*pi/N,N))); %���ɵ����ڵ����ң������㻯
ssin = sinwave(mod(round(mod(flo*t,1)*N),N)+1);  %����sin
scos = sinwave(mod(round(mod(flo*t,1)*N)+N/4,N)+1);  %����cos
nco = ssin+1i*scos;

%�������źŽ��л�Ƶ������������ź�λ�����½��ж���
mixOut = inSig.*nco;
mixOut=round(mixOut/2^(bitNumNcoSig-1)); 

%CIC�˲���ȡ������������ź�λ�����½��ж���
bitNumCicStages=30; %���������˲�����λ��
bitNumICicOut=bitNumInSig; %CIC�����λ��
cicDecimatRatio=2^5*9; %cic��ȡ288��
cicStagesNum=3;%cic����
cicGain=cicDecimatRatio^cicStagesNum; %�����CIC����
% hm = mfilt.cicdecim(cicDecimatRatio,1,cicStagesNum);
% hm.FilterInternals='FullPrecision';
% cicOut = double(filter(hm,mixOut));
% cicOut=round(cicOut/2^(log2(cicGain)));
hm = mfilt.cicdecim(cicDecimatRatio,1,cicStagesNum,bitNumInSig,bitNumICicOut,bitNumCicStages);
sts=int(hm.states);
set(hm,'InputFracLength',0);
cicOut = double(filter(hm,mixOut));
cicOut=round(cicOut/cicGain);



%��һ������˲�������������ź�λ�����½��ж���
bitNumHf1Coe=19;
% b = firhalfband(8,0.01,'dev'); %8ָ�����˲���������0.01ָ����ͨ���ڶ���
% h = mfilt.firdecim(2,b);
% h.Numerator=round(h.Numerator*2^(bitNumHf1Coe-1));
h=round([-0.031303406 0 0.281280518 0.499954244 0.281280518 0 -0.031303406]*2^(bitNumHf1Coe-1));
hf1Out =filter(h,1,cicOut);
hf1Out=round(hf1Out/2^(bitNumHf1Coe-1));
hf1Out=downsample(hf1Out,2);

%�ڶ�������˲�������������ź�λ�����½��ж���
bitNumHf2Coe=19;
% b = firhalfband(8,0.01,'dev'); %8ָ�����˲���������0.01ָ����ͨ���ڶ���
% h = mfilt.firdecim(2,b);
% h.Numerator=round(h.Numerator*2^(bitNumHf2Coe-1));
h=round([0.005929947 0 -0.049036026 0 0.29309082 0.499969482 0.29309082 0 -0.049036026 0 0.005929947]*2^(bitNumHf2Coe-1));
hf2Out =filter(h,1,hf1Out);
hf2Out=round(hf2Out/2^(bitNumHf2Coe-1));
hf2Out=downsample(hf2Out,2);

%����������˲�������������ź�λ�����½��ж���
bitNumHf3Coe=19;
% b = firhalfband(8,0.01,'dev'); %8ָ�����˲���������0.01ָ����ͨ���ڶ���
% h = mfilt.firdecim(2,b);
% h.Numerator=round(h.Numerator*2^(bitNumHf3Coe-1));
h=round([-0.00130558 0 0.012379646 0 -0.06055069 0 0.299453735 0.499954224 0.299453735 0 -0.06055069 0 0.012379646 0 -0.00130558]*2^(bitNumHf3Coe-1));
hf3Out =filter(h,1,hf2Out);
hf3Out=round(hf3Out/2^(bitNumHf3Coe-1));
hf3Out=downsample(hf3Out,2);

%FIR�˲�,����������ź�λ�����½��ж���
bitNumFirCoe=19;
hm = design(fdesign.lowpass(0.06,0.09,0.5,80));  %ϵ���ֱ��ӦFp��ͨ����ֹƵ�ʣ���Fst�������ʼƵ�ʣ���Ap��ͨ���ڶ�������Ast�����˥����
hm.Numerator=round(hm.Numerator*2^(bitNumFirCoe-1)); 
firOut = filter(hm,hf3Out);
firOut=round(firOut/2^(bitNumFirCoe-1)); 

%���źŽ���4����ȡ
firDecimatRatio=4;
AGCIn=downsample(firOut,firDecimatRatio);
sig_fix=AGCIn;
plot(real(AGCIn));
% AGCģ��
% ���ݲ�ͬ��Ӧ��AGC��Ϊ�̶�����仯�ʺͷǹ̶�����仯��
% �̶�����仯�ʣ�ȡֵ��ΧΪ0~1.4063dB����������M=15��E=15��
% �ǹ̶�����仯�ʣ�ȡֵ��ΧΪ0~1.9884dB
% ����������ָ����ÿoutput�㣬�ۼ������ӵ�����ֵΪ�̶�ֵ
% ����Ҫÿoutput����0.1dB���棬��M=9��E=12
% �Ƕ��������������ÿoutput�������仯��=1.5*agc_loop_gain_fac*thread
% agcģ��Ӧ��֧�̶ֹ����棬�̶�����仯�ʣ��ǹ̶�����仯��������ģʽ
% �ǹ̶�����仯���У���Ӧ��֧�������������ӣ�attack��decay��
% �źŴ��޵���ʱ��agc_errorΪ��ֵʱ����attackϵ����һ��Ƚϴ�
% �źŴ��е���ʱ��agc_errorΪ��ֵʱ����decayϵ����һ��Ƚ�С
% ��һ��Ŀ������
thread = 0.6;
% 15λ��������
thread_fix = fix(thread*2^15);
% attack��������
loop_gain_m_a = 14;
loop_gain_e_a = 15;
% decay��������
loop_gain_m_d = 14;
loop_gain_e_d = 15;
% ���㻷·������������
agc_loop_gain_fac_a = loop_gain_m_a/16/2^(15-loop_gain_e_a);
agc_loop_gain_fac_d = loop_gain_m_d/16/2^(15-loop_gain_e_d);
% ��ʼ������
sum_gain = 0; % �ۼ�����
mlt_agc_exp = zeros(1,length(sig_fix)); % ���������ָ������
mlt_agc_mantissa = zeros(1,length(sig_fix)); % ���������β������
sig_shift = zeros(1,length(sig_fix));
mul_agc_sig = zeros(1,length(sig_fix)); % �����źž���agc�Ľ��
polar_sig = zeros(1,length(sig_fix)); % �źŵ���
agc_error = zeros(1,length(sig_fix)); % �ź������������
agc_loop_gain = zeros(1,length(sig_fix)); % ��·����
sum_gain_look = zeros(1,length(sig_fix)); % �м�۲����
% -----------------------AGC------------------------
for i = 1:length(sig_fix)
% �������浽��������ı任����ָ���������4�Ľ����ȡ��������E����С������M
% E��Ϊ���������ָ�����֣�M��Ϊ���������С�����֡�
    flag = 0;
    mlt_agc_exp(i) = fix(sum_gain/4);
    mlt_agc_mantissa(i) = sum_gain/4-mlt_agc_exp(i);
    sig_shift(i) = sig_fix(i)*2^mlt_agc_exp(i);
    mul_agc_sig(i) = sig_shift(i)*(1+mlt_agc_mantissa(i)); % ��ֵagc�Ľ��
    % limit
    if (abs(real(mul_agc_sig(i))) > 2^23)
        flag = 1;
        mul_agc_sig(i) = sign(real(mul_agc_sig(i)))*2^23+1i*imag(mul_agc_sig(i));
    end
    if (abs(imag(mul_agc_sig(i))) > 2^23)
        %
        flag = 1;
        mul_agc_sig(i) = real(mul_agc_sig(i))+1i*sign(imag(mul_agc_sig(i)))*2^23;
    end
    %%%%%%%%%%%%%%%%%%%
    % �����źŵ�ģ����������15λ�޷�����
    polar_sig(i) = fix(abs(mul_agc_sig(i)/2^8));
    if polar_sig(i) > 2^15-1
        polar_sig(i) = 2^15-1;
    end
    % ȡ�źŷ��Ⱥ�����ֵ�����
    agc_error(i) = (thread_fix-polar_sig(i))/2^15;
    if agc_error(i) >= 0
        % ��������
        agc_loop_gain(i) = agc_loop_gain_fac_d*(agc_error(i));
    else
        if flag == 0
            agc_loop_gain(i) = agc_loop_gain_fac_a*(agc_error(i));
        else
            agc_loop_gain(i) = -4;
%             agc_loop_gain(i) = agc_loop_gain_fac_a*(agc_error(i))*64;
        end
    end
    % �ۼ�����
    sum_gain = sum_gain+agc_loop_gain(i);
end
figure;
hold on;

plot(real(mul_agc_sig));
