clc;
clear all;
close all;

%初始化参数
fs = 73.728e6; % 采样率
flo = 4.999e6;%期望信号的中心频率
t = 1/fs:1/fs:10e-2; %仿10毫秒数据
f = 5e6; % 输入信号频率

%生成仿真需要的输入信号，并对信号幅度加衰落且定点化
bitNumInSig=16;
inSig = .9*sin(2*pi*f*t);
plot(t,inSig)
inSig(1e5:2e5) = .3*inSig(1e5:2e5);
inSig(3e5:end) = .8*inSig(3e5:end);
inSig=round(inSig*2^(bitNumInSig-1));

%生成定点的本振信号
bitNumNcoSig=19;
N = 2^22;
sinwave = round(2^(bitNumNcoSig-1)*sin(linspace(0,2*pi-2*pi/N,N))); %生成单周期的余弦，并定点化
ssin = sinwave(mod(round(mod(flo*t,1)*N),N)+1);  %生成sin
scos = sinwave(mod(round(mod(flo*t,1)*N)+N/4,N)+1);  %生成cos
nco = ssin+1i*scos;

%与输入信号进行混频，并对其输出信号位宽重新进行定义
mixOut = inSig.*nco;
mixOut=round(mixOut/2^(bitNumNcoSig-1)); 

%CIC滤波抽取，并对其输出信号位宽重新进行定义
bitNumCicStages=30; %积分器与滤波器的位宽
bitNumICicOut=bitNumInSig; %CIC输出的位宽
cicDecimatRatio=2^5*9; %cic抽取288倍
cicStagesNum=3;%cic级数
cicGain=cicDecimatRatio^cicStagesNum; %计算此CIC增益
% hm = mfilt.cicdecim(cicDecimatRatio,1,cicStagesNum);
% hm.FilterInternals='FullPrecision';
% cicOut = double(filter(hm,mixOut));
% cicOut=round(cicOut/2^(log2(cicGain)));
hm = mfilt.cicdecim(cicDecimatRatio,1,cicStagesNum,bitNumInSig,bitNumICicOut,bitNumCicStages);
sts=int(hm.states);
set(hm,'InputFracLength',0);
cicOut = double(filter(hm,mixOut));
cicOut=round(cicOut/cicGain);



%第一级半带滤波，并对其输出信号位宽重新进行定义
bitNumHf1Coe=19;
% b = firhalfband(8,0.01,'dev'); %8指的是滤波器阶数，0.01指的是通带内抖动
% h = mfilt.firdecim(2,b);
% h.Numerator=round(h.Numerator*2^(bitNumHf1Coe-1));
h=round([-0.031303406 0 0.281280518 0.499954244 0.281280518 0 -0.031303406]*2^(bitNumHf1Coe-1));
hf1Out =filter(h,1,cicOut);
hf1Out=round(hf1Out/2^(bitNumHf1Coe-1));
hf1Out=downsample(hf1Out,2);

%第二级半带滤波，并对其输出信号位宽重新进行定义
bitNumHf2Coe=19;
% b = firhalfband(8,0.01,'dev'); %8指的是滤波器阶数，0.01指的是通带内抖动
% h = mfilt.firdecim(2,b);
% h.Numerator=round(h.Numerator*2^(bitNumHf2Coe-1));
h=round([0.005929947 0 -0.049036026 0 0.29309082 0.499969482 0.29309082 0 -0.049036026 0 0.005929947]*2^(bitNumHf2Coe-1));
hf2Out =filter(h,1,hf1Out);
hf2Out=round(hf2Out/2^(bitNumHf2Coe-1));
hf2Out=downsample(hf2Out,2);

%第三级半带滤波，并对其输出信号位宽重新进行定义
bitNumHf3Coe=19;
% b = firhalfband(8,0.01,'dev'); %8指的是滤波器阶数，0.01指的是通带内抖动
% h = mfilt.firdecim(2,b);
% h.Numerator=round(h.Numerator*2^(bitNumHf3Coe-1));
h=round([-0.00130558 0 0.012379646 0 -0.06055069 0 0.299453735 0.499954224 0.299453735 0 -0.06055069 0 0.012379646 0 -0.00130558]*2^(bitNumHf3Coe-1));
hf3Out =filter(h,1,hf2Out);
hf3Out=round(hf3Out/2^(bitNumHf3Coe-1));
hf3Out=downsample(hf3Out,2);

%FIR滤波,并对其输出信号位宽重新进行定义
bitNumFirCoe=19;
hm = design(fdesign.lowpass(0.06,0.09,0.5,80));  %系数分别对应Fp（通带截止频率）、Fst（阻带起始频率）、Ap（通带内抖动）、Ast（阻带衰减）
hm.Numerator=round(hm.Numerator*2^(bitNumFirCoe-1)); 
firOut = filter(hm,hf3Out);
firOut=round(firOut/2^(bitNumFirCoe-1)); 

%对信号进行4倍抽取
firDecimatRatio=4;
AGCIn=downsample(firOut,firDecimatRatio);
sig_fix=AGCIn;
plot(real(AGCIn));
% AGC模块
% 根据不同的应用AGC分为固定增益变化率和非固定增益变化率
% 固定增益变化率：取值范围为0~1.4063dB（增益因子M=15，E=15）
% 非固定增益变化率：取值范围为0~1.9884dB
% 定增量增益指的是每output点，累加器增加的增益值为固定值
% 如需要每output增加0.1dB增益，则M=9，E=12
% 非定增量增益情况：每output最大增益变化率=1.5*agc_loop_gain_fac*thread
% agc模块应该支持固定增益，固定增益变化率，非固定增益变化率这三种模式
% 非固定增益变化率中，又应该支持两个增益因子（attack和decay）
% 信号从无到有时，agc_error为负值时，用attack系数，一般比较大
% 信号从有到无时，agc_error为正值时，用decay系数，一般比较小
% 归一化目标门限
thread = 0.6;
% 15位定点门限
thread_fix = fix(thread*2^15);
% attack增益因子
loop_gain_m_a = 14;
loop_gain_e_a = 15;
% decay增益因子
loop_gain_m_d = 14;
loop_gain_e_d = 15;
% 计算环路增益量化因子
agc_loop_gain_fac_a = loop_gain_m_a/16/2^(15-loop_gain_e_a);
agc_loop_gain_fac_d = loop_gain_m_d/16/2^(15-loop_gain_e_d);
% 初始化变量
sum_gain = 0; % 累计增益
mlt_agc_exp = zeros(1,length(sig_fix)); % 线性增益的指数部分
mlt_agc_mantissa = zeros(1,length(sig_fix)); % 线性增益的尾数部分
sig_shift = zeros(1,length(sig_fix));
mul_agc_sig = zeros(1,length(sig_fix)); % 输入信号经过agc的结果
polar_sig = zeros(1,length(sig_fix)); % 信号的摸
agc_error = zeros(1,length(sig_fix)); % 信号与期望的误差
agc_loop_gain = zeros(1,length(sig_fix)); % 环路增益
sum_gain_look = zeros(1,length(sig_fix)); % 中间观测变量
% -----------------------AGC------------------------
for i = 1:length(sig_fix)
% 对数增益到线性增益的变换，把指数增益除以4的结果，取整数部分E，和小数部分M
% E作为线性增益的指数部分，M作为线性增益的小数部分。
    flag = 0;
    mlt_agc_exp(i) = fix(sum_gain/4);
    mlt_agc_mantissa(i) = sum_gain/4-mlt_agc_exp(i);
    sig_shift(i) = sig_fix(i)*2^mlt_agc_exp(i);
    mul_agc_sig(i) = sig_shift(i)*(1+mlt_agc_mantissa(i)); % 该值agc的结果
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
    % 计算信号的模，并量化到15位无符号数
    polar_sig(i) = fix(abs(mul_agc_sig(i)/2^8));
    if polar_sig(i) > 2^15-1
        polar_sig(i) = 2^15-1;
    end
    % 取信号幅度和期望值的误差
    agc_error(i) = (thread_fix-polar_sig(i))/2^15;
    if agc_error(i) >= 0
        % 计算增益
        agc_loop_gain(i) = agc_loop_gain_fac_d*(agc_error(i));
    else
        if flag == 0
            agc_loop_gain(i) = agc_loop_gain_fac_a*(agc_error(i));
        else
            agc_loop_gain(i) = -4;
%             agc_loop_gain(i) = agc_loop_gain_fac_a*(agc_error(i))*64;
        end
    end
    % 累计增益
    sum_gain = sum_gain+agc_loop_gain(i);
end
figure;
hold on;

plot(real(mul_agc_sig));
