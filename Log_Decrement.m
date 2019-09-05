%% Meca380, S2017, Logarithmic Decrement Example

clear
S=importdata('Step_input_strain.xlsx') % import data
figure(1); clf; plot(S.data(:,1),S.data(:,2),'.'); % plot raw data
title('raw data'); xlabel('t [s]'); ylabel('strain'); axis tight
dt=S.data(2,1)-S.data(1,1) % [s], (constant) sampling time
Fs=1/dt % [Hz], sampling rate

%% select "nice & clean" part

% msgbox('choose time span','modal')
figure(1); [t_user,junk]=ginput(2) % let user pick time interval
tlimits=xlim;
t_start=t_user(1)-tlimits(1)
t_end=t_user(2)-tlimits(1)
t0=S.data(round(t_start*Fs):round(t_end*Fs),1); % [s] time
e0=S.data(round(t_start*Fs):round(t_end*Fs),2); % strain
e_ss=mean(e0) % use the mean as approx of steady-state
figure(2); clf; plot(t0,e0,'.'); grid on; axis tight;
title('nice and clean part'); xlabel('t [s]'); ylabel('strain');
hold on; plot(xlim,e_ss*[1 1]);
legend('selected signal','mean')

%% subtract dc component, apply butterworth digital filter

t=t0-t0(1); % re-zero time vector
e=e0-e_ss; % subtract steady-state (DC component) approx
fc=200 % [Hz], cutoff frequency of the filter, (to remove spikes etc.)
[b,a] = butter(2,fc/(Fs/2)) % butterworth digital filter
ef=filtfilt(b,a,e); % applying filter, zero-phase
figure(3); clf;
plot(t,e,'-c', t,ef,'b.');
title('time reset, dc offset removed, low-pass filtered');
xlabel('t [s]'); ylabel('e - e_{mean}'); grid on;
legend('dc offset removed',sprintf('filtered fc=%gHz',fc));

%% visualize the "noise" component that was filtered out

ed=e-ef;
figure(4); clf;
subplot(2,1,1); plot(t,ed); grid on;
title('filtered "noise" component'); xlabel('t [s]'); ylabel('e - e_{filtered}');
subplot(2,1,2); hist(ed,100); grid on;
title('histogram of filtered "noise"'); xlabel('strain'); ylabel('occurrence')


%% identify top peaks

[pks,locs]=findpeaks(ef); % this function from signal processing toolbox
figure(5); clf; plot(t,ef,'b-', t(locs),pks,'or'); grid on
title('peak-detected'); xlabel('t [s]'); ylabel('strain');
[pks2,locs2]=findpeaks(-ef); % bottom peaks,
pks2=-pks2; hold on; plot(t(locs2),pks2,'sm');


%% period and frequency

Td=diff(t(locs)); % period of oscillation, from one top to the next top
fd=1./Td; % frequency
Td_mean=mean(Td) % average of individual periods
Td_mean2=(t(locs(end))-t(locs(1)))/(length(locs)-1) % average over cycles
fd_mean=mean(fd) % [Hz]
fd_mean2=1/Td_mean2
figure(6); clf;
subplot(2,1,1); plot( Td,'o' ); grid on;
ylabel('period [s]'); xlabel('detected peak number');
hold on; plot(xlim,Td_mean*[1 1],'r');
subplot(2,1,2); plot( fd,'o' ); grid on;
ylabel('frequency [Hz]'); xlabel('detected peak number');
hold on; plot(xlim,fd_mean*[1 1],'r');


%% logarithmic decrement

n=length(pks)-1
d=1/n*log(pks(1)/pks(end)) % logarithmic decrement, calculated from n+1 peaks
z=(1+(2*pi/d)^2)^(-0.5) % damping ratio by log decrement
wn=(2*pi*fd_mean)/sqrt(1-z^2) % [rad/s], natural freq, approx =wd when z<<1
fn=wn/2/pi % [Hz], natural frequency
figure(100); clf;
plot(0:length(pks)-1,log(pks),'o' ); hold on; grid on;
xlabel('cycles'); ylabel('ln(y_{peaks})'); title('log decrement');
plot([0 length(pks)-1],-d*[0 length(pks)-1]+log(pks(1)),'-') % connect two end points
p=polyfit(1:length(pks),log(pks)',1)
hold on; plot(0:length(pks)-1, polyval(p,1:length(pks)), '--');
d2=-p(1)
z2=(1+(2*pi/d2)^2)^(-0.5) % damping ratio by log decrement
legend('measured peaks',...
 sprintf('on end points: \\delta=%.4f, \\zeta=%.4f',d,z),...
 sprintf('curve fit: \\delta=%.4f, \\zeta=%.4f',d2,z2));





