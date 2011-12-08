%clear all
%close all
clc


speed = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2];
wavelength = [3];
theta = [1];
sigma = [1.25];

path='/Users/kpeterson/Documents/workspace/kris/output/3DGauss/new/';
figure; hold on;
for(s=[1:length(speed)]) 
	sp=-speed(s);
	for(w=[1:length(wavelength)]) 
		wv=wavelength(w);
		for(t=[1:length(theta)]) 
		%t=3;
			th=theta(t);
			for(si=[1:length(sigma)]) 
			%si=2;
				sig=sigma(si);
                filename=['char_speed_' , num2str(sp) , '_wl_' , num2str(wv) , '_theta_' , num2str(th) , '_sigma_' , num2str(sig)]
                a=readactivities([path, filename, '/a2.pvp']);
                %figure;
                %plot(squeeze(a(128,128,1,:)), 'r')
                for(f=[1:4])
                   amp(s,w,t,si,f)=max(squeeze(a(128,128,f,[20:60])))-min(squeeze(a(128,128,f,[20:60])));
                end
                
                filenamestim=[path, filename, '/a1.pvp'];
                [wx, wt] = fftofstimuli(filenamestim);
                wxarray(s) = wx;
                wtarray(s) = wt;
                
            end
        end
    end
end

for(t=[1:length(theta)])
    %t=3;
    th=theta(t);
    for(si=[1:length(sigma)])
        %si=2;
        sig=sigma(si);
        
        %for(s=[1:4])
        %     sp=-speed(s);
%         figure; hold on;
%         set(gca,'XTick',wavelength)
%         title(['amp vs wavelength theta = ' ,num2str(th), 'sigma = ', num2str(sig)]);
%         plot(wavelength,squeeze(amp(1,:,t,si,1)),'r')
%         plot(wavelength,squeeze(amp(2,:,t,si,1)),'b')
%         plot(wavelength,squeeze(amp(3,:,t,si,1)),'g')
%         plot(wavelength,squeeze(amp(4,:,t,si,1)),'k')
%         plot(wavelength,squeeze(amp(5,:,t,si,1)),'c')
%         plot(wavelength,squeeze(amp(6,:,t,si,1)),'m')
%         legend(num2str(speed(1)), num2str(speed(2)), num2str(speed(3)), num2str(speed(4)), num2str(speed(5)), num2str(speed(6)))
        %end
        %for(w=[1:4])
        %   wv=wavelength(w);
        figure; hold on;
        set(gca,'XTick',wtarray)
        
        title(['amp vs speed theta = ' ,num2str(th), 'sigma = ', num2str(sig)]);
        plot(wtarray,squeeze(amp(:,1,t,si,1)),'r')
%         plot(speed,squeeze(amp(:,2,t,si,1)),'b')
%         plot(speed,squeeze(amp(:,3,t,si,1)),'g')
%         plot(speed,squeeze(amp(:,4,t,si,1)),'k')
%        legend(num2str(wavelength(1)), num2str(wavelength(2)), num2str(wavelength(3)), num2str(wavelength(4)))
        %end
    end
end




%speed = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5];
speed = [1];
%wavelength = [3];
wavelength = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 16];
theta = [1];
sigma = [1.25];

path='/Users/kpeterson/Documents/workspace/kris/output/3DGauss/new/';
figure; hold on;
for(s=[1:length(speed)]) 
	sp=-speed(s);
	for(w=[1:length(wavelength)]) 
		wv=wavelength(w);
		for(t=[1:length(theta)]) 
		%t=3;
			th=theta(t);
			for(si=[1:length(sigma)]) 
			%si=2;
				sig=sigma(si);
                filename=['char_speed_' , num2str(sp) , '_wl_' , num2str(wv) , '_theta_' , num2str(th) , '_sigma_' , num2str(sig)]
                a=readactivities([path, filename, '/a2.pvp']);
%                 figure;
%                 plot(squeeze(a(128,128,1,:)), 'r')
                %plot(squeeze(a(128,128,2,[20:30])), 'b')
                %plot(squeeze(a(128,128,3,[20:30])), 'g')
                %plot(squeeze(a(128,128,4,[20:30])), 'k')
                for(f=[1:4])
                   amp(s,w,t,si,f)=max(squeeze(a(128,128,f,[20:30])))-min(squeeze(a(128,128,f,[20:30])));
                end
                
            end
        end
    end
end

for(t=[1:length(theta)])
    %t=3;
    th=theta(t);
    for(si=[1:length(sigma)])
        %si=2;
        sig=sigma(si);
        
        %for(s=[1:4])
        %     sp=-speed(s);
        figure; hold on;
        set(gca,'XTick',wavelength)
        title(['amp vs wavelength theta = ' ,num2str(th), 'sigma = ', num2str(sig)]);
        plot(wavelength,squeeze(amp(1,:,t,si,1)),'r')
%         plot(wavelength,squeeze(amp(2,:,t,si,1)),'b')
%         plot(wavelength,squeeze(amp(3,:,t,si,1)),'g')
%         plot(wavelength,squeeze(amp(4,:,t,si,1)),'k')
%         plot(wavelength,squeeze(amp(5,:,t,si,1)),'c')
%         plot(wavelength,squeeze(amp(6,:,t,si,1)),'m')
%         legend(num2str(speed(1)), num2str(speed(2)), num2str(speed(3)), num2str(speed(4)), num2str(speed(5)), num2str(speed(6)))
        %end
        %for(w=[1:4])
        %   wv=wavelength(w);
%         figure; hold on;
%         set(gca,'XTick',speed)
%         
%         title(['amp vs speed theta = ' ,num2str(th), 'sigma = ', num2str(sig)]);
%         plot(speed,squeeze(amp(:,1,t,si,1)),'r')
%         plot(speed,squeeze(amp(:,2,t,si,1)),'b')
%         plot(speed,squeeze(amp(:,3,t,si,1)),'g')
%         plot(speed,squeeze(amp(:,4,t,si,1)),'k')
%        legend(num2str(wavelength(1)), num2str(wavelength(2)), num2str(wavelength(3)), num2str(wavelength(4)))
        %end
    end
end
