%%%%%
%%%%%
%%%%%
more off;

run_name       = 'HoriGapCalibration';
run_numbers    = [0:1:4];
num_time_steps = 2000;

num_x_steps    = num_time_steps;
begin_x_step   = 100;

legend_flag = 1;

for i = 1:length(run_numbers)
    run_num = run_numbers(i);

    out_path    = ['../output/',run_name,'/p',num2str(run_num),'/ns',num2str(num_time_steps),'/figs/'];
    mkdir(out_path);

    [time, C_G_GAP, C_G_I, C_V]        = pvp_readLIFGapptProbe(run_num,num_time_steps,run_name,"Cone",{'G_GAP','G_I','V'});
    [time, H_G_E, H_G_I, H_V]          = pvp_readLIFGapptProbe(run_num,num_time_steps,run_name,"Horizontal",{'G_E','G_I','V'});
    [time, Bon_G_E, Bon_G_I, Bon_V]    = pvp_readLIFptProbe(run_num,num_time_steps,run_name,"BipolarON",{'G_E','G_I','V'});
    [time, Gon_G_E, Gon_G_I, Gon_V]    = pvp_readLIFptProbe(run_num,num_time_steps,run_name,"GanglionON",{'G_E','G_I','V'});
    [time, Boff_G_E, Boff_G_I, Boff_V] = pvp_readLIFptProbe(run_num,num_time_steps,run_name,"BipolarOFF",{'G_E','G_I','V'});
    [time, Goff_G_E, Goff_G_I, Goff_V] = pvp_readLIFptProbe(run_num,num_time_steps,run_name,"GanglionOFF",{'G_E','G_I','V'});
     
    %figure());
    %plot(time,C_G_GAP,time,C_V,time,C_G_I*40-20); grid;
    %xlabel('time (ms)')
    %ylabel('mV')
    %if legend_flag
    %    legend('C\_G\_GAP','C\_V','C\_G\_I*40-20')
    %end
    %%%plot(time,C_G_GAP); grid;
    %axis([begin_x_step num_x_steps -70 10]);
    %titlestring = ['Cone Response\_p',num2str(run_num)];
    %title(titlestring,"fontsize",15);
    %print([out_path,'ConeResponse.pdf'],"-dpdf");
    %print([out_path,'ConeResponse.jpg'],"-djpg");

    figure();
    plot(time,H_G_E*20-20,time,H_V,time,H_G_I*40-20); grid;
    xlabel('time (ms)')
    ylabel('mV')
    if legend_flag
        legend('H\_G\_E*20-20','H\_V','H\_G\_I*40-20')
    end
    axis([begin_x_step num_x_steps -70 10]);
    titlestring = ['Horizontal Response\_p',num2str(run_num)];
    title(titlestring,"fontsize",15);
    print([out_path,'HorizontalResponse.pdf'],"-dpdf");
    print([out_path,'HorizontalResponse.jpg'],"-djpg");

    %figure();
    %plot(time,Bon_G_E*20-20,time,Bon_V,time,Bon_G_I*40-20); grid;
    %xlabel('time (ms)')
    %ylabel('mV')
    %if legend_flag
    %    legend('Bon\_G\_E*20-20','Bon\_V','Bon\_G\_I*40-20')
    %end
    %axis([begin_x_step num_x_steps -70 10]);
    %titlestring = ['BipolarON Response\_p',num2str(run_num)];
    %title(titlestring,"fontsize",15);
    %print([out_path,'BipolarONResponse.pdf'],"-dpdf");
    %print([out_path,'BipolarONResponse.jpg'],"-djpg");

    %figure();
    %plot(time,Boff_G_E*20-20,time,Boff_V,time,Boff_G_I*40-20); grid;
    %xlabel('time (ms)')
    %ylabel('mV')
    %if legend_flag
    %    legend('Boff\_G\_E*20-20','Boff\_V','Boff\_G\_I*40-20')
    %end
    %axis([begin_x_step num_x_steps -70 10]);
    %titlestring = ['BipolarOFF Response\_p',num2str(run_num)];
    %title(titlestring,"fontsize",15);
    %print([out_path,'BipolarOFFResponse.pdf'],"-dpdf");
    %print([out_path,'BipolarOFFResponse.jpg'],"-djpg");

    %figure();
    %plot(C_V,H_G_E*2);grid;
    %xlabel('C_V')
    %ylabel('H_G_E*2')
    %titlestring = ['Cone Sigmoid\_p',num2str(run_num)];
    %title(titlestring,"fontsize",15);
    %
    %figure();
    %plot(H_V,C_G_I*2); grid;% have to take the connection strength out
    %xlabel('H_V')
    %ylabel('C_G_I*2')
    %titlestring = ['Horizontal Sigmoid\_p',num2str(run_num)];
    %title(titlestring,"fontsize",15);
    %
    %figure();
    %plot(Bon_V,Gon_G_E*1);grid;
    %xlabel('Bon_V')
    %ylabel('Gon_G_E')
    %titlestring = ['BipolarON Sigmoid\_p',num2str(run_num)];
    %title(titlestring,"fontsize",15);
    %
    %figure();
    %plot(Boff_V,Goff_G_E*1);grid;
    %xlabel('Boff_V')
    %ylabel('Goff_G_E')
    %titlestring = ['BipolarOFF Sigmoid\_p',num2str(run_num)];
    %title(titlestring,"fontsize",15);
    %
    %figure();
    %plot(time,Gon_V);grid;
    %xlabel('time')
    %ylabel('Gon_V')
    %titlestring = ['GanglionON Vmem\_p',num2str(run_num)];
    %title(titlestring,"fontsize",15);
    %print([out_path,'GanglionONResponse.pdf'],"-dpdf");
    %print([out_path,'GanglionONResponse.jpg'],"-djpg");
    %
    %figure();
    %plot(time,Goff_V);grid;
    %xlabel('time')
    %ylabel('Goff_V')
    %titlestring = ['GanglionOFF Vmem\_p',num2str(run_num)];
    %title(titlestring,"fontsize",15);
    %print([out_path,'GanglionOFFResponse.pdf'],"-dpdf");
    %print([out_path,'GanglionOFFResponse.jpg'],"-djpg");
     

    disp(['C_V(90) = ',num2str(C_V(90))])
    disp(['C_V(190) = ',num2str(C_V(190))])
    Vrest = -55;
    ratio = (C_V(190)-C_V(90))/(C_V(90)-Vrest)

    ON  = Bon_V(90)
    OFF = Boff_V(90)
    ONOFF =  Bon_V(90)/Boff_V(90)
end
