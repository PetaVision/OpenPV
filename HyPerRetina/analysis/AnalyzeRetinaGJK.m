%%%%%
%%%%%
%%%%%
more off;
setenv("GNUTERM","X11");
close all;

%%run_name       = "RepositoryVersion";
run_name       = "biggraywhiteblackspots"
run_numbers    = 1;%[1:1:4];
num_time_steps = 3000;
%%num_x_steps    = num_time_steps;
num_x_steps    = 3000;
begin_x_step   = 000;

legend_flag = 1;

ImageBuffer_view= 0; %% not implemented
Cone_view       = 1;
Sigmoid_view    = 1;
Horizontal_view = 1;
Bipolar_view    = 1;
WFAmacrine_view = 1;
SFAmacrine_view = 1;
PAAmacrine_view = 1;
Ganglion_view   = 1;

for i = 1:length(run_numbers)
    run_num = i-1;

    out_path    = ['../output/',run_name,'/p',num2str(run_num),'/ns',num2str(num_time_steps),'/figs/'];
    mkdir(out_path);

    if Cone_view
        [time, C_G_GAP, C_G_I, C_V]           = pvp_readLIFGapptProbe(run_num,num_time_steps,run_name,'Cone',{'G_GAP','G_I','V'});
         
        figure();
        plot(time,C_G_GAP,time,C_V,time,C_G_I*40-20); grid;
        C_V(1000)
        C_V(1750)
        C_V(2500)
        xlabel('time (ms)')
        ylabel('mV')
        if legend_flag
            legend('G\_GAP','V','G\_I')
        end
        axis([begin_x_step num_x_steps -70 10]);
        titlestring = ['Cone Response\_p',num2str(run_num)];
        title(titlestring,"fontsize",15);
        print([out_path,'ConeResponse.pdf'],'-dpdf');
        print([out_path,'ConeResponse.jpg'],'-djpg');
        min(C_V(100:num_x_steps))
        max(C_V(100:num_x_steps))

        if Sigmoid_view
            [time, Bon_G_E, Bon_G_I, Bon_V]       = pvp_readLIFptProbe(run_num,num_time_steps,run_name,'BipolarON',{'G_E','G_I','V'});
            [time, Boff_G_E, Boff_G_I, Boff_V]    = pvp_readLIFptProbe(run_num,num_time_steps,run_name,'BipolarOFF',{'G_E','G_I','V'});
            figure();
            plot(C_V,Bon_G_E*2,C_V,Boff_G_E*2);grid;
            xlabel('C_V')
            ylabel('C_G_E')
            titlestring = ['Cone Sigmoid\_p',num2str(run_num)];
            title(titlestring,"fontsize",15);
            print([out_path,'ConeSigmoidResponse.pdf'],'-dpdf');
            print([out_path,'ConeSigmoidResponse.jpg'],'-djpg');
        end
    end

    if Horizontal_view
        [time, H_G_E, H_G_I, H_V]             = pvp_readLIFGapptProbe(run_num,num_time_steps,run_name,'Horizontal',{'G_E','G_I','V'});

        figure();
        plot(time,H_G_E*20-20,time,H_V,time,H_G_I*40-20); grid;
        xlabel('time (ms)')
        ylabel('mV')
        if legend_flag
            legend('G\_E','V','G\_I')
        end
        axis([begin_x_step num_x_steps -70 10]);
        titlestring = ['Horizontal Response\_p',num2str(run_num)];
        title(titlestring,"fontsize",15);
        print([out_path,'HorizontalResponse.pdf'],"-dpdf");
        print([out_path,'HorizontalResponse.jpg'],"-djpg");
        min(H_V(100:num_x_steps))
        max(H_V(100:num_x_steps))

        if Sigmoid_view
            figure();
            plot(H_V,C_G_I*2); grid;% have to take the connection strength out
            xlabel('H_V')
            ylabel('C_G_I*2')
            titlestring = ['Horizontal Sigmoid\_p',num2str(run_num)];
            title(titlestring,"fontsize",15);
        end
    end

    if Bipolar_view
        [time, Bon_G_E, Bon_G_I, Bon_V]       = pvp_readLIFptProbe(run_num,num_time_steps,run_name,'BipolarON',{'G_E','G_I','V'});
        [time, Boff_G_E, Boff_G_I, Boff_V]    = pvp_readLIFptProbe(run_num,num_time_steps,run_name,'BipolarOFF',{'G_E','G_I','V'});
        figure();
        plot(time,Bon_G_E*20-20,time,Bon_V,time,Bon_G_I*40-20); grid;
        xlabel('time (ms)')
        ylabel('mV')
        if legend_flag
            legend('G\_E','V','G\_I')
        end
        axis([begin_x_step num_x_steps -70 10]);
        titlestring = ['BipolarON Response\_p',num2str(run_num)];
        title(titlestring,"fontsize",15);
        print([out_path,'BipolarONResponse.pdf'],"-dpdf");
        print([out_path,'BipolarONResponse.jpg'],"-djpg");

        figure();
        plot(time,Boff_G_E*20-20,time,Boff_V,time,Boff_G_I*40-20); grid;
        xlabel('time (ms)')
        ylabel('mV')
        if legend_flag
            legend('G\_E','V','G\_I')
        end
        axis([begin_x_step num_x_steps -70 10]);
        titlestring = ['BipolarOFF Response\_p',num2str(run_num)];
        title(titlestring,"fontsize",15);
        print([out_path,'BipolarOFFResponse.pdf'],"-dpdf");
        print([out_path,'BipolarOFFResponse.jpg'],"-djpg");

        if Sigmoid_view
            figure();
            [time, Gon_G_E, Gon_G_I, Gon_V,Gon_A]       = pvp_readLIFptProbe(run_num,num_time_steps,run_name,'GanglionON',{'G_E','G_I','V','A'});
  
            plot(Bon_V,Gon_G_E*1);grid;
            xlabel('Bon_V')
            ylabel('Gon_G_E')
            titlestring = ['BipolarON Sigmoid\_p',num2str(run_num)];
            title(titlestring,"fontsize",15);
            
            figure();
            [time, Gon_G_E, Goff_G_I, Goff_V,Goff_A]       = pvp_readLIFptProbe(run_num,num_time_steps,run_name,'GanglionOFF',{'G_E','G_I','V','A'});
  
            plot(Boff_V,Goff_G_E*1);grid;
            xlabel('Boff_V')
            ylabel('Goff_G_E')
            titlestring = ['BipolarOFF Sigmoid\_p',num2str(run_num)];
            title(titlestring,"fontsize",15);
        end
    end

    if WFAmacrine_view
        [time, WFon_G_E, WFon_G_I, WFon_V]    = pvp_readLIFptProbe(run_num,num_time_steps,run_name,'WFAmacrineON',{'G_E','G_I','V'});
        [time, WFoff_G_E, WFoff_G_I, WFoff_V] = pvp_readLIFptProbe(run_num,num_time_steps,run_name,'WFAmacrineOFF',{'G_E','G_I','V'});

        figure();
        plot(time,WFon_G_E*20-20,time,WFon_V,time,WFon_G_I*40-20);grid;
        xlabel('time')
        ylabel('mV')
        if legend_flag
            legend('G\_E','V','G\_I')
        end
        titlestring = ['WFAmacrineON\_p',num2str(run_num)];
        title(titlestring,"fontsize",15);
        print([out_path,'WFAmacrineONResponse.pdf'],"-dpdf");
        print([out_path,'WFAmacrineONResponse.jpg'],"-djpg");

        figure();
        plot(time,WFoff_G_E*20-20,time,WFoff_V,time,WFoff_G_I*40-20);grid;
        xlabel('time')
        ylabel('mV')
        if legend_flag
            legend('G\_E','V','G\_I')
        end
        titlestring = ['WFAmacrineOFF\_p',num2str(run_num)];
        title(titlestring,"fontsize",15);
        print([out_path,'WFAmacrineOFFResponse.pdf'],"-dpdf");
        print([out_path,'WFAmacrineOFFResponse.jpg'],"-djpg");
    end

    if SFAmacrine_view
        [time, SF_G_E, SF_G_I, SF_V]          = pvp_readLIFptProbe(run_num,num_time_steps,run_name,'SFAmacrine',{'G_E','G_I','V'});

        figure();
        plot(time,SF_G_E*20-20,time,SF_V,time,SF_G_I*40-20);grid;
        xlabel('time')
        ylabel('mV')
        if legend_flag
            legend('G\_E','V','G\_I')
        end
        titlestring = ['SFAmacrine\_p',num2str(run_num)];
        title(titlestring,"fontsize",15);
        print([out_path,'SFAmacrineResponse.pdf'],"-dpdf");
        print([out_path,'SFAmacrineResponse.jpg'],"-djpg");
    end

    if PAAmacrine_view
        [time, PAon_G_E, PAon_G_I, PAon_V]    = pvp_readLIFptProbe(run_num,num_time_steps,run_name,'PAAmacrineON',{'G_E','G_I','V'});
        [time, PAoff_G_E, PAoff_G_I, PAoff_V] = pvp_readLIFptProbe(run_num,num_time_steps,run_name,'PAAmacrineOFF',{'G_E','G_I','V'});

        figure();
        plot(time,PAon_G_E*20-20,time,PAon_V,time,PAon_G_I*40-20);grid;
        xlabel('time')
        ylabel('mV')
        if legend_flag
            legend('G\_E','V','G\_I')
        end
        titlestring = ['PAAmacrineON\_p',num2str(run_num)];
        title(titlestring,"fontsize",15);
        print([out_path,'PAAmacrineONResponse.pdf'],"-dpdf");
        print([out_path,'PAAmacrineONResponse.jpg'],"-djpg");

        figure();
        plot(time,PAoff_G_E*20-20,time,PAoff_V,time,PAoff_G_I*40-20);grid;
        xlabel('time')
        ylabel('mV')
        if legend_flag
            legend('G\_E','V','G\_I')
        end
        titlestring = ['PAAmacrineOFF\_p',num2str(run_num)];
        title(titlestring,"fontsize",15);
        print([out_path,'PAAmacrineOFFResponse.pdf'],"-dpdf");
        print([out_path,'PAAmacrineOFFResponse.jpg'],"-djpg");
    end

    if Ganglion_view

        [time, Gon_G_E, Gon_G_I, Gon_V,Gon_A]       = pvp_readLIFptProbe(run_num,num_time_steps,run_name,'GanglionON',{'G_E','G_I','V','A'});
        figure();
        plot(time,Gon_G_E*20-20,time,Gon_V,time,Gon_G_I*40-20,Gon_A*25+50);grid;
        xlabel('time')
        ylabel('mV')
        if legend_flag
            legend('G\_E','V','G\_I')
        end

        Gon_f=sum(Gon_A(750:1250))*2;

        Ggray_f=sum(Gon_A(1500:2000))*2

        titlestring = ['GanglionON\_p',num2str(run_num),' with a frequency of ',num2str(Gon_f),' Hz in t=[750,1250]'];
        title(titlestring,"fontsize",15);
        print([out_path,'GanglionONResponse.pdf'],"-dpdf");
        print([out_path,'GanglionONResponse.jpg'],"-djpg");
 
        figure();
  
        [time, Goff_G_E, Goff_G_I, Goff_V,Goff_A]    = pvp_readLIFptProbe(run_num,num_time_steps,run_name,'GanglionOFF',{'G_E','G_I','V','A'});

        plot(time,Goff_G_E*20-20,time,Goff_V,time,Goff_G_I*40-20,Goff_A*25+50);grid;
        xlabel('time')
        ylabel('mV')
        if legend_flag
            legend('G\_E','V','G\_I')
        end

        Goff_f=sum(Goff_A(2250:2750))*2;
        Ggray_f=sum(Goff_A(1500:2000))*2

  


        titlestring = ['GanglionOFF\_p',num2str(run_num),' with a frequency of ',num2str(Goff_f),' Hz in t=[2250,2750]'];
        title(titlestring,"fontsize",15);
        print([out_path,'GanglionOFFResponse.pdf'],"-dpdf");
        print([out_path,'GanglionOFFResponse.jpg'],"-djpg");
    end
end
