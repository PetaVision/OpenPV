more off;
setenv("GNUTERM","X11");
close all;

run_name       = "spots";
run_numbers    = 1;%[1:1:4];
num_time_steps = 2000;

num_x_steps    = num_time_steps;
begin_x_step   = 1;

legend_flag = 1;

Cone_view       = 1;
Horizontal_view = 1;
Bipolar_view    = 1;
Sigmoid_view    = 0;

for i = 1:length(run_numbers)
    run_num = i-1;

    out_path    = ['../output/',run_name,'/p',num2str(run_num),'/ns',num2str(num_time_steps),'/figs/'];
    mkdir(out_path);

    if Cone_view
        [time, C_G_GAP, C_G_I, C_V]           = pvp_readLIFGapptProbe(run_num,num_time_steps,run_name,'Cone',{'G_GAP','G_I','V'});

        figure();
        plot(time,C_G_GAP,time,C_V,time,C_G_I*40-20); grid;
        xlabel('time (ms)')
        ylabel('mV')
        if legend_flag
            legend('G\_GAP','V','G\_I')
        end
        axis([begin_x_step num_x_steps -70 70]);
        titlestring = ['Cone Response\_p',num2str(run_num)];
        title(titlestring,"fontsize",15);
        print([out_path,'ConeResponse.pdf'],'-dpdf');
        print([out_path,'ConeResponse.jpg'],'-djpg');

        if Sigmoid_view
            figure();
            plot(C_V,C_G_E*2);grid;
            xlabel('C_V')
            ylabel('C_G_E*2')
            titlestring = ['Cone Sigmoid\_p',num2str(run_num)];
            title(titlestring,"fontsize",15);
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
        axis([begin_x_step num_x_steps -70 70]);
        titlestring = ['Horizontal Response\_p',num2str(run_num)];
        title(titlestring,"fontsize",15);
        print([out_path,'HorizontalResponse.pdf'],"-dpdf");
        print([out_path,'HorizontalResponse.jpg'],"-djpg");

        if Sigmoid_view
            figure();
            plot(H_V,H_G_I*2); grid;% have to take the connection strength out
            xlabel('H_V')
            ylabel('H_G_I*2')
            titlestring = ['Horizontal Sigmoid\_p',num2str(run_num)];
            title(titlestring,"fontsize",15);
        end
    end

    if Bipolar_view
        [time, B_G_E, B_G_I, B_V]       = pvp_readLIFptProbe(run_num,num_time_steps,run_name,'Bipolar',{'G_E','G_I','V'});
        figure();
        plot(time,B_G_E*20-20,time,B_V,time,B_G_I*40-20); grid;
        xlabel('time (ms)')
        ylabel('mV')
        if legend_flag
            legend('G\_E','V','G\_I')
        end
        axis([begin_x_step num_x_steps -70 70]);
        titlestring = ['Bipolar Response\_p',num2str(run_num)];
        title(titlestring,"fontsize",15);
        print([out_path,'BipolarResponse.pdf'],"-dpdf");
        print([out_path,'BipolarResponse.jpg'],"-djpg");

         if Sigmoid_view
            figure();
            plot(B_V,B_G_E*1);grid;
            xlabel('B_V')
            ylabel('B_G_E')
            titlestring = ['Bipolar Sigmoid\_p',num2str(run_num)];
            title(titlestring,"fontsize",15);

        end
    end
end
