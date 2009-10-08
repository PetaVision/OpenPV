function W_array = stdp_plotWeights(fname)
% plot "weights" (typically after turning on just one neuron)
global input_dir n_time_steps NK NO NX NY DTH 

filename = fname;
filename = [input_dir, filename];

NXP = 3;
NYP = 3;
patch_size = NXP*NYP;


fid=fopen(filename,'r');
W_array = fscanf(fid,'%g %g %g %g %g %g %g %g %g',[patch_size inf]);
W_array= W_array';
fclose(fid);


%figure;
%axis([0 NX*NXP  0 NY*NYP]);

%W_array(1,:)
T=size(W_array,1)/(NX*NY);
%pause
W_max = 10;  
%colormap('gray');
colormap(jet);
b_color = 1;     % use to scale weights to the full range
                 % of the color map
a_color = (length(get(gcf,'Colormap'))-1.0)/W_max;

 
    
a= (NXP-1)/2;    % NXP = 2a+1;
b= (NYP-1)/2;    % NYP = 2b+1;

if 0 % version with no boundary around patch
    
    n=0;
    for t=1:T
        fprintf('t = %d\n',t);
        A = [];
        for i=((NXP+1)/2):NYP:(NY*NYP)
            for j=((NXP+1)/2):NXP:(NX*NXP)
                n=n+1;
                patch = reshape(W_array(n,:),[NXP NYP]);
                A(i-a:i+a,j-b:j+b) = patch;
            end
        end
        
        if t==1
            Ainit = A;
            Aold = A;
        else
            
            %imagesc(A-Ainit,'CDataMapping','direct');
            imagesc(a_color*A+b_color);
            %When CDataMapping is direct, the values of
            %CData should be in the range 1 to length(get(gcf,'Colormap'))
            
            %imagesc(A-Aold);
            
            pause(0.1)
            %Aold = A;
        end
    end
    
end % version with no boundary around patch


if 1 % version with  boundary around patch
    NXPold = NXP;
    NYPold = NYP;
    NXP = NXP+2;
    NYP = NYP+2;
    a1= (NXP-1)/2;    % NXP = 2a+1;
    b1= (NYP-1)/2;    % NYP = 2b+1;
    n=0;
    PATCH = zeros(NXP,NYP);
    %PATCH = repmat(...,NXP,NXP);
    
            
    for t=1:T
        fprintf('t = %d\n',t);
        A = [];
        for i=((NXP+1)/2):NYP:(NY*NYP)
            for j=((NXP+1)/2):NXP:(NX*NXP)
                n=n+1;
                patch = reshape(W_array(n,:),[NXPold NYPold]);
                PATCH(a1+1-a:a1+1+a,b1+1-b:b1+1+b) = patch;
                A(i-a1:i+a1,j-b1:j+b1) = PATCH;
            end
        end
        
        if t==1
            Ainit = A;
            Aold = A;
        else
            
            %imagesc(A-Ainit,'CDataMapping','direct');
            %imagesc(A,'CDataMapping','direct');
            imagesc(a_color*A+b_color);
            axis square
            axis off
            %When CDataMapping is direct, the values of
            %CData should be in the range 1 to length(get(gcf,'Colormap'))
            %imagesc(A-Aold);
            pause(0.1)
            %Aold = A;
        end
    end
    
end % version with boundary around patch