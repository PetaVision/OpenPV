function [ffile, vfile] = pv_globals(layer)

    global NO

%     if ispc       % Windows
%         output_path = '..\src\output\';
%         input_path = '..\src\io\input\circle1_';
%     else
%        output_path = [input_dir,'/src/output/'];
%        input_path = [input_dir,'/src/io/input/test64_'];
%     end

    % Read parameters from file which pv created
%     load([output_path, 'params.txt'],'-ascii')
%     NX = params(1);
%     NY = params(2);
%     NO = params(3);
%     NK = params(4);
%     N = params(5);
%     DTH  = params(6);
%    n_time_steps = params(7);

    fifile = 'nofile';
    vifile = 'nofile';
	if (layer == 1) %retina
		ffile = 'f0_sparse.bin';
		vfile = 'V0.bin';
        NO = 1;
	elseif (layer == 2 ) %V1
		ffile = 'f1_sparse.bin';
		vfile = 'V1.bin';
        NO = 12;
	elseif (layer == 3) %V1 Inhib
		ffile = 'f2_sparse.bin';
		vfile = 'V2.bin';
        NO = 12;
	end
	
end
