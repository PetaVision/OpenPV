function [ffile, vfile] = pv_globals(layer)

    global N NK NO NX NY DTH n_time_steps output_path input_path
    global input_dir

    if ispc       % Windows
        output_path = '..\src\output\';
        input_path = '..\src\io\input\circle1_';
    else
       output_path = [input_dir,'/src/output/'];
       input_path = [input_dir,'/src/io/input/test64_'];
    end

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
	if (layer == 0) %retina
		ffile = 'f0_sparse.bin';
		vfile = 'V0.bin';
		N = N/(NK*NO);
		NK=1;
		NO=1;
	elseif (layer == 1 ) %V1Simple
		ffile = 'f1_sparse.bin';
		vfile = 'V1.bin';
		N = N/(NK);
		NK=1;
	elseif (layer == 2) %V1SimpleInhib
		ffile = 'f2_sparse.bin';
		vfile = 'V2.bin';
		N = N/(NK);
		NK=1;
	elseif (layer == 3) %V1SurroundInhib
		ffile = 'f3_sparse.bin';
		vfile = 'V3.bin';
		N = N/(NK * NO);
		NK=1;
        NO=1;
	elseif (layer == 4 ) %V1FeedbackInhib
		ffile = 'f4_sparse.bin';
		vfile = 'V4.bin';
% 		N = N/(NK);
% 		NK=1;
	elseif (layer == 5) %V1SimpleII
		ffile = 'f5_sparse.bin';
		vfile = 'V5.bin';
	elseif (layer == 6) %V1FeedbackIIInhib
		ffile = 'f6_sparse.bin';
		vfile = 'V6.bin';
% 		N = N/(NK);
% 		NK=1;
	end
	
end
