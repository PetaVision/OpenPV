#mpirun -np 4 --bind-to none Release/DepthLCA -p input/benchmark/train_np/rcorr/aws_rcorr_white_LCA.params -t 8;
#mpirun -np 4 --bind-to none Release/DepthLCA -p input/benchmark/validate_np/aws_rcorr_white_LCA.params -t 8;

Release/DepthLCA -p input/benchmark/train_np/rcorr/aws_rcorr_white_RELU.params -t 8;
Release/DepthLCA -p input/benchmark/validate_np/aws_rcorr_white_RELU.params -t 8;
