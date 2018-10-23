function [net, info] = cnn_crowd(varargin)

opts.expDir='';
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = 1; end;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------
  
net = cnn_crowd_init_dag() ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

[net, info] = crowd_cnn_train_dag(net, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train) ;

% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x) getDagNNBatch(bopts,x) ;
end

function inputs = getDagNNBatch(opts, batch)
% --------------------------------------------------------------------
[images1,labels1] = random_gen_train_ucf(batch) ;
[images2,labels2] = random_gen_train_partA(batch);
if opts.numGpus > 0
  images1 = gpuArray(images1) ;
  images2 = gpuArray(images2) ;
end
inputs = {'input', cat(4,single(images1),single(images2)), 'label1', single(labels1), 'label2', single(labels2)} ;
end
end
