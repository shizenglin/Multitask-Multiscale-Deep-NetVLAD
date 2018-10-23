function net = cnn_crowd_init_dag_v1(varargin)
%% Loading pretrained VGGNet-16
opts.cluster1=load('./model/cluster-ucf');
opts.cluster2=load('./model/cluster-partA');

net = load('/home/peiyong/Work/Zenglin/matconvnet/examples/crowd/Pretrained-model/imagenet-vgg-verydeep-16.mat') ;
net.layers = net.layers(1:29) ;
net = vl_simplenn_tidy(net) ;
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
net.layers(24).block.stride = [1 1] ;
%net.addLayer('concat1', dagnn.Concat('dim',1), {'x22','x29'},'concat1');

%%Slice multiInput to each task
net.addLayer('vlad', dagnn.VLAD('K', 64, 'D', 512, 'vladDim', 64*512), 'x29', 'vlad', {'vladf','vladc'});
net.addLayer('intranorm', dagnn.LRN('param', [2*512, 1e-12, 1, 0.5]), 'vlad', 'intranorm');

net.addLayer('reshape', dagnn.Reshape('newsize',[1 1 512*64]), 'intranorm', 'reshape');
net.addLayer('slice', dagnn.Slice('dim',4,'num',2), 'reshape',{'slice1','slice2'});

%%Processing task1
block= dagnn.Conv('size',  [1 1 512*64 784], 'hasBias', true, ...
                   'stride', 1, 'pad', 0);
net.addLayer('fc1', block, 'slice1', 'fc1', {'fc1f', 'fc1b'});
net.addLayer('relu1', dagnn.ReLU(), 'fc1', 'relu1');
net.addLayer('l2loss1', dagnn.Loss('loss', 'l2loss'), ...
      {'fc1', 'label1'}, 'objective1') ;
  
%%Processing task2
block= dagnn.Conv('size',  [1 1 512*64 784], 'hasBias', true, ...
                   'stride', 1, 'pad', 0);
net.addLayer('fc2', block, 'slice2', 'fc2', {'fc2f', 'fc2b'});
net.addLayer('relu2', dagnn.ReLU(), 'fc2', 'relu2');
net.addLayer('l2loss2', dagnn.Loss('loss', 'l2loss'), ...
      {'fc2', 'label2'}, 'objective2') ;


%%Initializing weights  
clstsAssign= relja_l2normalize_col(opts.cluster1.clsts);
dots= sort(clstsAssign'*opts.cluster1.trainDescs, 1, 'descend'); 
alpha= -log(0.01)/mean( dots(1,:) - dots(2,:) );
net.setLayerParams('vlad',{reshape(alpha*clstsAssign, [1,1,512,64]), reshape(opts.cluster1.clsts, [1,1,512,64])});

rng('default');
rng(0) ;
f=1/100 ;
net.setLayerParams('fc1',{f*randn(1,1,512*64,784, 'single'), zeros(1,784,'single')});

net.setLayerParams('fc2',{f*randn(1,1,512*64,784, 'single'), zeros(1,784,'single')});


f = net.getParamIndex('conv5_3f') ;
net.params(f).learningRate = 10;
f = net.getParamIndex('conv5_3b') ;
net.params(f).learningRate = 20 ;
% 
f = net.getParamIndex('vladf') ;
net.params(f).learningRate = 10 ;
f = net.getParamIndex('vladc') ;
net.params(f).learningRate = 10;

f = net.getParamIndex('fc1f') ;
net.params(f).learningRate = 10;
f = net.getParamIndex('fc1b') ;
net.params(f).learningRate = 20 ;

f = net.getParamIndex('fc2f') ;
net.params(f).learningRate = 10 ;
f = net.getParamIndex('fc2b') ;
net.params(f).learningRate = 20 ;


net.meta.inputSize = [224 224 3] ;
net.meta.trainOpts.learningRate = 0.00001;
net.meta.trainOpts.numEpochs = 20 ;
net.meta.trainOpts.batchSize = 10 ;
end
