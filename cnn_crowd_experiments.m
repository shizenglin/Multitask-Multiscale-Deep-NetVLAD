%% Experiment with the cnn_mnist_fc_bnorm
%clear all;
diary log2.txt;

%[net, info] = cnn_crowd('expDir','./model');
image_path='/home/peiyong/Work/Zenglin/caffe-szl/examples/crowd/data/ucf-cc-50/img';
dmap_path='/home/peiyong/Work/Zenglin/caffe-szl/examples/crowd/data/ucf-cc-50/dmap';
% 
netStruct=load('./model/4/net-epoch-10.mat') ;
net = dagnn.DagNN.loadobj(netStruct.net) ;
net.removeLayer('l2loss1');
net.removeLayer('l2loss2');
net.mode = 'test'; 
move(net, 'gpu');

predict=zeros(1,10);
frame=zeros(1,10);
patch_size=224;
for iim=1:10
    iim
    imgPath=fullfile(image_path,num2str(iim,'%d.jpg'));
    dmapPath=fullfile(dmap_path,num2str(iim,'%d.mat'));

    im = single(imread(imgPath))./255;
    im = single(cat(3, im, im, im));
    [h,w,c]=size(im);
    load(dmapPath);
    frame(iim)=sum(sum(dmap));
       
%     %% crop1/4 patch
%     patchim=im(1:h/2,1:w/2,:);
%     patchim=imresize(patchim,[patch_size,patch_size]);
%     net.eval({'input',gpuArray(cat(4,patchim,patchim))}) ;
%     predict(iim)=predict(iim)+sum(gather(net.vars(net.getVarIndex('relu1')).value));
%     
%     patchim=im(1:h/2,w/2+1:w,:);
%     patchim=imresize(patchim,[patch_size,patch_size]);
%     net.eval({'input',gpuArray(cat(4,patchim,patchim))}) ;
%     predict(iim)=predict(iim)+sum(gather(net.vars(net.getVarIndex('relu1')).value));
%     
%     patchim=im(h/2+1:h,1:w/2,:);
%     patchim=imresize(patchim,[patch_size,patch_size]);
%     net.eval({'input',gpuArray(cat(4,patchim,patchim))}) ;
%     predict(iim)=predict(iim)+sum(gather(net.vars(net.getVarIndex('relu1')).value));
%     
%     patchim=im(h/2+1:h,w/2+1:w,:);
%     patchim=imresize(patchim,[patch_size,patch_size]);
%     net.eval({'input',gpuArray(cat(4,patchim,patchim))}) ;
%     predict(iim)=predict(iim)+sum(gather(net.vars(net.getVarIndex('relu1')).value));
    
    %% uniformly crop the most area
    for i = 1:patch_size:(h-223),
        for j = 1:patch_size:(w-223),
            patchim = im(i:i+223, j:j+223, :);
            net.eval({'input',gpuArray(cat(4,patchim,patchim))}) ;
            predict(iim)=predict(iim)+sum(gather(net.vars(net.getVarIndex('relu1')).value));
        end
    end

%% crop the edge area
    for i = 1:patch_size:(h-223),
        patchim = im(i:i+223, w-223:w, :);
        patchim=imresize(patchim,[patch_size,patch_size]);
        net.eval({'input',gpuArray(cat(4,patchim,patchim))}) ;
        predict(iim)=predict(iim)+sum(gather(net.vars(net.getVarIndex('relu1')).value));
    end
    for j = 1:patch_size:(w-223),
        patchim = im(h-223:h, j:j+223, :);
        patchim=imresize(patchim,[patch_size,patch_size]);
        net.eval({'input',gpuArray(cat(4,patchim,patchim))}) ;
        predict(iim)=predict(iim)+sum(gather(net.vars(net.getVarIndex('relu1')).value));
    end

%% crop the corner area
    patchim = im(h-223:h, w-223:w, :);
    patchim=imresize(patchim,[patch_size,patch_size]);
    net.eval({'input',gpuArray(cat(4,patchim,patchim))}) ;
    predict(iim)=predict(iim)+sum(gather(net.vars(net.getVarIndex('relu1')).value));
end

MAE=sum(abs(predict-frame))/10
MSE=sqrt(sum(power(predict-frame,2))/10)