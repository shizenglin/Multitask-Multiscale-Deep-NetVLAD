%% Experiment with the cnn_mnist_fc_bnorm
% clear all;


image_path='/home/peiyong/Work/Zenglin/caffe-szl/examples/crowd/data/ShanghaiTech/partA/test/img';
dmap_path='/home/peiyong/Work/Zenglin/caffe-szl/examples/crowd/data/ShanghaiTech/partA/test/dmap';
% 
netStruct=load('./model/net-epoch-14.mat') ;
net = dagnn.DagNN.loadobj(netStruct.net) ;
net.removeLayer('l2loss1');
net.removeLayer('l2loss2');
net.mode = 'test'; 
move(net, 'gpu');
imgnum=182;
predict=zeros(1,imgnum);
frame=zeros(1,imgnum);
patch_size=224;
for iim=1:imgnum
    iim
    imgPath=fullfile(image_path,num2str(iim+0,'IMG_%d.jpg'));
    dmapPath=fullfile(dmap_path,num2str(iim+0,'DMAP_%d.mat'));

    im=imread(imgPath);
    if size(im,3)>1
        im = rgb2gray(im);
    end
    im = single(im)./255;
    im = single(cat(3, im, im, im));
    [h, w, c] = size(im);
    load(dmapPath);
    frame(iim)=sum(sum(dmap));
     if h<224
        im=imresize(im,[226,w]);
        h=226;
    end 
    %% crop1/4 patch
    patchim=im(1:fix(h/2),1:fix(w/2),:);
    patchim=imresize(patchim,[224,224]);
    net.eval({'input',gpuArray(cat(4,patchim,patchim))}) ;
    predict(iim)=predict(iim)+sum(gather(net.vars(net.getVarIndex('relu2')).value));
    
    patchim=im(1:fix(h/2),fix(w/2)+1:w,:);
    patchim=imresize(patchim,[224,224]);
    net.eval({'input',gpuArray(cat(4,patchim,patchim))}) ;
    predict(iim)=predict(iim)+sum(gather(net.vars(net.getVarIndex('relu2')).value));
    
    patchim=im(fix(h/2)+1:h,1:fix(w/2),:);
    patchim=imresize(patchim,[224,224]);
    net.eval({'input',gpuArray(cat(4,patchim,patchim))}) ;
    predict(iim)=predict(iim)+sum(gather(net.vars(net.getVarIndex('relu2')).value));
    
    patchim=im(fix(h/2)+1:h,fix(w/2)+1:w,:);
    patchim=imresize(patchim,[224,224]);
    net.eval({'input',gpuArray(cat(4,patchim,patchim))}) ;
    predict(iim)=predict(iim)+sum(gather(net.vars(net.getVarIndex('relu2')).value));
    
%     %% uniformly crop the most area
%     for i = 1:patch_size:(h-223),
%         for j = 1:patch_size:(w-223),
%             patchim = im(i:i+223, j:j+223, :);
%             net.eval({'input',gpuArray(cat(4,patchim,patchim))}) ;
%             predict(iim)=predict(iim)+sum(gather(net.vars(net.getVarIndex('relu2')).value));
%         end
%     end
% 
% %% crop the edge area
%     for i = 1:patch_size:(h-223),
%         patchim = im(i:i+223, w-223:w, :);
%         patchim=imresize(patchim,[224,224]);
%         net.eval({'input',gpuArray(cat(4,patchim,patchim))}) ;
%         predict(iim)=predict(iim)+sum(gather(net.vars(net.getVarIndex('relu2')).value));
%     end
%     for j = 1:patch_size:(w-223),
%         patchim = im(h-223:h, j:j+223, :);
%         patchim=imresize(patchim,[224,224]);
%         net.eval({'input',gpuArray(cat(4,patchim,patchim))}) ;
%         predict(iim)=predict(iim)+sum(gather(net.vars(net.getVarIndex('relu2')).value));
%     end
% 
% %% crop the corner area
%     patchim = im(h-223:h, w-223:w, :);
%     patchim=imresize(patchim,[224,224]);
%     net.eval({'input',gpuArray(cat(4,patchim,patchim))}) ;
%     predict(iim)=predict(iim)+sum(gather(net.vars(net.getVarIndex('relu2')).value));
end

MSE=sqrt(sum(power(predict-frame,2))/imgnum)
MAE=sum(abs(predict-frame))/imgnum

