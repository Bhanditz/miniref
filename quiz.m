%QUIZ visualizes confusion matrix for a pre-trained model

run('matconvnet/matlab/vl_setupnn');

opts.dataDir = 'data';
opts.expDir = 'exp';
opts.modelType = 'experiment12';

trained = load('exp/miniplaces-experiment12-simplenn/net-epoch-60.mat');
net = trained.net;
net.layers{end}.type = 'softmax';
imdb = cnn_miniplaces_setup_data('dataDir', opts.dataDir);
val = find(imdb.images.set == 2);
image = imdb.images.name(val);
label = imdb.images.label(val);
labelname = imdb.classes.name;
avg = repmat(permute(net.normalization.averageImage, [3,2,1]), ...
             net.normalization.imageSize(1:2));

% load and preprocess an image
for fi = 1:30
    name = image{fi};
    fn = fullfile(opts.dataDir, 'images', name);
    im = imread(fn);
    im_ = single(im) ; % note: 0-255 range
    im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
    im_ = im_ - avg;

    % run the CNN
    res = vl_simplenn(net, im_, [], [], 'disableDropout', 1, 'cudnn', 1) ;

    % show the classification result
    scores = squeeze(gather(res(end).x)) ;
    [bestScore, best] = sort(scores, 'descend') ;
    if find(best == label(fi)) <= 5
        disp(sprintf('%s is classified correctly as %s', name, labelname{label(fi)}'));
        continue
    end
    disp(sprintf('%s is MISSED as %s', name, labelname{label(fi)}));

    titlelist = {sprintf('%s acutal %s (%.3f)', name, labelname{label(fi)}, ...
           bestScore(find(best == label(fi))))};
    for ind = 1:5
        info = sprintf('%s (%d), score %.3f',...
          labelname{best(ind)}, best(ind), bestScore(ind)) ;
        disp(info);
        titlelist{end + 1} = info ;
    end
    
    figure(fi) ; clf ; imagesc(im) ; t = title(titlelist);
    set(t,'Interpreter','none');
end
