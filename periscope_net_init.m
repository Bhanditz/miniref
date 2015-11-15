function [net] = periscope_net_init(varargin)
% Modification of the refnet to make a periscope net
% adapted from matconvnet-1.0-beta14/matconvnet-1.0-beta14/examples/cnn_imagenet_init.m

opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
opts.weightInitMethod = 'gaussian' ;
opts.model = 'periscope' ;
opts.batchNormalization = false ;
opts = vl_argparse(opts, varargin) ;

% Define layers
net.normalization.imageSize = [126, 126, 3] ;
switch opts.model
  case 'refNet1'
      net = refNet1(net, opts) ;
  case 'periscope'
      net = periscopeNet(net, opts) ;
  case 'periscope2'
      net = periscope2Net(net, opts) ;
  otherwise
    error('Unknown model ''%s''', opts.model) ;
end


switch lower(opts.weightInitMethod)
  case {'xavier', 'xavierimproved'}
    net.layers{end}.weights{1} = net.layers{end}.weights{1} / 10 ;
end
net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;

net.normalization.border = 128 - net.normalization.imageSize(1:2) ;
net.normalization.interpolation = 'bicubic' ;
net.normalization.averageImage = [] ;
net.normalization.keepAspect = true ;
 


end


% --------------------------------------------------------------------
function net = add_block(net, opts, id, h, w, in, out, stride, pad, init_bias)
% --------------------------------------------------------------------
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
  name = 'fc' ;
else
  name = 'conv' ;
end
net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
                           'weights', {{init_weight(opts, h, w, in, out, 'single'), zeros(out, 1, 'single')}}, ...
                           'stride', stride, ...
                           'pad', pad, ...
                           'learningRate', [1 2], ...
                           'weightDecay', [opts.weightDecay 0]) ;
if opts.batchNormalization
  net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%d',id), ...
                             'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single')}}, ...
                             'learningRate', [2 1], ...
                             'weightDecay', [0 0]) ;
end
net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id)) ;
end
% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

switch lower(opts.weightInitMethod)
  case 'gaussian'
    sc = 0.01/opts.scale ;
    weights = randn(h, w, in, out, type)*sc;
  case 'xavier'
    sc = sqrt(3/(h*w*in)) ;
    weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
  case 'xavierimproved'
    sc = sqrt(2/(h*w*out)) ;
    weights = randn(h, w, in, out, type)*sc ;
  otherwise
    error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end

end
% --------------------------------------------------------------------
function net = add_norm(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
  net.layers{end+1} = struct('type', 'normalize', ...
                             'name', sprintf('norm%s', id), ...
                             'param', [5 1 0.0001/5 0.75]) ;
end
end
% --------------------------------------------------------------------
function net = add_dropout(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
  net.layers{end+1} = struct('type', 'dropout', ...
                             'name', sprintf('dropout%s', id), ...
                             'rate', 0.5) ;
end

end
% --------------------------------------------------------------------
function net = refNet1(net, opts)
% 3 convnet + 1 FC + 1 softmax
% --------------------------------------------------------------------
%% add_block(net, opts, id, h, w, in, out, stride, pad, init_bias)

net.layers = {} ;

net = add_block(net, opts, '1', 8, 8, 3, 64, 2, 0) ;
net = add_norm(net, opts, '1') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;


net = add_block(net, opts, '2', 5, 5, 32, 96, 1, 2) ;
net = add_norm(net, opts, '2') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '3', 3, 3, 96, 128, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool3', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '4', 6, 6, 128, 512, 1, 0) ;
net = add_dropout(net, opts, '4') ;

net = add_block(net, opts, '5', 1, 1, 512, 100, 1, 0) ;
net.layers(end) = [] ;
if opts.batchNormalization, net.layers(end) = [] ; end
end

% --------------------------------------------------------------------
function net = periscopeNet(net, opts)
% 4 convnet + 1 FC + 1 softmax
% --------------------------------------------------------------------
%% add_block(net, opts, id, h, w, in, out, stride, pad, init_bias)

net.layers = {} ;

% conv1
net = add_block(net, opts, '1', 8, 8, 3, 64, 2, 0) ;
net = add_norm(net, opts, '1') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;


% conv2
net = add_block(net, opts, '2', 5, 5, 32, 96, 1, 2) ;
net = add_norm(net, opts, '2') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

% conv3
net = add_block(net, opts, '3', 3, 3, 96, 192, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool3', ...
                           'method', 'max', ...
                           'pool', [4 4], ...
                           'stride', 3, ...
                           'pad', 0) ;

% conv4
net = add_block(net, opts, '4', 3, 3, 192, 512, 1, 0) ;
net = add_dropout(net, opts, '4') ;

% fc5
net = add_block(net, opts, '5', 2, 2, 512, 512, 1, 0) ;
net = add_dropout(net, opts, '5') ;

% fc6
net = add_block(net, opts, '6', 1, 1, 512, 100, 1, 0) ;

net.layers(end) = [] ;
if opts.batchNormalization, net.layers(end) = [] ; end
end


% --------------------------------------------------------------------
function net = periscope2Net(net, opts)
% 5 convnet + 1 FC + 1 softmax
% --------------------------------------------------------------------
%% add_block(net, opts, id, h, w, in, out, stride, pad, init_bias)

net.layers = {} ;

% conv1
net = add_block(net, opts, '1', 8, 8, 3, 64, 2, 0) ;
net = add_norm(net, opts, '1') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;


% conv2
net = add_block(net, opts, '2', 3, 3, 32, 128, 1, 2) ;
net = add_norm(net, opts, '2') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;

% conv3
net = add_block(net, opts, '3', 3, 3, 128, 256, 1, 1) ;
net = add_dropout(net, opts, '3') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool3', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

% conv4
net = add_block(net, opts, '4', 3, 3, 128, 512, 1, 0) ;
net = add_dropout(net, opts, '4') ;

% conv5
net = add_block(net, opts, '5', 3, 3, 128, 512, 1, 0) ;
net = add_dropout(net, opts, '5') ;

% conv6
net = add_block(net, opts, '6', 3, 3, 128, 512, 1, 0) ;
net = add_dropout(net, opts, '6') ;

% fc7
net = add_block(net, opts, '7', 1, 1, 512, 100, 1, 0) ;
net.layers(end) = [] ;
if opts.batchNormalization, net.layers(end) = [] ; end
end


