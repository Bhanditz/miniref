function imdb = cnn_miniplaces_setup_data(varargin)
% CNN_MINIPLACES_SETUP_DATA  Initialize MiniPlaces challenge data
%    This function creates an IMDB structure pointing to a local copy
%    of the MiniPlaces data.
%
%    Within the 'data' folder (which can be a symlink), create the following
%    hierarchy:
%
%    <DATA>/images/train/
%    <DATA>/images/val/
%    <DATA>/images/test/
%    <DATA>/delopment_kit: content of development_kit tarfile

opts.dataDir = 'data' ;
opts.lite = false ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                  Load categories metadata
% -------------------------------------------------------------------------

trainPath = fullfile(opts.dataDir, 'development_kit', 'data', 'train.txt');
valPath = fullfile(opts.dataDir, 'development_kit', 'data', 'val.txt');
catPath = fullfile(opts.dataDir, 'development_kit', 'data', 'categories.txt');

fprintf('using label file %s\n', catPath) ;
fprintf('using training file %s\n', trainPath) ;

[cats, labels] = textread(catPath, '%s %d');

imdb.classes.name = cats ;
imdb.classes.description = cats ;
imdb.imageDir = fullfile(opts.dataDir, 'images') ;

% -------------------------------------------------------------------------
%                                                           Training images
% -------------------------------------------------------------------------

[names, labels] = textread(trainPath, '%s %d');

% names = strcat(['train' filesep], names) ;

imdb.images.id = 1:numel(names) ;
imdb.images.name = names' ;
imdb.images.set = ones(1, numel(names)) ;
imdb.images.label = labels' + 1 ;

% -------------------------------------------------------------------------
%                                                         Validation images
% -------------------------------------------------------------------------
% WORKING HERE

[names, labels] = textread(valPath, '%s %d');

imdb.images.id = horzcat(imdb.images.id, (1:numel(names)) + 1e7 - 1) ;
imdb.images.name = horzcat(imdb.images.name, names') ;
imdb.images.set = horzcat(imdb.images.set, 2*ones(1,numel(names))) ;
imdb.images.label = horzcat(imdb.images.label, labels' + 1) ;

% -------------------------------------------------------------------------
%                                                               Test images
% -------------------------------------------------------------------------

ims = dir(fullfile(opts.dataDir, 'images', 'test', '*.jpg')) ;
names = sort({ims.name}) ;
labels = zeros(1, numel(names)) ;

names = strcat(['test' filesep], names) ;

imdb.images.id = horzcat(imdb.images.id, (1:numel(names)) + 2e7 - 1) ;
imdb.images.name = horzcat(imdb.images.name, names) ;
imdb.images.set = horzcat(imdb.images.set, 3*ones(1,numel(names))) ;
imdb.images.label = horzcat(imdb.images.label, labels) ;

% -------------------------------------------------------------------------
%                                                            Postprocessing
% -------------------------------------------------------------------------

% sort categories by WNID (to be compatible with other implementations)
[imdb.classes.name,perm] = sort(imdb.classes.name) ;
imdb.classes.description = imdb.classes.description(perm) ;
relabel(perm) = 1:numel(imdb.classes.name) ;
ok = imdb.images.label >  0 ;
imdb.images.label(ok) = relabel(imdb.images.label(ok)) ;

if opts.lite
  % pick a small number of images for the first 10 classes
  % this cannot be done for test as we do not have test labels
  clear keep ;
  for i=1:10
    sel = find(imdb.images.label == i) ;
    train = sel(imdb.images.set(sel) == 1) ;
    val = sel(imdb.images.set(sel) == 2) ;
    train = train(1:256) ;
    val = val(1:40) ;
    keep{i} = [train val] ;
  end
  test = find(imdb.images.set == 3) ;
  keep = sort(cat(2, keep{:}, test(1:1000))) ;
  imdb.images.id = imdb.images.id(keep) ;
  imdb.images.name = imdb.images.name(keep) ;
  imdb.images.set = imdb.images.set(keep) ;
  imdb.images.label = imdb.images.label(keep) ;
end
