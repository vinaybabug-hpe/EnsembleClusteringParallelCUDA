size = 10;
dirDataset = [getenv('HOME'),'/Documents/Algorithm/Matlab/mytoolbox/clustering/methods_fitting_cuda/SpectralClustering/dataset'];
fileName = [dirDataset, '/x.mat'];
X = load(fileName);
X= X.X_temp(1:size,:);
idxs = mexSpectralClustering(4, 0.1,'euc',X);



