#!/usr/bin/env python3

train = True # True or False, default: True, False loads existing parameters

import joblib
import matplotlib.pyplot as plt
import numpy as np
#
from sklearn import svm
from sklearn.decomposition import PCA
#
import mnist_io

if __name__ == '__main__':
    ###### TRAINING ######
    if train:
        print('###### Training SVM ######')
    else:
        print('###### Loading SVM parameters ######')

    # SETUP
    imgs_train = mnist_io.load_images(mnist_io.file_imgs_train, mnist_io.nImgs_train)
    imgs_train = np.reshape(imgs_train, (np.size(imgs_train, 0), int(mnist_io.img_size**2))) # flatten images for SVM
    imgs_train_mean = np.mean(imgs_train)
    imgs_train_std = np.std(imgs_train)
    imgs_train = imgs_train - imgs_train_mean # normalize such that E(imgs) = 0
    imgs_train = imgs_train / imgs_train_std # normalize such that Var(imgs) = 1
    labels_train = mnist_io.load_labels(mnist_io.file_labels_train, mnist_io.nImgs_train)

    # TRAIN/SAVE OR LOAD
    if train:
        # SETUP
        clf = svm.SVC(C=1.0,
                      kernel='rbf',
                      degree=3, # ignored for 'rbf'
                      gamma='scale',
                      coef0=0.0,
                      shrinking=True,
                      probability=True, # to obtain probability values when plotting PCA
                      tol=1e-3,
                      cache_size=int(4e3), # default: 200
                      class_weight=None,
                      verbose=False, # deactivate for speed
                      max_iter=int(-1), # default: -1 (equal to no limit)
                      decision_function_shape='ovr',
                      break_ties=False,
                      random_state=1) # to make output reproducible
#         clf = svm.NuSVC(nu=0.5, # interpretable parameter compared to C
#                         kernel='rbf',
#                         degree=3, # ignored for 'rbf'
#                         gamma='scale',
#                         coef0=0.0,
#                         shrinking=True,
#                         probability=True, # to obtain probability values when plotting PCA
#                         tol=0.001,
#                         cache_size=int(4e3), # default: 200
#                         class_weight=None,
#                         verbose=False, # deactivate for speed
#                         max_iter=-1, # default: -1 (equal to no limit)
#                         decision_function_shape='ovr',
#                         break_ties=False,
#                         random_state=None)
    
        # TRAIN
        clf.fit(imgs_train, labels_train)
        # SAVE
        joblib.dump(clf, 'svm_trained.pkl')
    else:
        clf = joblib.load('svm_trained.pkl')
        print('Finished loading SVM parameters')
        print()
    
    # PRINT
    if train:
        if not(clf.fit_status_):
            print('Training succesful: True')
        else:
            print('Training succesful: False')
        print('Finished training SVM')
        print()
    
    ###### ANALYSIS ######
    print('###### Analyzing SVM ######')
    
    # SETUP
    imgs_test = mnist_io.load_images(mnist_io.file_imgs_test, mnist_io.nImgs_test)
    imgs_test = np.reshape(imgs_test, (np.size(imgs_test, 0), int(mnist_io.img_size**2))) # flatten images for SVM
    imgs_test = imgs_test - imgs_train_mean # normalize such that E(imgs) = 0
    imgs_test = imgs_test / imgs_train_std # normalize such that Var(imgs) = 1
    labels_test = mnist_io.load_labels(mnist_io.file_labels_test, mnist_io.nImgs_test)

    # PREDICT
    labels_predict = clf.predict(imgs_test)
    p = np.sum(labels_predict == labels_test, dtype=float) / float(mnist_io.nImgs_test)
    
    # PRINT
    print('Classification accuracy on test data:')
    print('{:0.4f}'.format(p))
    print()
   
    # PCA
    skip_factor = 10
    nPC = 3
    pca = PCA(n_components=nPC,
              copy=True,
              whiten=False,
              svd_solver='auto',
              tol=0.0,
              iterated_power='auto')
    z_pca = pca.fit_transform(imgs_test[::skip_factor])
    
    print('Variance explained by PCs:')
    for i in range(nPC):
        print('PC{:02d}:\t{:0.4f}'.format(i+1, pca.explained_variance_ratio_[i]))
    print('total:\t{:0.4f}'.format(sum(pca.explained_variance_ratio_)))
    print()
    
    nMesh = 100
    i_fig = 1
    for i in range(nPC-1):
        for j in range(i+1, nPC):
            min_pc1 = np.min(z_pca[:, i])
            max_pc1 = np.max(z_pca[:, i])
            min_pc2 = np.min(z_pca[:, j])
            max_pc2 = np.max(z_pca[:, j])
            xx_pc, yy_pc = np.meshgrid(np.linspace(min_pc1, max_pc1, nMesh),
                                       np.linspace(min_pc2, max_pc2, nMesh)) # generate mesh in PC space
            xy_pc = np.zeros((int(nMesh**2), nPC), dtype=np.float64)
            xy_pc[:, i] = xx_pc.ravel()
            xy_pc[:, j] = yy_pc.ravel()
            xy = pca.inverse_transform(xy_pc) # map mesh to image space
            log_prob = clf.predict_log_proba(xy).reshape(nMesh, nMesh, 10) # evaluate probability of mesh coordinates
            min_log_prob = np.min(log_prob)
            max_log_prob = np.max(log_prob)
            
            # PLOT
            fig = plt.figure(i_fig, figsize=(12, 6))
            i_fig += 1
            fig.clear()
            for k in range(10):
                ax = fig.add_subplot(2, 5, k+1)
                ax.clear()
                ax.axis('off')
                ax.imshow(log_prob[:, :, k],
                          interpolation='none',
                          extent=(min_pc1, max_pc1, min_pc2, max_pc2),
                          vmin=min_log_prob,
                          vmax=max_log_prob,
                          aspect='auto',
                          origin='lower',
                          cmap=plt.cm.viridis)
                mask = (labels_test[::skip_factor] == k)
                z_plot = z_pca[mask]
                ax.plot(z_plot[:, i], z_plot[:, j],
                        color='black', alpha=0.5, linestyle='', marker='.')
                ax.set_title('#{:01d}'.format(k))
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#                 if (k > 4):
#                     ax.set_xlabel('PC{:01d}'.format(i+1))
#                 if ((k == 0) or (k == 5)):
#                     ax.set_ylabel('PC{:01d}'.format(j+1))
            fig.suptitle('PC{:01d} vs. PC{:01d}'.format(i+1, j+1),
                         x=0.5, y=0.98,
                         va='center', ha='center')
            fig.canvas.draw()
            fig.tight_layout()
            fig.canvas.draw()
            plt.show(block=False)
            plt.pause(1e-2) # to make sure figures are actually displayed
            # PRINT
            print('Plotting PCs #{:01d} and #{:01d}'.format(i+1, j+1))
            # SAVE
            fig.savefig('pc{:01d}_vs_pc{:01d}.png'.format(i+1, j+1),
                        dpi=300,
                        transparent=True,
                        format='png',
                        pad_inches=0)
            
    # PRINT
    print('Finished analyzing SVM')
    print()
    #
    plt.show(block=True)