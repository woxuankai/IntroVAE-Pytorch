import sys
import numpy as np
import sklearn
import sklearn.manifold
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import time
from operator import itemgetter 

if len(sys.argv) == 1:
    npzfile = np.load('result_reduction.npz')
    indices = npzfile['indices']
    X_2d = npzfile['X_2d']
    X_2d = np.split(X_2d, indices)
    samples = random.sample(range(len(X_2d)), 10)
    selected = np.concatenate(itemgetter(*samples)(X_2d))

    fig, ax = plt.subplots()
    lines = [ax.plot([], [], '+-', label=str(ind+1))[0] for ind in samples]
    texts = [ax.text(0,0,'') for ind in samples]
    plt.legend()
    ax.set_xlim([selected[:,0].min(), selected[:,0].max()])
    ax.set_ylim([selected[:,1].min(), selected[:,1].max()])

    #def init():
    #    for line in lines:
    #        line.set_data([],[])
    #    return lines

    def animate(count):
        modified = []
        for line, text, ind in zip(lines, texts, samples):
            if count <= len(X_2d[ind]):
                line.set_data(X_2d[ind][:count, 0], X_2d[ind][:count, 1])
                text.set_position(X_2d[ind][count-1])
                text.set_text(str(count-1))
                modified.append(line)
                modified.append(text)
        return modified

    ani = animation.FuncAnimation(fig, animate, \
            blit=False, interval=200, save_count=120)
    ani.save('movie.mp4')
    plt.show()

else:
    features = []
    indices = []
    index_cur = 0
    for filepath in sys.argv[1:]:
        feature = np.load(filepath)
        start = feature.shape[0]//4
        stop = feature.shape[0]//4*3
        feature = feature[start:stop,:]
        index_cur += feature.shape[0]
        features.append(feature)
        indices.append(index_cur)
    indices.pop()
    X = np.concatenate(features)
    print('collected '+str(len(features))+' files, ' \
            + str(X.shape[0])+' features')
    reduction = sklearn.manifold.Isomap(n_components=2)
    #reduction = sklearn.manifold.TSNE(n_components=2)
    X_2d = reduction.fit_transform(X)
    np.savez('result_reduction.npz', indices=indices, X=X, X_2d=X_2d)
    print('dim reduction fitted and saved')
