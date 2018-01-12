import os
import sys
import numpy as np
from skimage import io

def read_data(dir):
    X = []
    size = None
    print('reading data...')    
    for file in sorted(os.listdir(dir)):
        img = io.imread(os.path.join(dir, file))
        size = img.shape
        X.append(img.flatten())
    print('reading data done...')
    return np.array(X), size

def reconstruct(X, size, top, DATA_FILE, target_data):
    mean = np.mean(X, axis=0)
    X_center = X - mean
    print('reconstructing...') 
    U, S, V = np.linalg.svd(X_center, full_matrices=False)
    eigenvector = V
    
    img = io.imread(os.path.join(DATA_FILE, target_data))
    img = img.flatten()
    project = np.dot(eigenvector[:top,:], img - mean)
    recon = mean + np.dot(eigenvector[:top,:].T, project)
    recon -= np.min(recon)
    recon /= np.max(recon)
    recon = (recon*255).astype(np.uint8)        
    io.imsave('reconstruction.jpg', recon.reshape(size))    
    print('reconstruction done...')    
    

def main(args):
    img, size = read_data(args[1])
    reconstruct(img, size, 4, args[1], args[2])

if __name__ == '__main__':
    main(sys.argv)
