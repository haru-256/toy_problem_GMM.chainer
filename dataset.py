from chainer.dataset import DatasetMixin
import numpy as np
import math


class GmmDataset(DatasetMixin):
    """
    Generate GMM of 2dim data set as Toy problem

    Parameters
        ---------------
        datasize: int
            the number of dataset

        num_cluster: int
            the number of cluster

        scale: float
            scale used as mean of clusters

        std: float
            std used as mean of clusters

        seed: int
            fix random by this value

    """

    def __init__(self, datasize, seed, num_cluster=8, std=1, scale=1):
        print('Make dataset...')
        self._data = self.gaussian_mixture_circle(
            datasize, seed, num_cluster, scale, std)
        print('Finish !!', end='\n\n')

    def gaussian_mixture_circle(self, datasize, seed, num_cluster=8, scale=1, std=1):
        """
        make GMM data

        Parameters
        ---------------
        datasize: int
            the number of dataset

        num_cluster: int
            the number of cluster

        scale: float
            scale used as mean of clusters

        std: float
            std used as mean of clusters

        seed: int
            fix random by this value

        Returns
        --------------
        data: array whose shape is (datasize, 2)

        """
        np.random.seed(seed)  # fix random
        rand_indices = np.random.randint(0, num_cluster, size=datasize)
        base_angle = math.pi * 2 / num_cluster
        angle = rand_indices * base_angle
        mean = np.zeros((datasize, 2), dtype=np.float32)
        mean[:, 0] = np.cos(angle) * scale
        mean[:, 1] = np.sin(angle) * scale

        return np.random.normal(mean, std, (datasize, 2)).astype(np.float32)

    def __len__(self):
        return len(self._data)

    def get_example(self,  i):
        return self._data[i]
