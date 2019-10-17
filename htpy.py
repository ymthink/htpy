# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2019/10/14 13:45


import numpy as np
from scipy.linalg import svd
import scipy.io as sio


class Node:
    def __init__(self, left, right, indices, rank, is_leaf, level):
        self.left = left
        self.right = right
        self.indices = indices
        self.rank = rank
        self.is_leaf = is_leaf
        self.level = level
        self.u = None
        self.b = None

    def set_rank(self, rank):
        self.rank = rank

    def set_u(self, u):
        self.u = u

    def set_b(self, b):
        self.b = b


def indices_tree(n_mode):
    s = []
    root = Node(None, None, np.arange(0, n_mode), 0, False, 0)
    s.append(root)
    level_max = 0
    while len(s) > 0:
        cur_node = s.pop()
        if cur_node.level > level_max:
            level_max = cur_node.level
        if len(cur_node.indices) > 1:
            mid = len(cur_node.indices) // 2
            left_node = Node(None, None, np.arange(cur_node.indices[0], cur_node.indices[mid]), 0, False, cur_node.level+1)
            right_node = Node(None, None, np.arange(cur_node.indices[mid], cur_node.indices[-1]+1), 0, False, cur_node.level+1)
            cur_node.left = left_node
            cur_node.right = right_node
            s.append(left_node)
            s.append(right_node)
        else:
            cur_node.is_leaf = True

    return root, level_max


# leaf to root truncation
def truncate_ltr(x, rmax):
    x_ = np.copy(x)
    shape = np.shape(x)
    n_mode = len(shape)
    root, level_max = indices_tree(n_mode)

    # compute leaf decomposition
    for node in find_leaf(root):
        other_indices = np.concatenate([np.arange(0, node.indices[0]), np.arange(node.indices[0]+1, n_mode)])
        trans_indices = np.concatenate([node.indices, other_indices])
        x_mat = np.transpose(x, trans_indices)
        x_mat = np.reshape(x_mat, [np.prod(shape[node.indices]), np.prod(shape[other_indices])])
        u, s, vt = svd(x_mat)
        if len(s) > rmax:
            u = u[:, :rmax]
            s = s[:rmax]
            vt = vt[:rmax, :]
            node.rank = rmax
        else:
            node.rank = len(s)
        node.set_u(u)
        shape_core = np.shape(x_)
        trans_indices = np.concatenate([other_indices, node.indices])
        x_ = np.transpose(x_, trans_indices)
        x_ = np.reshape(x_, [np.prod(shape_core[other_indices]), shape_core[node.indices]])
        x_ = np.matmul(x_, u.T)
        shape_core[node.indices[0]] = node.rank
        x_ = np.reshape(x_, shape_core[trans_indices])
        trans_indices = np.concatenate([np.arange(0, node.indices[0]), np.array(n_mode-1), np.arange(node.indices[0], n_mode-1)])
        x_ = np.transpose(x_, trans_indices)

    x = x_

    # compute cluster decomposition
    for level in range(level_max):
        for node in find_cluster(root, level):
            shape_core = np.shape(x)
            other_indices = np.concatenate([np.arange(0, node.indices[0]), np.arange(node.indices[-1], n_mode)])
            trans_indices = np.concatenate([node.indices, other_indices])
            x_mat = np.transpose(x, trans_indices)
            x_mat = np.reshape(x_mat, [np.prod(shape_core[node.indices]), np.prod(shape_core[other_indices])])
            u, s, vt = svd(x_mat)
            if len(s) > rmax:
                u = u[:, :rmax]
                s = s[:rmax]
                vt = vt[:rmax, :]
                node.rank = rmax
            else:
                node.rank = len(s)

            b = np.reshape(u, [node.left.rank, node.right.rank, node.rank])
            node.set_b(b)

            shape_core = np.shape(x_)
            other_indices = np.concatenate([np.arange(0, node.indices[0]), np.arange(node.indices[-1], n_mode)])
            trans_indices = np.concatenate([node.indices, other_indices])
            x_mat_ = np.transpose(x_, trans_indices)
            x_mat_ = np.reshape(x_mat_, [np.prod(shape_core[node.indices]), np.prod(shape_core[other_indices])])
            u_x_mat_ = np.matmul(u.T, x_mat_)

            cur_indices = []
            for i in range(len(trans_indices)):
                if trans_indices[i] < node.indices[0]:
                    cur_indices.append(trans_indices[i])
                elif trans_indices[i] > node.indices[-1]:
                    cur_indices.append(trans_indices[i]-len(node.indices)-1)
                elif trans_indices == node.indices[0]:
                    cur_indices.append(node.indices[0])

            new_indices = []
            for i in range(len(cur_indices)):
                for j in range(len(cur_indices)):
                    if cur_indices[j] == i:
                        new_indices.append(j)
                        break

            new_shape = np.concatenate([[node.rank], other_indices])
            x_ = np.reshape(u_x_mat_, new_shape)
            x_ = np.transpose(x_, new_indices)

        x = x_


def find_leaf(root):
    s = []
    s.append(root)
    while len(s) > 0:
        cur_node = s.pop()
        if cur_node.is_leaf:
            yield cur_node
        else:
            s.append(cur_node.left)
            s.append(cur_node.right)


def find_cluster(root, level):
    s = []
    s.append(root)
    while len(s) > 0:
        cur_node = s.pop()
        if not cur_node.is_leaf:
            s.append(cur_node.left)
            s.append(cur_node.right)
            if cur_node.level == level:
                yield cur_node


if __name__ == '__main__':
    root = indices_tree(5)
    print(root)



