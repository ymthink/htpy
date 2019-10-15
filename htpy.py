# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2019/10/14 13:45


import numpy as np
from scipy.linalg import svd
import scipy.io as sio


class Node:
    def __init__(self, left, right, indices, rank, is_leaf):
        self.left = left
        self.right = right
        self.indices = indices
        self.rank = rank
        self.is_leaf = is_leaf

    def set_rank(self, rank):
        self.rank = rank


def indices_tree(n_mode):
    s = []
    root = Node(None, None, np.arange(0, n_mode), 0, False)
    s.append(root)
    while len(s) > 0:
        cur_node = s.pop()
        if len(cur_node.indices) > 1:
            mid = len(cur_node.indices) // 2
            left_node = Node(None, None, np.arange(cur_node.indices[0], cur_node.indices[mid]), 0, False)
            right_node = Node(None, None, np.arange(cur_node.indices[mid], cur_node.indices[-1]+1), 0, False)
            cur_node.left = left_node
            cur_node.right = right_node
            s.append(left_node)
            s.append(right_node)
        else:
            cur_node.is_leaf = True

    return root


def truncate_ltr(x, rmax):
    shape = np.shape(x)
    n_mode = len(shape)
    root = indices_tree(n_mode)
    for node in find_leaf(root):
        other_indices = np.concatenate([np.arange(0, node.indices[0]), np.arange(node.indices[0]+1, n_mode)])
        trans_indices = np.concatenate([node.indices, other_indices])
        x_mat = np.transpose(x, trans_indices)
        x_mat = np.reshape(x_mat, [np.prod(node.indices), np.prod(other_indices)])
        u, s, vt = svd(x_mat)
        if len(s) > rmax:
            u = u[:, :rmax]
            s = s[:rmax]
            vt = vt[:rmax, :]
            node.rank = rmax
        else:
            node.rank = len(s)




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


def find_notleaf(root):
    s = []
    s.append(root)
    while len(s) > 0:
        cur_node = s.pop()
        if not cur_node.is_leaf:
            s.append(cur_node.left)
            s.append(cur_node.right)
            yield cur_node


if __name__ == '__main__':
    root = indices_tree(5)
    print(root)



