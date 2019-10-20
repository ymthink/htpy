# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2019/10/14 13:45


import numpy as np
from scipy.linalg import svd
import scipy.io as sio
from collections import deque


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
    shape = np.array(np.shape(x))
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
        shape_core = np.array(np.shape(x_))
        if node.indices[0] == 0:
            trans_indices = np.concatenate([node.indices, other_indices])
            x_ = np.transpose(x_, trans_indices)
            x_ = np.reshape(x_, [np.prod(shape_core[node.indices]), np.prod(shape_core[other_indices])])
            x_ = np.matmul(u.T, x_)
            shape_core[node.indices[0]] = node.rank
            x_ = np.reshape(x_, shape_core[trans_indices])

        else:
            trans_indices = np.concatenate([other_indices, node.indices])
            x_ = np.transpose(x_, trans_indices)
            x_ = np.reshape(x_, [np.prod(shape_core[other_indices]), np.prod(shape_core[node.indices])])
            x_ = np.matmul(x_, u)
            shape_core[node.indices[0]] = node.rank
            x_ = np.reshape(x_, shape_core[trans_indices])

        new_indices = []
        for i in range(len(shape_core)):
            for j in range(len(shape_core)):
                if trans_indices[j] == i:
                    new_indices.append(j)
                    break

        x_ = np.transpose(x_, new_indices)

    x = x_

    rmax -= 1
    # compute cluster decomposition
    for level in range(level_max, -1, -1):
        count = 0
        for node in find_cluster(root, level):
            if level < (level_max - 1):
                cur_indices = np.arange(2*count, 2*(count+1))
                cur_mode = 2 ** (level + 1)
            else:
                cur_indices = node.indices
                cur_mode = n_mode

            shape_core = np.array(np.shape(x))
            other_indices = np.concatenate([np.arange(0, cur_indices[0]), np.arange(cur_indices[-1]+1, cur_mode)])
            trans_indices = np.concatenate([cur_indices, other_indices])
            x_mat = np.transpose(x, trans_indices)
            x_mat = np.reshape(x_mat, [np.prod(shape_core[cur_indices]), np.prod(shape_core[other_indices])])
            if level == 0:
                node.rank = 1
                u = x_mat
            else:
                u, s, vt = svd(x_mat)
                if len(s) > rmax:
                    u = u[:, :rmax]
                    s = s[:rmax]
                    vt = vt[:rmax, :]
                    node.rank = rmax
                else:
                    node.rank = len(s)
                    u = u[:, :node.rank]

            b = np.reshape(u, [node.left.rank, node.right.rank, node.rank])
            node.set_b(b)

            shape_core = np.array(np.shape(x_))
            cur_mode -= count
            cur_indices -= count
            other_indices = np.concatenate([np.arange(0, cur_indices[0]), np.arange(cur_indices[-1]+1, cur_mode)])
            trans_indices = np.concatenate([cur_indices, other_indices])
            x_mat_ = np.transpose(x_, trans_indices)
            x_mat_ = np.reshape(x_mat_, [np.prod(shape_core[cur_indices]), np.prod(shape_core[other_indices])])
            u_x_mat_ = np.matmul(u.T, x_mat_)

            new_indices = []
            for i in range(len(trans_indices)):
                if trans_indices[i] < cur_indices[0]:
                    new_indices.append(trans_indices[i])
                elif trans_indices[i] > cur_indices[-1]:
                    new_indices.append(trans_indices[i]-len(cur_indices)+1)
                elif trans_indices[i] == cur_indices[0]:
                    new_indices.append(cur_indices[0])

            trans_indices = []
            for i in range(len(new_indices)):
                for j in range(len(new_indices)):
                    if new_indices[j] == i:
                        trans_indices.append(j)
                        break

            new_shape = np.concatenate([[node.rank], shape_core[other_indices]])
            x_ = np.reshape(u_x_mat_, new_shape)
            x_ = np.transpose(x_, trans_indices)
            count += 1

        x = x_

    return root, level_max


def ht_full(root, level_max):
    for level in range(level_max, -1, -1):
        for node in find_cluster(root, level):
            shape_b = np.array(np.shape(node.b))
            b_mat = np.reshape(node.b, [node.left.rank, np.prod(shape_b[1:])])
            shape_left = np.array(np.shape(node.left.u))
            left_mat = np.reshape(node.left.u, [np.prod(shape_left[:-1]), node.left.rank])
            left_b_mat = np.matmul(left_mat, b_mat)
            left_b_mat = np.reshape(left_b_mat, [np.prod(shape_left[:-1]), node.right.rank, node.rank])
            left_b_mat = np.transpose(left_b_mat, [0, 2, 1])
            left_b_mat = np.reshape(left_b_mat, [np.prod(shape_left[:-1])*node.rank, node.right.rank])
            shape_right = np.array(np.shape(node.right.u))
            right_mat = np.reshape(node.right.u, [np.prod(shape_right[:-1]), node.right.rank])
            right_mat = np.transpose(right_mat)
            left_b_right_mat = np.matmul(left_b_mat, right_mat)
            left_b_right_mat = np.reshape(left_b_right_mat, [np.prod(shape_left[:-1]), node.rank, np.prod(shape_right[:-1])])
            new_shape = np.concatenate([shape_left[:-1], [node.rank], shape_right[:-1]])
            left_b_right = np.reshape(left_b_right_mat, new_shape)
            rank_index = len(shape_left[:-1])
            new_indices = np.concatenate([np.arange(0, rank_index), np.arange(rank_index+1, len(new_shape)), [rank_index]])
            left_b_right = np.transpose(left_b_right, new_indices)
            node.set_u(left_b_right)

    x_ht = np.squeeze(root.u)
    return x_ht


def find_leaf(root):
    q = deque()
    q.append(root)
    while len(q) > 0:
        cur_node = q.popleft()
        if cur_node.is_leaf:
            yield cur_node
        else:
            q.append(cur_node.left)
            q.append(cur_node.right)


def find_cluster(root, level):
    q = deque()
    q.append(root)
    while len(q) > 0:
        cur_node = q.popleft()
        if not cur_node.is_leaf:
            q.append(cur_node.left)
            q.append(cur_node.right)
            if cur_node.level == level:
                yield cur_node


if __name__ == '__main__':
    # x = sio.loadmat('x.mat')['x']
    x = np.random.random([5, 6, 7, 8, 9])
    sio.savemat('x.mat', {'x':x})
    root, level_max = truncate_ltr(x, 50)
    x_ht = ht_full(root, level_max)

    err = np.sum(x_ht - x)
    print(err)




