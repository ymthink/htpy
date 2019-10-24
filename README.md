# htpy
A Hierarchical Tensor Decomposition library in Python

## Data Structure
- All component in HT format are defined in a binary tree, which is an instance of class `Node`.

## Usage
- Run `truncate_ltr(x, rmax)` to decompose a tensor `x` to HT format, whose maximum rank of each node is `rmax`.
- Run `ht_full(x)` to recover an HT format to the original tensor.
