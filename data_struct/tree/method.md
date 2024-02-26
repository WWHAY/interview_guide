# 树
对于数据结构和算法来说，关于树的结构一般是二叉树，对于B树、B+树、红黑树等多个子树的树来说，不是一个常考的考点
## 思考技巧
### 深度优先搜索
### 广度优先搜索
### 递归
#### 前序遍历
```
func preOrder(node *TreeNode) {
		if node == nil {
			return
		}
		list = append(list, node)
		order(node.Left)
		order(node.Right)
	}
```
#### 中序遍历
**二叉搜索树为升序排列**
```
func inorderTraversal(root *TreeNode) (res []int) {
	var inorder func(node *TreeNode)
	inorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		inorder(node.Left)
		res = append(res, node.Val)
		inorder(node.Right)
	}
	inorder(root)
	return
}

```
#### 后序遍历
```
func postorderTraversal(root *TreeNode) (res []int) {
    var postorder func(*TreeNode)
    postorder = func(node *TreeNode) {
        if node == nil {
            return
        }
        postorder(node.Left)
        postorder(node.Right)
        res = append(res, node.Val)
    }
    postorder(root)
    return
}
```
### 迭代
#### 前序遍历
```
func preorderTraversal(root *TreeNode) (vals []int) {
    stack := []*TreeNode{}
    node := root
    for node != nil || len(stack) > 0 {
        for node != nil {
            vals = append(vals, node.Val)
            stack = append(stack, node)
            node = node.Left
        }
        node = stack[len(stack)-1].Right
        stack = stack[:len(stack)-1]
    }
    return
}
```
#### 中序遍历
**二叉搜索树为升序排列**
```
func inorderTraversal(root *TreeNode) (res []int) {
	stack := []*TreeNode{}
	for root != nil || len(stack) > 0 {
		for root != nil {
			stack = append(stack, root)
			root = root.Left
		}
		root = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		res = append(res, root.Val)
		root = root.Right
	}
	return
}
```
#### 后序遍历
prev的作用是证明右子树已经访问完成了，或者是正在访问的右子树是叶子结点，所以可以直接添加，同时更新pre的值就OK
只有左节点是可以直接加入的
但是前提一定保证的是stack中第一个一定是左节点
```
func postorderTraversal(root *TreeNode) (res []int) {
    stk := []*TreeNode{}
    var prev *TreeNode
    for root != nil || len(stk) > 0 {
        for root != nil {
            stk = append(stk, root)
            root = root.Left
        }
        root = stk[len(stk)-1]
        stk = stk[:len(stk)-1]
        if root.Right == nil || root.Right == prev {
            res = append(res, root.Val)
            prev = root
            root = nil
        } else {
            stk = append(stk, root)
            root = root.Right
        }
    }
    return
}
```

## 二叉树和哈希的联动
在二叉树的计算当中，使用哈希记录当前节点的路径和以及节点数之类的，是常用的解法之一
## 特点
二叉搜索树的中序遍历是升序序列，题目给定的数组是按照升序排序的有序数组，因此可以确保数组是二叉搜索树的中序遍历序列。
## 总结
1. 对于树相关的问题，一定要利用题目的特点，以及指针对构思问题

## 二叉搜索树
若任意节点的左子树不空，则所有的左子树上的节点的值都它的小于根节点的值
若任意节点的右子树不空，则所有的右子树的节点的值都大于它的根节点的值
任意节点的左右子树也分别为二叉搜索树
### 特点
1. 中序遍历二叉查找树可以得到一个关键字的有序序列， -- 左子树的值一定小于右子树的值
2. 一个无序序列可以通过构建二叉查找树变成一个有序序列，构建的过程就是对无序序列的过程，每次插入的节点都是二叉查找树的子节点，在插入操作时，不必移动其它节点，只需改动节点指针，由空变成非空即可
3. 搜索 删除  插入的复杂度等于树高，期望O(logn), 最坏O(n)(变成线性树)



