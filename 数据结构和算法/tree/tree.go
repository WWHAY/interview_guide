package tree

import "math"

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// 翻转二叉树
// 给你一棵二叉树的根节点 root ，翻转这棵二叉树，并返回其根节点。
func invertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	left := invertTree(root.Left)
	right := invertTree(root.Right)
	root.Left = right
	root.Right = left
	return root
}

// 中序遍历
// 给定一个二叉树的根节点 root ，返回 它的 中序 遍历 。
func inorderTraversal(root *TreeNode) []int {
	res := []int{}
	for root != nil {
		// 左子树
		if root.Left != nil {
			// 这个目的是将所有的叶子结点都指向root
			p := root.Left
			for p.Right != nil && p.Right != root {
				p = p.Right
			}

			if p.Right == nil {
				p.Right = root
				root = root.Left
			} else {
				res = append(res, root.Val)
				p.Right = nil
				root = root.Right
			}
		} else {
			// 左子树为空了，该遍历右子树了
			res = append(res, root.Val)
			// p在最开始的时候，把所有的叶子结点的右指针都会指向root
			// root.Right也就是会到了叶子结点的父节点
			root = root.Right
		}
	}
	return res
}

// 递归
func inorderTraversal1(root *TreeNode) []int {
	var inorder func(node *TreeNode)
	res := []int{}
	inorder = func(node *TreeNode) {
		if node == nil {
			return
		}

		inorder(node.Left)
		res = append(res, node.Val)
		inorder(node.Right)
	}

	inorder(root)
	return res
}

// 给定一个二叉树 root ，返回其最大深度。
// 二叉树的 最大深度 是指从根节点到最远叶子节点的最长路径上的节点数。
func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}

	return max(maxDepth(root.Left), maxDepth(root.Right)) + 1
}

func max(x, y int) int {
	if x > y {
		return x
	}

	return y
}

// 判断二叉树是否对称
// 给你一个二叉树的根节点 root ， 检查它是否轴对称。
func isSymmetric(root *TreeNode) bool {
	return check(root, root)
}

func check(p, q *TreeNode) bool {
	if p == nil && q == nil {
		return true
	}

	if p == nil || q == nil {
		return false
	}

	return p.Val == q.Val && check(p.Right, q.Left) && check(p.Left, q.Right)
}

// 给你一棵二叉树的根节点，返回该树的 直径 。
// 二叉树的 直径 是指树中任意两个节点之间最长路径的 长度 。这条路径可能经过也可能不经过根节点 root 。
// 两节点之间路径的 长度 由它们之间边数表示。
func diameterOfBinaryTree(root *TreeNode) int {
	var depth func(root *TreeNode) int
	ans := 0
	depth = func(root *TreeNode) int {
		if root == nil {
			return 0
		}

		r := depth(root.Right)
		l := depth(root.Left)
		ans = max(ans, l+r+1)
		return max(r, l) + 1
	}
	depth(root)
	return ans - 1
}

// 给你二叉树的根节点 root ，返回其节点值的 层序遍历 。 （即逐层地，从左到右访问所有节点）。
func levelOrder(root *TreeNode) [][]int {
	if root == nil {
		return nil
	}
	quene := []*TreeNode{root}
	ans := [][]int{}
	for len(quene) > 0 {
		sz := len(quene)
		temp := []int{}
		for sz > 0 {
			node := quene[0]
			quene = quene[1:]
			temp = append(temp, node.Val)

			if node.Left != nil {
				quene = append(quene, node.Left)
			}

			if node.Right != nil {
				quene = append(quene, node.Right)
			}
			sz--
		}
		ans = append(ans, temp)
	}
	return ans
}

// 给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树。

// 高度平衡 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1 」的二叉树。
func sortedArrayToBST(nums []int) *TreeNode {
	if len(nums) == 0 {
		return nil
	}
	return helper(nums, 0, len(nums)-1)
}

func helper(nums []int, left, right int) *TreeNode {
	if left > right {
		return nil
	}
	mid := (left + right) / 2
	root := &TreeNode{
		Val: nums[mid],
	}

	root.Left = helper(nums, left, mid-1)
	root.Right = helper(nums, mid+1, right)
	return root
}

// 给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。
// 有效 二叉搜索树定义如下：
// 节点的左子树只包含 小于 当前节点的数。
// 节点的右子树只包含 大于 当前节点的数。
// 所有左子树和右子树自身必须也是二叉搜索树。
func isValidBST(root *TreeNode) bool {
	if root == nil {
		return false
	}

	return helper1(root, math.MinInt64, math.MaxInt64)
}

func helper1(node *TreeNode, left, right int) bool {
	if node == nil {
		return true
	}
	if node.Val <= left || node.Val >= right {
		return false
	}
	return helper1(node.Left, left, node.Val) && helper1(node.Right, node.Val, right)
}

// 给定一个二叉搜索树的根节点 root ，和一个整数 k ，请你设计一个算法查找其中第 k 个最小元素（从 1 开始计数）。
type MyBst struct {
	root    *TreeNode
	nodeNum map[*TreeNode]int // 统计以每个结点为根结点的子树的结点数，并存储在哈希表中
}

// 对当前结点 node\textit{node}node 进行如下操作：

// 如果 node\textit{node}node 的左子树的结点数 left\textit{left}left 小于 k−1k-1k−1，则第 kkk 小的元素一定在 node\textit{node}node 的右子树中，令 node\textit{node}node 等于其的右子结点，kkk 等于 k−left−1k - \textit{left} - 1k−left−1，并继续搜索；
// 如果 node\textit{node}node 的左子树的结点数 left\textit{left}left 等于 k−1k-1k−1，则第 kkk 小的元素即为 nodenodenode ，结束搜索并返回 node\textit{node}node 即可；
// 如果 node\textit{node}node 的左子树的结点数 left\textit{left}left 大于 k−1k-1k−1，则第 kkk 小的元素一定在 node\textit{node}node 的左子树中，令 node\textit{node}node 等于其左子结点，并继续搜索。

// 统计以 node 为根结点的子树的结点数
func (t *MyBst) countNodeNum(node *TreeNode) int {
	if node == nil {
		return 0
	}
	t.nodeNum[node] = 1 + t.countNodeNum(node.Left) + t.countNodeNum(node.Right)
	return t.nodeNum[node]
}

// 返回二叉搜索树中第 k 小的元素
func (t *MyBst) kthSmallest(k int) int {
	node := t.root
	for {
		leftNodeNum := t.nodeNum[node.Left]
		if leftNodeNum < k-1 {
			node = node.Right
			k -= leftNodeNum + 1
		} else if leftNodeNum == k-1 {
			return node.Val
		} else {
			node = node.Left
		}
	}
}

func kthSmallest1(root *TreeNode, k int) int {
	t := &MyBst{root, map[*TreeNode]int{}}
	t.countNodeNum(root)
	return t.kthSmallest(k)
}

func kthSmallest(root *TreeNode, k int) int {
	if root == nil {
		return 0
	}
	res := []int{}
	var inorder func(root *TreeNode)
	inorder = func(root *TreeNode) {
		if root == nil {
			return
		}

		inorder(root.Left)
		res = append(res, root.Val)
		inorder(root.Right)
	}
	inorder(root)

	if len(res) < k {
		return 0
	}

	return res[k-1]
}

// 给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
func rightSideViewBfs(root *TreeNode) []int {
	if root == nil {
		return nil
	}
	quene := []*TreeNode{root}
	ans := []int{}
	for len(quene) > 0 {
		sz := len(quene)
		ans = append(ans, quene[0].Val)
		for sz > 0 {
			node := quene[0]
			quene = quene[1:]

			// 先记录右节点
			if node.Right != nil {
				quene = append(quene, node.Right)
			}

			if node.Left != nil {
				quene = append(quene, node.Left)
			}

			sz--
		}
	}
	return ans
}

// 最核心的点是在于：
// 1.  depth + 1 和len的比较
// 2. 根右左，一定是先遍历右边，再去遍历左边
func rightSideViewDfs(root *TreeNode) []int {
	//特判
	if root == nil {
		return nil
	}

	var (
		dfs func(node *TreeNode, depth int) // dfs 根右左 “中序遍历” 记录每一层最右侧的元素
		ans []int
	)

	dfs = func(node *TreeNode, depth int) {
		if node == nil {
			return
		}
		//每层第一个出现的元素——最右侧元素
		if depth == len(ans) {
			ans = append(ans, node.Val)
		}
		//根右左 “中序遍历”
		dfs(node.Right, depth+1)
		dfs(node.Left, depth+1)
	}

	dfs(root, 0)
	return ans
}

// 给你二叉树的根结点 root ，请你将它展开为一个单链表：
// 展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
// 展开后的单链表应该与二叉树 先序遍历 顺序相同。
// 需要注意的是一定要理解指针和变量之间的区别
func flatten(root *TreeNode) {
	if root == nil {
		return
	}
	// 前序遍历
	var order func(root *TreeNode)
	list := []*TreeNode{}
	order = func(node *TreeNode) {
		if node == nil {
			return
		}
		list = append(list, node)
		order(node.Left)
		order(node.Right)
	}
	order(root)

	// 这个是关键，list列表中都是node，改变node即是改变该值
	for i := 1; i < len(list); i++ {
		prev, curr := list[i-1], list[i]
		prev.Left, prev.Right = nil, curr
	}
}

// 迭代
// 前序遍历
func flatten1(root *TreeNode) {
	list := []*TreeNode{}
	stack := []*TreeNode{}
	node := root
	for node != nil || len(stack) > 0 {
		for node != nil {
			// 顶点元素
			list = append(list, node)
			stack = append(stack, node)
			node = node.Left
		}
		node = stack[len(stack)-1]
		node = node.Right
		stack = stack[:len(stack)-1]
	}

	for i := 1; i < len(list); i++ {
		prev, curr := list[i-1], list[i]
		prev.Left, prev.Right = nil, curr
	}
}

// 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
// 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}

	if root == p || root == q {
		return root
	}

	left := lowestCommonAncestor(root.Left, p, q)
	right := lowestCommonAncestor(root.Right, p, q)

	if left != nil && right != nil {
		return root
	}

	if left == nil {
		return right
	}

	return left
}

func lowestCommonAncestor1(root, p, q *TreeNode) *TreeNode {
	parent := map[int]*TreeNode{}
	visited := map[int]bool{}

	// 记录所有节点的父节点
	var dfs func(*TreeNode)
	dfs = func(r *TreeNode) {
		if r == nil {
			return
		}

		// 叶子结点的父节点
		if r.Left != nil {
			parent[r.Left.Val] = r
			dfs(r.Left)
		}
		if r.Right != nil {
			parent[r.Right.Val] = r
			dfs(r.Right)
		}
	}
	dfs(root)

	// 遍历p所有的父节点
	for p != nil {
		visited[p.Val] = true
		p = parent[p.Val]
	}

	// 通过p遍历的父节点，一个节点为true，既是最近公共父节点
	for q != nil {
		if visited[q.Val] {
			return q
		}
		q = parent[q.Val]
	}

	return nil
}
