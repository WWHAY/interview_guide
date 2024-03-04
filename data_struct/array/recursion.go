package array

// 给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。
// 解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
// 题解：回溯和深度优先搜索
func subsets(nums []int) (ans [][]int) {
	set := []int{}
	var dfs func(cur int)
	dfs = func(cur int) {
		// 如果cur 和nums的长度相等停止递归
		if cur == len(nums) {
			// 切片是指针传递，将切片的元素传给ans，所以声明了一个空数组
			ans = append(ans, append([]int{}, set...))
			return
		}
		// 包含自己的元素
		set = append(set, nums[cur])
		dfs(cur + 1)
		// 去除自身元素，当cur=1是，set已经是空了，开始从第二元素开始回溯
		set = set[:len(set)-1]
		dfs(cur + 1)
	}
	dfs(0)
	return ans
}

// 给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。
// 题解：深度优先搜索，确定状态变量，确定回溯表达式，确定递归终止条件，恢复状态值，临时结果存储，这个几个是做回溯类型的题目最为关键的节点
func permute(nums []int) (ans [][]int) {
	n := len(nums)
	onPath := make([]bool, n)
	path := make([]int, n)
	var dfs func(cur int)
	dfs = func(cur int) {
		if cur == n {
			ans = append(ans, append([]int{}, path...))
			return
		}

		for j, on := range onPath {
			if !on {
				path[cur] = nums[j]
				onPath[j] = true
				dfs(cur + 1)
				// 复原
				onPath[j] = false
			}
		}
	}
	dfs(0)
	return
}
