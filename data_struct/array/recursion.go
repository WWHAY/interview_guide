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
// 全排列是数组中元素的顺序
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

// 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
// 给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
// 题解：固定第一个数，然后选择第二个数，判断index和string长度的问题
// 定义全局变量
var phoneMap map[string]string = map[string]string{
	"2": "abc",
	"3": "def",
	"4": "ghi",
	"5": "jkl",
	"6": "mno",
	"7": "pqrs",
	"8": "tuv",
	"9": "wxyz",
}

var combinations []string

func backtrack(digits string, index int, combination string) {
	if len(digits) == index {
		combinations = append(combinations, combination)
		return
	}

	digit := string(digits[index])
	letters := phoneMap[digit]
	for i := 0; i < len(letters); i++ {
		backtrack(digits, index+1, combination+string(letters[i]))
	}
}

func letterCombinations(digits string) []string {
	if len(digits) == 0 {
		return nil
	}
	combinations = []string{}
	backtrack(digits, 0, "")
	return combinations
}

// 给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，
// 找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。

// candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。

// 对于给定的输入，保证和为 target 的不同组合数少于 150 个。
func combinationSum(candidates []int, target int) [][]int {
	if len(candidates) == 0 {
		return nil
	}
	res := [][]int{}
	path := []int{}
	var dfs func(totalTarget, idx int)
	dfs = func(totalTarget, idx int) {
		if idx == len(candidates) {
			return
		}

		if totalTarget == 0 {
			res = append(res, append([]int{}, path...))
			return
		}
		// 查看数组中是否存在数值==target
		dfs(totalTarget, idx+1)

		// 回溯
		if totalTarget-candidates[idx] >= 0 {
			path = append(path, candidates[idx])
			dfs(totalTarget-candidates[idx], idx)
			path = path[:len(path)-1]
		}
	}
	dfs(target, 0)
	return res
}

// 数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。
func generateParenthesis(n int) []string {
	if n == 0 {
		return nil
	}

	ans := []string{}
	path := ""
	var dfs func(open, close int)
	dfs = func(open, close int) {
		if len(path) == 2*n {
			ans = append(ans, path)
		}

		if open < n {
			path += "("
			dfs(open+1, close)
			path = path[:len(path)-1]
		}

		if close < open {
			path += ")"
			dfs(open, close+1)
			path = path[:len(path)-1]
		}
	}
	dfs(0, 0)
	return ans
}

// 给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是 回文串 。返回 s 所有可能的分割方案。

// 回文串 是正着读和反着读都一样的字符串。
// 题解：动态规划+回溯的双重解法
func partition(s string) [][]string {
	// 预处理,判断从i到j是否是回文字符串
	dp := make([][]bool, len(s))
	for i := range dp {
		dp[i] = make([]bool, len(s))
		for j := range dp[i] {
			dp[i][j] = true // 初始化数值
		}
	}

	for i := len(s) - 1; i >= 0; i-- {
		for j := i + 1; j < len(s); j++ {
			// 字符串，首位相等&&中间部分也是字符串
			dp[i][j] = s[i] == s[j] && dp[i+1][j-1]
		}
	}
	ans := [][]string{}
	path := []string{}
	var dfs func(i int)
	dfs = func(i int) {
		if i == len(s) {
			ans = append(ans, append([]string{}, path...))
			return
		}

		// 回溯
		for j := i; j < len(s); j++ {
			// 判断是否是回文数
			if dp[i][j] {
				path = append(path, s[i:j+1])
				dfs(j + 1)
				path = path[:len(path)-1]
			}
		}
	}
	dfs(0)
	return ans
}
