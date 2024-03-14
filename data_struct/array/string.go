package array

// 1. 最小覆盖子串
// 给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。
// 注意：
// 对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量。
// 如果 s 中存在这样的子串，我们保证它是唯一的答案。
// 示例 1：
// 输入：s = "ADOBECODEBANC", t = "ABC"
// 输出："BANC"
// 解释：最小覆盖子串 "BANC" 包含来自字符串 t 的 'A'、'B' 和 'C'。

// 2. 无重复字符的最长子串
// 给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
// 示例 1:
// 输入: s = "abcabcbb"
// 输出: 3
// 解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
func lengthOfLongestSubstring(s string) int {
	cnt := make(map[byte]int, 0)
	ans := 0
	r := -1
	for l := 0; l < len(s); l++ {
		if l != 0 {
			delete(cnt, s[l-1])
		}

		for r+1 < len(s) && cnt[s[r+1]] == 0 {
			cnt[s[r+1]]++
			r++
		}

		ans = max(ans, r-l+1)
	}
	return ans
}

// 给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号

// 最重要的点是：格式正确且连续：关键词在于连续
// 当出现“...((...))”这种情况的时候分为两种 -- b',a',...,a,b
// i-dp[i-1] 的值是否大于2 --> b和b'之间是否是连续的合格的子串
func longestValidParentheses(s string) int {
	ans := 0
	dp := make([]int, len(s))
	for i := 1; i < len(s); i++ {
		// 只有s[i] == ')'更新dp
		if s[i] == '(' {
			continue
		}

		if s[i-1] == '(' {
			if i >= 2 {
				dp[i] = dp[i-2] + 2
			} else {
				dp[i] = 2
			}
		} else if i-dp[i-1] > 0 && s[i-dp[i-1]-1] == '(' {
			if i-dp[i-1] >= 2 {
				dp[i] = dp[i-1] + dp[i-dp[i-1]-2] + 2
			} else {
				// i-1之前都是连续的
				dp[i] = dp[i-1] + 2
			}

		}
		ans = max(ans, dp[i])
	}
	return ans
}

// 给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

// 求在该柱状图中，能够勾勒出来的矩形的最大面积。
// 题解：一定要找到i的左右两边小于heights[i]的坐标，左侧最小的哨兵是-1,右侧的哨兵是n
// 设置哨兵是为了满足最小值是宽度是 n + 1 -1 :等于左边的宽度
func largestRectangleArea(heights []int) int {
	n := len(heights)
	if n == 0 {
		return 0
	}

	// 定义变量
	left, right := make([]int, n), make([]int, n)
	stack := []int{}

	// 左侧遍历
	for i := 0; i < n; i++ {
		for len(stack) > 0 && heights[stack[len(stack)-1]] > heights[i] {
			// 维持i左侧的最小的栈
			stack = stack[:len(stack)-1]
		}
		if len(stack) == 0 {
			// 最小值是自己本身，-1可以理解为可以直接延伸到到，因为左边没有比他更小的
			// 不会出现高度不够的情况
			left[i] = -1
		} else {
			left[i] = stack[len(stack)-1]
		}

		stack = append(stack, i)
	}

	// 还原单调栈
	// 右侧遍历
	stack = []int{}
	for i := n - 1; i >= 0; i-- {
		for len(stack) > 0 && heights[stack[len(stack)-1]] >= heights[i] {
			// 维持i右侧的最小的栈
			stack = stack[:len(stack)-1]
		}

		if len(stack) == 0 {
			// 本身是右侧的最小值
			// 可以延伸到数组末端
			right[i] = n
		} else {
			right[i] = stack[len(stack)-1]
		}

		stack = append(stack, i)
	}

	// 左侧和右侧的最小值已经到位的情况下
	ans := 0
	for i := 0; i < n; i++ {
		ans = max(ans, (right[i]-left[i]-1)*heights[i])
	}
	return ans
}
