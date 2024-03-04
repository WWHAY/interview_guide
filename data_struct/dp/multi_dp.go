package dp

import (
	"math"
	"sort"
)

// 多维数组

// 给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。
// 计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。
// 你可以认为每种硬币的数量是无限的。
// 题解：遍历顺序和组合数和排列数有关的时候需要确定内部和外部循环的顺序
// 求从coin[i]开始，然后后面的值从前的值获取结果
func coinChange(coins []int, amount int) int {
	// 定义动态规划的数组
	// dp[j]：凑足总额为j所需钱币的最少个数为dp[j]
	dp := make([]int, amount+1)
	// 初始化数组
	for i := range dp {
		// 背包是0的时候，不需要兑换任何的零钱
		if i == 0 {
			continue
		}
		dp[i] = math.MaxInt
	}
	// 遍历：求钱币最小个数，那么钱币有顺序和没有顺序都可以，都不影响钱币的最小个数
	// 不求组合或者是排列就无所谓了
	// 遍历物品，各个部分的最小值
	for i := 0; i < len(coins); i++ {
		for j := coins[i]; j <= amount; j++ {
			// 背包中之前已经填充了物品了
			if dp[j-coins[i]] != math.MaxInt {
				dp[j] = min(dp[j], dp[j-coins[i]]+1)
			}
		}
	}
	if dp[amount] == math.MaxInt {
		return -1
	}
	return dp[amount]
}

// 给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。
// 完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。
// 例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。
// 题解：动态规划的方程式：f[i]=1+min(f[i-j*j]):根据平方的性质，j的取值范围在[1,n^1/2]之间
func numSquares(n int) int {
	// 初始化数组
	f := make([]int, n+1)
	for i := 1; i <= n; i++ {
		minn := math.MaxInt //初始值
		for j := 1; j*j <= i; j++ {
			minn = min(minn, f[i-j*j])
		}
		// 枚举前一个值的最小平和+1，即是本身的最小平方和
		f[i] = minn + 1
	}
	return f[n]
}

// 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，
// 影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，
// 如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

// 给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
// 题解：核心点在于要分析出来表达式：当数组长度等于0，直接退出；当i=0，取第一个元素，当数组长度大于等于2：dp[0]=nums[0]
// dp[1] = max(nums[0], nums[1])
// 通用表达式，一定要分清楚，是否偷第k个：dp[i] = max(dp[i-2]+nums[i], dp[i-1])
// 动态规划的核心问题就是一定要找出对应的表达式
func rob(nums []int) int {
	// 参数检验
	if len(nums) == 0 {
		return 0
	}

	// 只有一间房子
	if len(nums) == 1 {
		return nums[0]
	}

	// 数量超过两个以上
	dp := make([]int, len(nums))
	dp[0] = nums[0]
	dp[1] = max(nums[0], nums[1])
	// 超过两个
	for i := 2; i < len(nums); i++ {
		dp[i] = max(dp[i-2]+nums[i], dp[i-1])
	}
	return dp[len(nums)-1]
}

// 给你一个整数数组 nums ，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
// 测试用例的答案是一个 32-位 整数。
// 子数组 是数组的连续子序列。
// 题解：min维持住负数的最小值，防止有负数的产生，ans不断地更新最大值
func maxProduct(nums []int) int {
	if len(nums) == 0 {
		return 0
	}

	maxn, minn, ans := nums[0], nums[0], nums[0]
	for i := 1; i < len(nums); i++ {
		mx, mn := maxn, minn
		maxn = max(mx*nums[i], max(nums[i], mn*nums[i]))
		minn = min(mn*nums[i], min(nums[i], mx*nums[i]))
		ans = max(maxn, ans)
	}
	return ans
}

// 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。

// 子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。
// 例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
// 题解：坐标i前面的数值一定比nums[i]小，此时记录f[i]的最大值
func lengthOfLIS(nums []int) int {
	f := make([]int, len(nums))
	ans := 0
	for i, x := range nums {
		for j, y := range nums[:i] {
			if y < x {
				f[i] = max(f[i], f[j])
			}
		}
		f[i]++
		ans = max(ans, f[i])
	}
	return ans
}

// 题解：维护一个数组，遍历元素组，在初始数组中用searchInt的函数（表示元素的插入位置，如果返回值==数组长度，表示不存在>=该数组的存在）
// 如果返回值则代表插入位置，表示原数组元素中存在一个值可以插入到数组中
func lengthOfLIS1(nums []int) int {
	g := []int{}
	for _, x := range nums {
		j := sort.SearchInts(g, x)
		if j == len(g) { // >=x 的 g[j] 不存在
			g = append(g, x)
		} else {
			g[j] = x
		}
	}
	return len(g)

}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

func min(x, y int) int {
	if x > y {
		return y
	}
	return x
}

// 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
// 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
// 问总共有多少条不同的路径？
func uniquePaths(m int, n int) int {
	// i:走到第j个方块的路径数
	dp := make([]int, n)
	// 初始化数组
	for v := range dp {
		dp[v] = 1
	}

	// 递推公式：dp[i][j] = dp[i-1][j]+dp[i][j-1]
	// 优化之后：dp[j] += dp[j-1]
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			dp[j] += dp[j-1]
		}
	}

	return dp[n-1]
}

func longestCommonSubsequence(text1 string, text2 string) int {
	m, n := len(text1), len(text2)
	// dp数组比原数组多了一个空字符的存在，所以会在元字符的基础上+1
	dp := make([][]int, len(text1)+1)
	for v := range dp {
		dp[v] = make([]int, len(text2)+1)
	}

	for i, c1 := range text1 {
		for j, c2 := range text2 {
			if c1 == c2 {
				dp[i+1][j+1] = dp[i][j] + 1
			} else {
				dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
			}
		}
	}
	// 有考虑空字符的存在
	return dp[m][n]
}

// 给你一个字符串 s，找到 s 中最长的回文子串。
// 如果字符串的反序与原始字符串相同，则该字符串称为回文字符串。
func longestPalindrome(s string) string {
	length := len(s)
	// dp表示s[i:j]是否是回文字符串
	dp := make([][]bool, length)
	for i := range dp {
		dp[i] = make([]bool, length)
		dp[i][i] = true // 初始化，一个字符就是一个字符串
	}
	result := s[0:1] // 默认值，我在设置默认值的时候出现了问题
	// dp表达式,l也就是表示start和end之间的距离
	for l := 2; l <= length; l++ {
		for start := 0; start < length-l+1; start++ {
			end := start + l - 1
			if s[start] != s[end] {
				continue
			} else if l < 3 {
				dp[start][end] = true // 两个字符
			} else {
				dp[start][end] = dp[start+1][end-1]
			}
			distance := end - start + 1
			if dp[start][end] && distance > len(result) {
				result = s[start : end+1]
			}
		}
	}
	return result
}

// 给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数  。

// 你可以对一个单词进行如下三种操作：
// 插入一个字符
// 删除一个字符
// 替换一个字符
// 题解：dp[i][j] :word1中的i位置转到word2中的j位置所需要的最少次数
// 分为两种情况：1. 最后一个元素相等的话，dp[i][j] = dp[i-1][j-1]的值
// 2. 最后一个元素不相等：dp[i][j] = min(dp[i-1][j-1], min(dp[i-1][j], dp[i][j-1])) + 1
// 其中dp[i-1][j-1]表示替换；dp[i-1][j]:删除：表示前一个元素已经相等，此处的i属于多余的元素，删除；
// dp[i][j-1])表示插入（word1的当前元素和word2的上一个元素相等，次数需要插入）
func minDistance(word1 string, word2 string) int {
	m, n := len(word2), len(word1)
	// 定义dp
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, m+1)
	}

	// 初始化的值
	// 第一行
	for j := 0; j <= m; j++ {
		dp[0][j] = j
	}

	// 第一列
	for i := 0; i <= n; i++ {
		dp[i][0] = i
	}

	// 通用
	for i := 1; i < n+1; i++ {
		for j := 1; j < m+1; j++ {
			if word1[i-1] == word2[j-1] {
				dp[i][j] = dp[i-1][j-1] // 替换
			} else {
				dp[i][j] = min(dp[i-1][j-1], min(dp[i-1][j], dp[i][j-1])) + 1
			}
		}
	}

	return dp[n][m]
}
