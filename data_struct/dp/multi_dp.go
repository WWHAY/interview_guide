package dp

import "math"

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
