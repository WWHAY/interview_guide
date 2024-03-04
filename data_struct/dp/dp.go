package dp

// 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
// 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
// f(x)=f(x−1)+f(x−2) 这个公式就是动态规划的通项公式
func climbStairs(n int) int {
	p, q, r := 0, 0, 1
	for i := 1; i <= n; i++ {
		p = q
		q = r
		r = p + q
	}
	return r
}

// 给定一个非负整数 numRows，生成「杨辉三角」的前 numRows 行。
// 在「杨辉三角」中，每个数是它左上方和右上方的数的和。
func generate(numRows int) [][]int {
	ans := make([][]int, numRows)
	for i := 0; i < numRows; i++ {
		ans[i] = make([]int, i+1)
		ans[i][0] = 1
		ans[i][i] = 1
		// j < i:直接排除了i=0和i=1的情况，因为j是从1开始的
		for j := 1; j < i; j++ {
			ans[i][j] = ans[i-1][j-1] + ans[i-1][j]
		}
	}
	return ans
}

// 给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
// 子数组 是数组中的一个连续部分。

// 最大连续子数组之和的解法解析：DP
// f(i)代表以i结尾的最大连续子数组之和，可以考虑nums[i]是单独成为一段，还是加入到f(i)中，
// 取决于f(i) = max{nums[i]+nums[i-1],nums[i]}，判断在i处是否让该值降低
func maxSubArray(nums []int) int {
	max := nums[0]
	for i := 1; i < len(nums); i++ {
		// nums中i的值代表着是前面数组之和,
		// 大小判断取决于数组是否是连续的
		if nums[i]+nums[i-1] > nums[i] {
			nums[i] += nums[i-1]
		}
		if nums[i] > max {
			max = nums[i]
		}
	}

	return max
}

// 给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
// 状态定义：dp[i][j]表示从数组的 [0, i] 这个子区间内挑选一些正整数，每个数只能用一次，使得这些数的和恰好等于 j。
// 状态转移方程：很多时候，状态转移方程思考的角度是「分类讨论」，对于「0-1 背包问题」而言就是「当前考虑到的数字选与不选」。
// 不选择 nums[i]，如果在 [0, i - 1] 这个子区间内已经有一部分元素，使得它们的和为 j ，那么 dp[i][j] = true；
// 选择 nums[i]，如果在 [0, i - 1] 这个子区间内就得找到一部分元素，使得它们的和为 j - nums[i],那么dp[i][j-nums[i]] = true。
// 题解：最为关键的一部分在于新一轮的结果只和上一轮的结果有关系，所以可以覆盖
// j是递减的，如果nums[i]>j的话，其实没有必要继续往下进行了；
func canPartition(nums []int) bool {
	// 计算目标值
	target := 0
	for _, v := range nums {
		target += v
	}

	// 和必须是偶数
	if target%2 == 1 {
		return false
	}

	// map：key指的是j也即是【0，i】中是否存在j的和
	dp := make(map[int]bool, 0)
	if nums[0] <= target {
		dp[nums[0]] = true
	}

	// 动态规划
	target = target / 2
	for i := 1; i < len(nums); i++ {
		// dp[j-nums[i]]
		for j := target; nums[i] <= j; j-- {
			if dp[target] {
				return true
			}
			// 基于上一轮的结果得到新的一轮的值
			dp[j] = dp[j] || dp[j-nums[i]]
		}
	}
	return dp[target]
}

// 给你一个字符串 s 和一个字符串列表 wordDict 作为字典。如果可以利用字典中出现的一个或多个单词拼接出 s 则返回 true。
// 注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。
func wordBreak(s string, wordDict []string) bool {
	wordListMap := make(map[string]bool, len(wordDict))
	for _, v := range wordDict {
		wordListMap[v] = true
	}

	dp := make([]bool, len(s)+1)
	dp[0] = true //空字符串成立
	// [0,j] 成立&& [j,i]成立，则i成立
	for i := 1; i <= len(s); i++ {
		for j := 0; j < i; j++ {
			// 分割点前和分割点的单词都存在，那么存在
			if dp[j] && wordListMap[s[j:i]] {
				dp[i] = true
				break
			}
		}
	}
	return dp[len(s)]
}
