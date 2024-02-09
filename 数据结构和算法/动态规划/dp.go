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
