package greedy

// 给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。
// 你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
// 返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。
// 题解，在第i天卖出股票，最大的利润是前i-1天最低买入的值
func maxProfit(prices []int) int {
	if len(prices) == 0 {
		return 0
	}

	minPrice := prices[0]
	ans := 0
	for _, price := range prices {
		ans = max(ans, price-minPrice)
		minPrice = min(minPrice, price)
	}
	return ans
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}
