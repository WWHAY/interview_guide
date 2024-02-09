package arrary

// 查找固定的值
// 给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
// 请必须使用时间复杂度为 O(log n) 的算法。
func searchInsert(nums []int, target int) int {
	if len(nums) == 0 {
		return 0
	}
	left, right := 0, len(nums)-1
	ans := len(nums)
	for left <= right {
		mid := (left + right) / 2
		if nums[mid] < target {
			left = mid + 1
		}

		if nums[mid] >= target {
			ans = mid
			right = mid - 1
		}
	}
	return ans
}

// 给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。
// 如果数组中不存在目标值 target，返回 [-1, -1]。
// 你必须设计并实现时间复杂度为 O(log n) 的算法解决此问题。

// 哈希表：空间换时间，n
func searchRange(nums []int, target int) []int {
	numsMap := make(map[int][]int, 0)
	for k, v := range nums {
		numsMap[v] = append(numsMap[v], k)
	}
	ans := []int{-1, -1}
	if v, ok := numsMap[target]; ok {
		ans[0] = v[0]
		ans[1] = v[len(v)-1]
	}
	return ans
}

func searchRange1(nums []int, target int) []int {
	ans := []int{-1, -1}
	left := binarySearch(nums, target, true)
	right := binarySearch(nums, target, false) - 1
	// 因为判断的条件是第一个>= target 和 第一个>target的值-1
	// 但是并不能保证left和right对应的值都是target，所以还是需要验证一下
	if left <= right && right < len(nums) && nums[left] == target && nums[right] == target {
		ans[0] = left
		ans[1] = right
	}
	return ans
}

// 二分查找：最关键是一定要想清楚对应的赋值；
func binarySearch(nums []int, target int, lower bool) int {
	// 定义双指针
	left, right := 0, len(nums)-1
	// 设置默认
	ans := len(nums)
	// 二分查找
	for left <= right {
		// 判断是左侧还是右侧
		mid := (left + right) / 2
		if nums[mid] > target || (lower && nums[mid] >= target) {
			right = mid - 1
			// 不断更新ans，因为第一次的值不一定是符合条件的，最后一个一半是符合条件的
			// 第一个大于等于target == 第一个等于target
			// 第一个大于target == 右半段
			ans = mid
		} else {
			left = mid + 1
		}
	}
	return ans
}
