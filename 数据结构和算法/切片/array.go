package arrary

import "sort"

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

// 已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,2,4,5,6,7] 在变化后可能得到：
// 若旋转 4 次，则可以得到 [4,5,6,7,0,1,2]
// 若旋转 7 次，则可以得到 [0,1,2,4,5,6,7]
// 注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。

// 给你一个元素值 互不相同 的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。

// 你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。
func findMin(nums []int) int {
	low, high := 0, len(nums)-1
	for low < high {
		privot := low + (high-low)/2
		if nums[privot] < nums[high] {
			high = privot
		} else {
			low = privot + 1
		}
	}
	return nums[low]
}

// 合并区间
// 以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。
// 请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。

func merge(intervals [][]int) [][]int {
	// 讲数组按照左边界排序，可以直接使用sort的算法
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})

	// end < start ：表示不重合，则加入区间
	// end >= start ： 表示重合，则定义左右边界，确定下一次的循环，知道最后确定右边界
	res := make([][]int, 0, len(intervals))
	left, right := intervals[0][0], intervals[0][1]
	for i := 1; i < len(intervals); i++ {
		if right < intervals[i][0] {
			res = append(res, []int{left, right})
			left, right = intervals[i][0], intervals[i][1]
		} else {
			right = max(intervals[i][1], right)
		}
	}

	// 可以直接考虑数组长度等于1的情况
	res = append(res, []int{left, right})
	return res
}

// 最大值
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

// 最小值
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

// 给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。
// 示例 1:
// 输入: nums = [1,2,3,4,5,6,7], k = 3
// 输出: [5,6,7,1,2,3,4]
// 解释:
// 向右轮转 1 步: [7,1,2,3,4,5,6]
// 向右轮转 2 步: [6,7,1,2,3,4,5]
// 向右轮转 3 步: [5,6,7,1,2,3,4]
func rotateArray(nums []int, k int) {
	// 辅助数组
	newNums := make([]int, len(nums))
	for i, v := range nums {
		// 根据规律计算的
		newNums[(i+k)%len(nums)] = v
	}
	// 数组是可以直接赋值的，但是切片不能直接等于
	// nums = newNums 这个是错误的
	copy(nums, newNums)
}

// 利用最大公约数求解问题，数组之间的替换
func rotate1(nums []int, k int) {
	n := len(nums)

	// count 是最大调换次数
	for start, count := 0, gcb(k, n); start < count; start++ {
		pre, cur := nums[start], start
		// 一次调换
		for ok := true; ok; ok = cur != start {
			next := (cur + k) % n
			pre, nums[next], cur = nums[next], pre, next
		}
	}
}

// a和b的最大公约数
func gcb(a, b int) int {
	for a != 0 {
		a, b = b%a, a
	}
	return b
}

// a和b的最小公倍数:最小公倍数=两数之积/最大公约数
func lcm(a, b int) int {
	return (a * b) / gcb(a, b)
}

// 给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。
// step1: 记录首位是否是0，如果是0，则是true
// 注意点：（第一行和第一列被修改，无法记录它们是否原本包含 000。因此我们需要额外使用两个标记变量分别记录第一行和第一列是否原本包含 000。）
// step2：将元素所在列的行行首和列首改为0
// step3: 遍历整个数组，从1开始遍历，更改元素值为0
func setZeroes(matrix [][]int) {
	n, m := len(matrix), len(matrix[0])
	col0 := false
	for _, r := range matrix {
		if r[0] == 0 {
			col0 = true
		}
		for j := 1; j < m; j++ {
			if r[j] == 0 {
				r[0] = 0
				matrix[0][j] = 0
			}
		}
	}

	for i := n - 1; i >= 0; i-- {
		for j := 1; j < m; j++ {
			// 行首和列首为0，则表示为0
			if matrix[i][0] == 0 || matrix[0][j] == 0 {
				matrix[i][j] = 0
			}
		}

		if col0 {
			matrix[i][0] = 0
		}
	}
}

// 给你一个满足下述两条属性的 m x n 整数矩阵：
// 每行中的整数从左到右按非严格递增顺序排列。
// 每行的第一个整数大于前一行的最后一个整数。
// 给你一个整数 target ，如果 target 在矩阵中，返回 true ；否则，返回 false 。
// 这个题的解法是似曾相识，可以使用二分查找，也可以使用右上角或者左下角的元素
// 因为需要做加减法
func searchMatrix(matrix [][]int, target int) bool {
	if len(matrix) == 0 {
		return false
	}
	m, n := len(matrix), len(matrix[0])
	x, y := 0, n-1
	for x < m && y >= 0 {
		if matrix[x][y] == target {
			return true
		}
		if matrix[x][y] > target {
			y--
		} else {
			x++
		}
	}
	return false
}

// 给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。
// 你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。
// 可以通过观察得到，旋转数组，可以先水平翻转，之后在对角线翻转
// 水平轴翻转  matrix[row][col]= matrix[n−row−1][col]
// 主对角线翻转 matrix[row][col] = matrix[col][row]
// 但是上述的技巧只适合于n*n的矩阵
func rotate(matrix [][]int) {
	n := len(matrix)
	// 水平翻转
	for i := 0; i < n/2; i++ {
		// 只需要将i对应的数组交换，因为列坐标不变
		matrix[i], matrix[n-i-1] = matrix[n-i-1], matrix[i]
	}

	// 对角线
	for i := 0; i < n; i++ {
		for j := 0; j < i; j++ {
			// 对角线下方的和对角线上方的做交换
			matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
		}
	}
}

func rotateAuxiliaryArry(matrix [][]int) {
	n := len(matrix)
	tmp := make([][]int, n)
	for i := range tmp {
		tmp[i] = make([]int, n)
	}
	// 根据规律可以知道，旋转之后的公式为：matrix[row][col] == matrixnew[col][n−row−1]
	for i, row := range matrix {
		for j, v := range row {
			tmp[j][n-1-i] = v
		}
	}
	copy(matrix, tmp) // 拷贝 tmp 矩阵每行的引用
}
