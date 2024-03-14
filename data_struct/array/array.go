package array

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

// 给定一个包含红色、白色和蓝色、共 n 个元素的数组 nums ，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
// 我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。
// 必须在不使用库内置的 sort 函数的情况下解决这个问题。

// 题解：对于这个问题可以理解为排序，但是这个是利用了数组中只有三类元素，快排是可以考虑的
func sortColors(nums []int) {
	p0, p1 := 0, 0
	for i, c := range nums {
		if c == 0 {
			nums[p0], nums[i] = nums[i], nums[p0]
			if p0 < p1 {
				nums[p1], nums[i] = nums[i], nums[p1]
			}
			p0++
			p1++
		} else if c == 1 {
			nums[p1], nums[i] = nums[i], nums[p1]
			p1++
		}
	}
}

// 给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。
// 请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
// 你必须设计并实现时间复杂度为 O(n) 的算法解决此问题。
// 题解：只排序前k的数组
func findKthLargest(nums []int, k int) int {
	return quickSelect(nums, 0, len(nums)-1, len(nums)-k)
}

func quickSelect(nums []int, l, r, k int) int {
	if l == r {
		return nums[k]
	}
	position := nums[l]
	i := l - 1
	j := r + 1
	for i < j {
		for i++; nums[i] < position; i++ {
		}
		for j--; nums[j] > position; j-- {
		}
		// postion左边的元素小于position，右边的元素大于position
		if i < j {
			nums[i], nums[j] = nums[j], nums[i]
		}
	}

	// 判断k的位置
	if k <= j {
		return quickSelect(nums, l, j, k)
	} else {
		return quickSelect(nums, j+1, r, k)
	}
}

// 给定一个包含 n + 1 个整数的数组 nums ，其数字都在 [1, n] 范围内（包括 1 和 n），可知至少存在一个重复的整数。
// 假设 nums 只有 一个重复的整数 ，返回 这个重复的数 。
// 你设计的解决方案必须 不修改 数组 nums 且只用常量级 O(1) 的额外空间。
// 题解：这个题目主要是关于快慢指针，判断是否是回环链表
// a:表示链表头到环节点的,b是相遇点,c是相遇点到相交点的距离
// 2(a+b)=a+b+kL --- > a=kL−b   --->a=(k−1)L+(L−b)=(k−1)L+c
func findDuplicate(nums []int) int {
	slow, fast := 0, 0
	for slow, fast = nums[slow], nums[nums[fast]]; slow != fast; slow, fast = nums[slow], nums[nums[fast]] {
	}
	slow = 0
	for slow != fast {
		slow = nums[slow]
		fast = nums[fast]
	}
	return slow
}

// 整数数组的一个 排列  就是将其所有成员以序列或线性顺序排列。
// 例如，arr = [1,2,3] ，以下这些都可以视作 arr 的排列：[1,2,3]、[1,3,2]、[3,1,2]、[2,3,1] 。
// 整数数组的 下一个排列 是指其整数的下一个字典序更大的排列。更正式地，如果数组的所有排列根据其字典顺序从小到大排列在一个容器中，
// 那么数组的 下一个排列 就是在这个有序容器中排在它后面的那个排列。如果不存在下一个更大的排列，
// 那么这个数组必须重排为字典序最小的排列（即，其元素按升序排列）。
// 例如，arr = [1,2,3] 的下一个排列是 [1,3,2] 。
// 类似地，arr = [2,3,1] 的下一个排列是 [3,1,2] 。
// 而 arr = [3,2,1] 的下一个排列是 [1,2,3] ，因为 [3,2,1] 不存在一个字典序更大的排列。
// 给你一个整数数组 nums ，找出 nums 的下一个排列。
// 必须 原地 修改，只允许使用额外常数空间。
func nextPermutation(nums []int) {
	if len(nums) == 0 {
		return
	}

	i, j, k := len(nums)-2, len(nums)-1, len(nums)-1
	for i >= 0 && nums[i] >= nums[j] {
		i--
		j--
	}

	// i不是第一个值
	if i >= 0 {
		for nums[i] >= nums[k] {
			k--
		}
		// 较大值和较小值的替换
		nums[i], nums[k] = nums[k], nums[i]
	}

	// 后半段升序
	for i, j := j, len(nums)-1; i < j; i, j = i+1, j-1 {
		nums[i], nums[j] = nums[j], nums[i]
	}
}

// 给定一个长度为 n 的 0 索引整数数组 nums。初始位置为 nums[0]。
// 每个元素 nums[i] 表示从索引 i 向前跳转的最大长度。换句话说，如果你在 nums[i] 处，你可以跳转到任意 nums[i + j] 处:
// 0 <= j <= nums[i]
// i + j < n
// 返回到达 nums[n - 1] 的最小跳跃次数。生成的测试用例可以到达 nums[n - 1]。
// 题解：记录跳跃过程中的最大距离，如果大于最大距离可以直接跳跃过去，因为在i到end中随机产生的最大距离，都比end的坐标大
// 所以将maxpostion赋值给end是合理的
func jump(nums []int) int {
	// 长度
	length := len(nums)
	// 步数
	step := 0
	// 最大距离
	maxPosition := 0
	// 哪个坐标达到最大距离
	end := 0
	// 不访问最后一个元素，因为在访问最后一个元素之前
	// 边界一定会大于等于这个元素，否则就跳不到这个位置了，会增加额外的跳跃次数
	for i := 0; i < length-1; i++ {
		// 比较当前位置和最大距离的区别
		maxPosition = max(maxPosition, i+nums[i])
		if i == end {
			end = maxPosition
			step++
		}
	}
	return step
}

// 给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
// 题解：双指针永远更新临近的柱状图的大小
// 还有一种算法是记录右侧的最高值，和左侧的最高值做比较，取最小的，更新i的值
// 其实比较难理解的是为什么取一侧就可以了，极端思路left和right中间的都小，那么其实水的最大深度也就是left和right的最小值
// 因为两侧都有木板阻止水溜走
func trap(height []int) int {
	// 定义双指针以及左侧最大值和右侧最大值
	max_left := 0
	max_right := 0
	left := 1
	right := len(height) - 2
	ans := 0
	// 关键点，max_left=max(max_left,height[i-1])
	// max_right=max(max_right,heigh[j+1])
	// 所以可以得出当height[j+1]>height[i-1]==》max_right>max_left
	// 取左右两侧最大值的最小值
	for i := 1; i < len(height)-1; i++ {
		// max_left<max_right
		// 从左到右
		if height[left-1] < height[right+1] {
			max_left = max(max_left, height[left-1])
			if max_left > height[left] {
				ans += max_left - height[left]
			}
			left++
		} else {
			// 从右到左
			max_right = max(max_right, height[right+1])
			if max_right > height[right] {
				ans += max_right - height[right]
			}
			right--
		}
	}
	return ans
}

// 给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
// 算法的时间复杂度应该为 O(log (m+n)) 。
// 题解：合并数组，求解中位数
func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	arrays := mergeArray(nums1, nums2)
	if len(arrays)%2 == 0 {
		left := float64(arrays[len(arrays)/2-1])
		right := float64(arrays[len(arrays)/2])
		return (left + right) / 2
	}
	return float64(arrays[(len(arrays)-1)/2])
}

func mergeArray(nums1, nums2 []int) []int {
	if len(nums1) == 0 {
		return nums2
	}

	if len(nums2) == 0 {
		return nums1
	}

	ans := []int{}
	first, second := 0, 0
	for first < len(nums1) && second < len(nums2) {
		if nums1[first] < nums2[second] {
			ans = append(ans, nums1[first])
			first++
		} else {
			ans = append(ans, nums2[second])
			second++
		}
	}

	if first == len(nums1) {
		ans = append(ans, nums2[second:]...)
	}

	if second == len(nums2) {
		ans = append(ans, nums1[first:]...)
	}
	return ans
}

// 按照国际象棋的规则，皇后可以攻击与之处在同一行或同一列或同一斜线上的棋子。

// n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。

// 给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。

// 每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。
// 题解，回溯---定义变量和临时数组，初始化，确定回溯函数，恢复原始状态
var solutions [][]string

func solveNQueens(n int) [][]string {
	solutions = [][]string{}
	// 初始化
	queens := make([]int, n)
	for i := 0; i < n; i++ {
		queens[i] = -1
	}
	columns, d1s, d2s := map[int]bool{}, map[int]bool{}, map[int]bool{}
	backtrack1(queens, n, 0, columns, d1s, d2s)
	return solutions
}

// 定义回溯函数,关键点在于queenes[row] = i -->表示第row的第i列
func backtrack1(queens []int, n, row int, columns, diagonals1, diagonals2 map[int]bool) {
	// 结束终止条件
	if row == n {
		board := generateBoard(queens, n)
		solutions = append(solutions, board)
		return
	}

	for i := 0; i < n; i++ {
		if columns[i] {
			continue
		}

		// 左对角线，行下标和列下表之差相等
		diagonal1 := row - i
		if diagonals1[diagonal1] {
			continue
		}

		// 右对角线, 行下标和列小标之和相等
		diagonal2 := row + i
		if diagonals2[diagonal2] {
			continue
		}
		// 满足条件可以放置
		queens[row] = i
		columns[i] = true
		diagonals1[diagonal1], diagonals2[diagonal2] = true, true
		backtrack1(queens, n, row+1, columns, diagonals1, diagonals2)

		// 复原
		queens[row] = -1
		delete(columns, i)
		delete(diagonals1, diagonal1)
		delete(diagonals2, diagonal2)
	}
}

// 生成board
func generateBoard(queens []int, n int) []string {
	board := []string{}
	// 根据queens的数组生成对应的
	for i := 0; i < n; i++ {
		row := make([]byte, n)
		for j := 0; j < n; j++ {
			row[j] = '.'
		}
		// 每一行都会放置一个皇后，所以初始值是-1，之后会更新对应的值的，
		// 所以走完回溯的Queen的row对应的值一定是对应防止皇后的列
		row[queens[i]] = 'Q'
		board = append(board, string(row))
	}
	return board
}

// 给你一个下标从 0 开始、由正整数组成的数组 nums 。
// 你可以在数组上执行下述操作 任意 次：
// 选中一个同时满足 0 <= i < nums.length - 1 和 nums[i] <= nums[i + 1] 的整数 i 。将元素 nums[i + 1] 替换为 nums[i] + nums[i + 1] ，并从数组中删除元素 nums[i] 。
// 返回你可以从最终数组中获得的 最大 元素的值。
// 观察合并的结果，前面的<= 后面的，后面如果小的话
// 如果前面一个元素的值>后面的元素，则后面的元素必然不会参与合并
func maxArrayValue(nums []int) int64 {
	sum := int64(nums[len(nums)-1])
	for i := len(nums) - 2; i >= 0; i-- {
		if int64(nums[i]) <= sum {
			sum += int64(nums[i])
		} else {
			sum = int64(nums[i])
		}
	}
	return sum
}
