package heap

import "container/heap"

// 给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。
// 题解：实现一个堆的interface，即是实现了这个堆，golang中有堆函数，可以直接套用
// 最大堆，value是值，但是会对应一个key，所以需要使用断言的方式
func topKFrequent(nums []int, k int) []int {
	occurences := map[int]int{}
	// 定义元素出现的次数
	for _, v := range nums {
		occurences[v]++
	}
	h := &IHeap{}
	heap.Init(h) // 实现了堆的函数
	for key, value := range occurences {
		// 推入
		heap.Push(h, [2]int{key, value})
		if h.Len() > k {
			heap.Pop(h)
		}
	}
	ret := make([]int, k)
	for i := 0; i < k; i++ {
		ret[k-i-1] = heap.Pop(h).([2]int)[0]
	}
	return ret
}

// 一维记录堆的位置，二维记录key和value，也即是数字对应出现的次数
type IHeap [][2]int

// 堆的长度
func (h IHeap) Len() int {
	return len(h)
}

// 小于
func (h IHeap) Less(i, j int) bool {
	return h[i][1] < h[j][1]
}

// push
func (h *IHeap) Push(v interface{}) {
	*h = append(*h, v.([2]int)) // 断言
}

// pop 弹出来第一个元素，栈
func (h *IHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// 交换
func (h IHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

// 给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。
// 请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
// 你必须设计并实现时间复杂度为 O(n) 的算法解决此问题。
// 题解：大顶堆，前k个数
func findKthLargest(nums []int, k int) int {
	heapSize := len(nums)
	// 创建大顶堆
	buildMaxHeap(nums, heapSize)
	// 删除前k个元素
	for i := len(nums) - 1; i >= len(nums)-k+1; i-- {
		nums[0], nums[i] = nums[i], nums[0]
		heapSize--
		maxHeap(nums, 0, heapSize)
	}
	// 已经删除了前k个元素了，所以堆顶的元素即是我们要求的值
	return nums[0]
}

func buildMaxHeap(nums []int, heapSize int) {
	for i := heapSize / 2; i >= 0; i-- {
		maxHeap(nums, i, heapSize)
	}
}

// 实现大顶堆
func maxHeap(nums []int, i, heapSize int) {
	// 根据大顶堆的父子节点的定义，去定义父子节点
	l, r, largest := 2*i+1, 2*i+2, i
	// 找出最大值
	if l < heapSize && nums[l] > nums[largest] {
		largest = l
	}

	if r < heapSize && nums[r] > nums[largest] {
		largest = r
	}

	// 交换节点
	if largest != i {
		nums[largest], nums[i] = nums[i], nums[largest]
		maxHeap(nums, largest, heapSize)
	}
}
