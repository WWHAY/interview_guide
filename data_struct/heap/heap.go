package heap

import (
	"container/heap"
	"sort"
)

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

// 中位数是有序整数列表中的中间值。如果列表的大小是偶数，则没有中间值，中位数是两个中间值的平均值。

// 例如 arr = [2,3,4] 的中位数是 3 。
// 例如 arr = [2,3] 的中位数是 (2 + 3) / 2 = 2.5 。
// 实现 MedianFinder 类:

// MedianFinder() 初始化 MedianFinder 对象。

// void addNum(int num) 将数据流中的整数 num 添加到数据结构中。

// double findMedian() 返回到目前为止所有元素的中位数。与实际答案相差 10-5 以内的答案将被接受。
// 题解：实现一个堆的push和pop，默认是最小堆，pop弹出来的是最小的元素，对于minQ和maxQ的话，maxQ天然是递增的顺序，minQ需要加入负数
// 1，2,3,4,5 -- minQ(-3,-2,-1) --- 才能保证最大值能够被弹出来，maxQ（4,5）-- 天然是递增的，可以保证最小值被弹出来
// initSlice是递增排列，对于heap也是的
type MedianFinder struct {
	minQ, maxQ hp
}

func Constructor() MedianFinder {
	return MedianFinder{}
}

func (this *MedianFinder) AddNum(num int) {
	min, max := &this.minQ, &this.maxQ
	if min.Len() == 0 || -min.IntSlice[0] >= num {
		// 将数值加入到minQ
		heap.Push(min, -num)
		// 奇数变偶数
		if max.Len()+1 < min.Len() {
			heap.Push(max, -heap.Pop(min).(int))
		}
	} else {
		// 将数值加入到maxQ中，同时检查长度看是否将maxQ的最小值加入到minQ中
		heap.Push(max, num)
		if max.Len() > min.Len() {
			heap.Push(min, -heap.Pop(max).(int))
		}
	}
}

func (this *MedianFinder) FindMedian() float64 {
	min, max := this.minQ, this.maxQ
	if min.Len() == max.Len() {
		return float64(max.IntSlice[0]-min.IntSlice[0]) / 2
	}
	return float64(-min.IntSlice[0])
}

type hp struct {
	sort.IntSlice
}

// 递减的顺序排列 --- 堆
func (h *hp) Push(v interface{}) {
	h.IntSlice = append(h.IntSlice, v.(int))
}

// 堆是最小堆 -- 堆顶元素是最小值
func (h *hp) Pop() interface{} {
	a := h.IntSlice
	v := a[len(a)-1]
	h.IntSlice = a[:len(a)-1]
	return v
}

/**
 * Your MedianFinder object will be instantiated and called as such:
 * obj := Constructor();
 * obj.AddNum(num);
 * param_2 := obj.FindMedian();
 */
