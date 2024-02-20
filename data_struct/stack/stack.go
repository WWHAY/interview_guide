package stack

import "math"

// 给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，
// 其中 answer[i] 是指对于第 i 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，
// 请在该位置用 0 来代替。
// 题解：这个栈用来维护的温度是从高到低的index单调栈，因为日子只能往前看，所以栈是可以按照天数去算的
// 气温的高低可以通过栈顶元素的判断去看的
// 最后一天的元素是默认为0
func dailyTemperatures(temperatures []int) []int {
	ans := make([]int, len(temperatures))
	stack := []int{}
	for i := 0; i < len(temperatures); i++ {
		temperature := temperatures[i]
		for len(stack) > 0 && temperature > temperatures[stack[len(stack)-1]] {
			preIndex := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			ans[preIndex] = i - preIndex
		}
		stack = append(stack, i)
	}
	return ans
}

// 设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。
// 实现 MinStack 类:
// MinStack() 初始化堆栈对象。
// void push(int val) 将元素val推入堆栈。
// void pop() 删除堆栈顶部的元素。
// int top() 获取堆栈顶部的元素。
// int getMin() 获取堆栈中的最小元素。
// 题解：辅助站，最小栈和栈对应，最小栈里面对应着当前元素的最小值
type MinStack struct {
	MinStack []int
	stack    []int
}

func Constructor() MinStack {
	return MinStack{
		stack:    []int{},
		MinStack: []int{math.MaxInt64},
	}
}

func (this *MinStack) Push(val int) {
	this.stack = append(this.stack, val)
	top := this.MinStack[len(this.MinStack)-1]
	this.MinStack = append(this.MinStack, min(top, val))
}

func (this *MinStack) Pop() {
	this.stack = this.stack[:len(this.stack)-1]
	this.MinStack = this.MinStack[:len(this.MinStack)-1]
}

func (this *MinStack) Top() int {
	return this.stack[len(this.stack)-1]
}

func (this *MinStack) GetMin() int {
	return this.MinStack[len(this.MinStack)-1]
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}
