package link

// 请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
// 实现 LRUCache 类：
// LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
// int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
// void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。
// 函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。

// 实现方式： 哈希+双向链表
// 涉及到O(1)级别获取key的内容
// 双向链表便于插入和删除
type DLinkNode struct {
	Key  int
	Val  int
	Pre  *DLinkNode
	Next *DLinkNode
}

// 头尾指针便于插入元素和初始化列表
type LRUCache struct {
	size     int
	capacity int
	cache    map[int]*DLinkNode
	head     *DLinkNode
	tail     *DLinkNode
}

func InitDLinkNode(key, value int) *DLinkNode {
	return &DLinkNode{
		Key: key,
		Val: value,
	}
}

func Constructor(capacity int) LRUCache {
	l := LRUCache{
		capacity: capacity,
		cache:    map[int]*DLinkNode{},
		head:     InitDLinkNode(0, 0),
		tail:     InitDLinkNode(0, 0),
	}
	l.head.Next = l.tail
	l.tail.Pre = l.head
	return l
}

func (this *LRUCache) Get(key int) int {
	node, ok := this.cache[key]
	if !ok {
		return -1
	}
	this.moveToHead(node)
	return node.Val
}

func (this *LRUCache) Put(key int, value int) {
	if node, ok := this.cache[key]; ok {
		node.Val = value
		this.moveToHead(node)
	} else {
		node := InitDLinkNode(key, value)
		this.addToHead(node)
		this.cache[key] = node
		this.size++
		if this.size > this.capacity {
			tail := this.removeTail()
			delete(this.cache, tail.Key)
			this.size--
		}
	}
}

func (this *LRUCache) addToHead(node *DLinkNode) {
	node.Pre = this.head
	node.Next = this.head.Next
	this.head.Next.Pre = node
	this.head.Next = node
}

func (this *LRUCache) removeTail() *DLinkNode {
	node := this.tail.Pre
	this.removeNode(node)
	return node
}

func (this *LRUCache) moveToHead(node *DLinkNode) {
	// 先删除，再加入
	this.removeNode(node)
	this.addToHead(node)
}

// 为了删除map的数值
func (this *LRUCache) removeNode(node *DLinkNode) {
	node.Pre.Next = node.Next
	node.Next.Pre = node.Pre
	node.Next = nil
}
