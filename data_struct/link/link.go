package link

// 链表
type ListNode struct {
	Next *ListNode
	Val  int
}

// 将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。
func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	if list1 == nil && list2 == nil {
		return nil
	}
	head := &ListNode{}
	newHead := head
	for list1 != nil && list2 != nil {
		if list1.Val <= list2.Val {
			newHead.Next = list1
			list1 = list1.Next
		} else {
			newHead.Next = list2
			list2 = list2.Next
		}
		newHead = newHead.Next
	}
	if list1 != nil {
		newHead.Next = list1
	}

	if list2 != nil {
		newHead.Next = list2
	}
	return head.Next
}

func mergeTwoLists1(list1 *ListNode, list2 *ListNode) *ListNode {
	if list1 == nil {
		return list2
	}
	if list2 == nil {
		return list1
	}

	if list1.Val <= list2.Val {
		list1.Next = mergeTwoLists(list1.Next, list2)
		return list1
	} else {
		list2.Next = mergeTwoLists(list1, list2.Next)
		return list2
	}
}

// 给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
func reverseList(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}

	var pre *ListNode // 翻转链表的头部，这个是尾部，所以必须是nil
	cur := head
	for cur != nil {
		// 改变指针关系
		next := cur.Next
		cur.Next = pre
		pre = cur
		cur = next
	}
	return pre
}

// 递归翻转链表
func reverseList1(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	newHead := reverseList(head.Next)
	head.Next.Next = head
	head.Next = nil
	return newHead
}

// 相交链表
// 给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null 。
// 保证链表无环
func getIntersectionNode(headA, headB *ListNode) *ListNode {
	if headA == nil || headB == nil {
		return nil
	}
	linkMap := make(map[*ListNode]bool, 0)
	newHeadA := headA
	for newHeadA != nil {
		linkMap[newHeadA] = true
		newHeadA = newHeadA.Next
	}
	newHeadB := headB
	for newHeadB != nil {
		if _, ok := linkMap[newHeadB]; ok {
			return newHeadB
		}
		newHeadB = newHeadB.Next
	}
	return nil
}

// 快慢指针判断是否相交
func getIntersectionNode1(headA, headB *ListNode) *ListNode {
	if headA == nil || headB == nil {
		return nil
	}
	pa, pb := headA, headB
	// 不想交的话，pa=pb=nil
	// 相交，pa=pb=相交的节点，因为a+b+c= c+b+a
	for pa != pb {
		if pa == nil {
			pa = headB
		} else {
			pa = pa.Next
		}

		if pb == nil {
			pb = headA
		} else {
			pb = pb.Next
		}
	}

	return pa
}

// 给你一个链表的头节点 head ，判断链表中是否有环。
// 如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。
// 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。
// 注意：pos 不作为参数进行传递 。仅仅是为了标识链表的实际情况。
// 如果链表中存在环 ，则返回 true 。 否则，返回 false 。
func hasCycle(head *ListNode) bool {
	if head == nil || head.Next == nil {
		return false
	}

	slow, fast := head, head.Next
	for slow != fast {
		if fast == nil || fast.Next == nil {
			return false
		}

		slow = slow.Next
		// 核心问题是，龟兔赛跑：兔子一定要比乌龟多跑两个
		fast = fast.Next.Next
	}

	return true
}

// 给你一个单链表的头节点 head ，请你判断该链表是否为回文链表。如果是，返回 true ；否则，返回 false 。
func isPalindrome(head *ListNode) bool {
	if head == nil {
		return false
	}
	arrary := []int{}
	for head != nil {
		arrary = append(arrary, head.Val)
		head = head.Next
	}

	first, last := 0, len(arrary)-1
	for first <= last {
		if arrary[first] != arrary[last] {
			return false
		}
		first++
		last--
	}
	return true
}

func endOfFirstHalf(head *ListNode) *ListNode {
	fast := head
	slow := head
	for fast.Next != nil && fast.Next.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
	}
	return slow
}

func isPalindrome1(head *ListNode) bool {
	if head == nil {
		return true
	}

	// 找到前半部分链表的尾节点并反转后半部分链表
	firstHalfEnd := endOfFirstHalf(head)
	// 翻转后半程的连边
	secondHalfStart := reverseList(firstHalfEnd.Next)

	// 比对两边链接
	// 判断是否回文
	p1 := head
	p2 := secondHalfStart
	result := true
	for result && p2 != nil {
		if p1.Val != p2.Val {
			result = false
		}
		p1 = p1.Next
		p2 = p2.Next
	}

	// 还原链表并返回结果
	firstHalfEnd.Next = reverseList(secondHalfStart)
	return result
}

// 给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
// 如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。
// 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。
// 如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。
// 不允许修改 链表。

// a：表示起始位置到入环点；b表示入环点到相遇点的位置，c表示相遇点到入环点的位置
// 链表方向： 起始点 --> 入环点 --> 相遇点 ---> 入环点（此处回环）
// a+(n-1)(b+c) = 2(a+b)
// 快慢指针，最关键的是在于有了 a=c+(n−1)(b+c)a=c+(n-1)(b+c)a=c+(n−1)(b+c) 的等量关系
// 我们会发现：从相遇点到入环点的距离加上 n−1n-1n−1 圈的环长，恰好等于从链表头部到入环点的距离。

func detectCycle1(head *ListNode) *ListNode {
	slow, fast := head, head
	for fast != nil {
		slow = slow.Next
		if fast.Next == nil {
			return nil
		}
		fast = fast.Next.Next

		if fast == slow {
			p := head
			for p != slow {
				p = p.Next
				slow = slow.Next
			}
			return p
		}
	}
	return nil
}

// 空间换时间
// 利用map确定已经访问的节点
func detectCycle(head *ListNode) *ListNode {
	// 头指针是空
	if head == nil {
		return nil
	}
	seen := make(map[*ListNode]struct{}, 0)
	for head != nil {
		if _, ok := seen[head]; ok {
			return head
		}
		seen[head] = struct{}{}
		head = head.Next
	}

	return nil

}

// 给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。
// 请你将两个数相加，并以相同形式返回一个表示和的链表。
// 你可以假设除了数字 0 之外，这两个数都不会以 0 开头。
// 这个题目的重点是逆序排列
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	var head *ListNode
	var tail *ListNode
	// 进位
	carry := 0
	for l1 != nil || l2 != nil {
		n1, n2 := 0, 0
		if l1 != nil {
			n1 = l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			n2 = l2.Val
			l2 = l2.Next
		}

		sum := n1 + n2 + carry
		// 逆序排列，所以是个 --》十 --》百等，对10取模是位数，对10整数是进位数
		sum, carry = sum%10, sum/10
		if head == nil {
			head = &ListNode{
				Val: sum,
			}
			tail = head
		} else {
			tail.Next = &ListNode{
				Val: sum,
			}
			tail = tail.Next
		}
	}
	// 增加原始位数，需要增加一个节点
	if carry > 0 {
		tail.Next = &ListNode{
			Val: carry,
		}
	}
	return head
}

// 编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：
// 每行的元素从左到右升序排列。
// 每列的元素从上到下升序排列。
// 这个解法有点像是z行的那个算法题
func searchMatrix(matrix [][]int, target int) bool {
	if len(matrix) == 0 {
		return false
	}
	// 从左上角到开始搜索
	x, m, y := 0, len(matrix), len(matrix[0])-1
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

// 给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。
func swapPairs(head *ListNode) *ListNode {
	dummyHead := &ListNode{Val: 0, Next: head}
	temp := dummyHead
	for temp.Next != nil && temp.Next.Next != nil {
		// 这个是最关键的一步
		node1 := temp.Next
		node2 := temp.Next.Next
		temp.Next = node2
		node1.Next = node2.Next
		node2.Next = node1
		temp = node1
	}
	return dummyHead.Next
}

// 给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	length := 0
	temp := head
	for temp != nil {
		temp = temp.Next
		length++
	}
	dummy := &ListNode{
		Val:  0,
		Next: head,
	}
	cur := dummy
	for i := 0; i < length-n; i++ {
		cur = cur.Next
	}
	cur.Next = cur.Next.Next
	return dummy.Next
}

// 特殊列表
type Node struct {
	Val    int
	Next   *Node
	Random *Node
}

// 给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。

// 构造这个链表的 深拷贝。 深拷贝应该正好由 n 个 全新 节点组成，其中每个新节点的值都设为其对应的原节点的
// 新节点的 next 指针和 random 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。
// 复制链表中的指针都不应指向原链表中的节点 。

// 例如，如果原链表中有 X 和 Y 两个节点，其中 X.random --> Y 。那么在复制链表中对应的两个节点 x 和 y ，同样有 x.random --> y 。

// 返回复制链表的头节点。

// 用一个由 n 个节点组成的链表来表示输入/输出中的链表。每个节点用一个 [val, random_index] 表示：

// val：一个表示 Node.val 的整数。
// random_index：随机指针指向的节点索引（范围从 0 到 n-1）；如果不指向任何节点，则为  null 。
// 你的代码 只 接受原链表的头节点 head 作为传入参数。
func copyRandomList(head *Node) *Node {
	if head == nil {
		return nil
	}

	// 复制节点
	for node := head; node != nil; node = node.Next.Next {
		node.Next = &Node{
			Val:  node.Val,
			Next: node.Next,
		}
	}

	// cp random节点
	for node := head; node != nil; node = node.Next.Next {
		if node.Random != nil {
			node.Next.Random = node.Random.Next
		}
	}

	// 将next节点指向cp的节点
	dummy := head.Next
	for node := head; node != nil; node = node.Next {
		nodeNew := node.Next // 指向原数组的节点
		node.Next = node.Next.Next
		// 判断不是最后一项元素
		if node.Next != nil {
			nodeNew.Next = nodeNew.Next.Next
		}
	}
	return dummy
}

// 给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。

// k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。

// 你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。
// 题解：双指针：pre和end将链表分为：已翻转、待翻转、未反转
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func reverseKGroup(head *ListNode, k int) *ListNode {
	if head == nil {
		return nil
	}
	// 翻转链表
	reverseList := func(node *ListNode) *ListNode {
		var pre *ListNode
		cur := node
		for cur != nil {
			next := cur.Next
			cur.Next = pre
			pre = cur
			cur = next
		}
		return pre
	}
	dummy := &ListNode{
		Next: head,
	}

	pre, end := dummy, dummy
	for end != nil {
		for i := 0; i < k && end != nil; i++ {
			end = end.Next
		}
		if end == nil {
			break
		}
		next := end.Next
		start := pre.Next
		end.Next = nil
		pre.Next = reverseList(start)
		start.Next = next
		pre = start
		end = pre
	}
	return dummy.Next
}

// 给你一个链表数组，每个链表都已经按升序排列。

// 请你将所有链表合并到一个升序链表中，返回合并后的链表。
// 题解：两个两的合并，顺序合并，还可以用分治减少计算次数：宣传减少k/2
func mergeKLists(lists []*ListNode) *ListNode {
	m := len(lists)

	if m == 0 {
		return nil
	}

	if m == 1 {
		return lists[0]
	}

	// 合并
	left := mergeKLists(lists[:m/2])  // 合并左半部分
	right := mergeKLists(lists[m/2:]) // 合并右部分
	return mergerTwoList(left, right) // 最后组合合并
}
func mergerTwoList(list1, list2 *ListNode) *ListNode {
	l1, l2 := list1, list2
	if l1 == nil {
		return l2
	}

	if l2 == nil {
		return l1
	}

	head := &ListNode{}
	node := head
	for l1 != nil && l2 != nil {
		if l1.Val < l2.Val {
			node.Next = l1
			l1 = l1.Next
		} else {
			node.Next = l2
			l2 = l2.Next
		}
		node = node.Next
	}
	if l1 != nil {
		node.Next = l1
	}

	if l2 != nil {
		node.Next = l2
	}

	return head.Next
}

// 给你两个链表 list1 和 list2 ，它们包含的元素分别为 n 个和 m 个。
// 请你将 list1 中下标从 a 到 b 的全部节点都删除，并将list2 接在被删除节点的位置。
// 题解：a的前置和b的后置，list2的尾节点
func mergeInBetween(list1 *ListNode, a int, b int, list2 *ListNode) *ListNode {
	if list1 == nil {
		return nil
	}

	if list2 == nil {
		return list1
	}
	// 计算出a前置节点
	pre := list1
	for i := 0; i < a-1; i++ {
		pre = pre.Next
	}
	end := list1
	// 计算出b的前置节点
	for i := 0; i < b; i++ {
		end = end.Next
	}

	pre.Next = list2
	for list2.Next != nil {
		list2 = list2.Next
	}

	list2.Next = end.Next
	return list1
}
