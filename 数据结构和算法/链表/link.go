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
