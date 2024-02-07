package main

// 1. 最小覆盖子串
// 给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。
// 注意：
// 对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量。
// 如果 s 中存在这样的子串，我们保证它是唯一的答案。
// 示例 1：
// 输入：s = "ADOBECODEBANC", t = "ABC"
// 输出："BANC"
// 解释：最小覆盖子串 "BANC" 包含来自字符串 t 的 'A'、'B' 和 'C'。

// 2. 无重复字符的最长子串
// 给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
// 示例 1:
// 输入: s = "abcabcbb"
// 输出: 3
// 解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
func lengthOfLongestSubstring(s string) int {
	cnt := make(map[byte]int, 0)
	ans := 0
	r := -1
	for l := 0; l < len(s); l++ {
		if l != 0 {
			delete(cnt, s[l-1])
		}

		for r+1 < len(s) && cnt[s[r+1]] == 0 {
			cnt[s[r+1]]++
			r++
		}

		ans = max(ans, r-l+1)
	}
	return ans
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
