package graph

// 在给定的 m x n 网格 grid 中，每个单元格可以有以下三个值之一：

// 值 0 代表空单元格；
// 值 1 代表新鲜橘子；
// 值 2 代表腐烂的橘子。
// 每分钟，腐烂的橘子 周围 4 个方向上相邻 的新鲜橘子都会腐烂。

// 返回 直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 -1 。
// 题解：在于记录好橘子的数量&坏橘子的位子，然后利用广度优先搜索
func orangesRotting(grid [][]int) int {
	// 记录坏掉的橘子,好橘子的数量
	good := 0
	// 坏橘子的坐标
	bad := [][2]int{}
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] == 1 {
				good++
			}
			if grid[i][j] == 2 {
				bad = append(bad, [2]int{i, j})
			}
		}
	}

	// 扩散次数
	cnt := 0
	// 遍历坏橘子，开始感染好橘子
	for len(bad) > 0 {
		next := [][2]int{} // 记录本次腐烂的橘子数量
		for len(bad) > 0 {
			// 记录当前橘子
			cur := bad[0]
			bad = bad[1:]
			x, y := cur[0], cur[1]

			// 扩散上下左右的橘子，逻辑相同
			if x+1 < len(grid) && grid[x+1][y] == 1 {
				good--
				grid[x+1][y] = 2
				next = append(next, [2]int{x + 1, y})
			}

			if x-1 >= 0 && grid[x-1][y] == 1 {
				good--
				grid[x-1][y] = 2
				next = append(next, [2]int{x - 1, y})
			}
			if y-1 >= 0 && grid[x][y-1] == 1 {
				good--
				grid[x][y-1] = 2
				next = append(next, [2]int{x, y - 1})
			}
			if y+1 < len(grid[0]) && grid[x][y+1] == 1 {
				good--
				grid[x][y+1] = 2
				next = append(next, [2]int{x, y + 1})
			}
		}

		// 本次腐烂的橘子数量大于0，则扩散+1
		if len(next) > 0 {
			cnt++
		}

		// 下一次扩散橘子的数量
		bad = next
	}

	// 还有橘子没有被腐烂
	if good > 0 {
		return -1
	}

	return cnt
}
