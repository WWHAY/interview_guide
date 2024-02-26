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

// 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
// 岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
// 此外，你可以假设该网格的四条边均被水包围。
// 深度优先搜索：关键是在于递归
func numIslands(grid [][]byte) int {
	m, n := len(grid), len(grid[0])

	var dfs func(r, c int)
	dfs = func(r, c int) {
		// 递归终止条件
		if r < 0 || r >= m || c < 0 || c >= n {
			return
		}

		// 水or已经访问过的陆地
		if grid[r][c] != '1' {
			return
		}

		// 标记已经被访问过了
		grid[r][c] = 0
		dfs(r-1, c)
		dfs(r+1, c)
		dfs(r, c-1)
		dfs(r, c+1)
	}
	ans := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] == '1' {
				// 遇到陆地，就+1，把四周相连的陆地变成非陆地，防止无限循环
				ans++
				dfs(i, j)
			}
		}
	}
	return ans
}

// 广度优先搜索
func numIslandsBfs(grid [][]byte) int {
	m, n := len(grid), len(grid[0])
	ans := 0
	quene := [][2]int{}
	// 增加陆地的坐标
	for i := 0; i < m; i++ {
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] == '1' {
				// 多少次广度优先搜索
				ans++
				quene = append(quene, [2]int{i, j})
				for len(quene) > 0 {
					neighbor := quene[0]
					quene = quene[1:]
					r, c := neighbor[0], neighbor[1]
					if r+1 < m && grid[r+1][c] == '1' {
						quene = append(quene, [2]int{r + 1, c})
						grid[r+1][c] = '0'
					}

					if r-1 >= 0 && grid[r-1][c] == '1' {
						quene = append(quene, [2]int{r - 1, c})
						grid[r-1][c] = '0'
					}

					if c-1 >= 0 && grid[r][c-1] == '1' {
						quene = append(quene, [2]int{r, c - 1})
						grid[r][c-1] = '0'
					}

					if c+1 < n && grid[r][c+1] == '1' {
						quene = append(quene, [2]int{r, c + 1})
						grid[r][c+1] = '0'
					}
				}
			}
		}
	}
	return ans
}

// 你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。
// 在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，
// 其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。
// 例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
// 请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。
func canFinish(numCourses int, prerequisites [][]int) bool {
	// 定义变量
	// 可以修的课程数
	finishCourse := []int{}
	// 先修课程 -- 修习的课程
	preToCour := make([][]int, numCourses)
	// 一个课程有多少个先修课程
	numPreCourses := make([]int, numCourses)

	// 初始化变量
	for _, course := range prerequisites {
		// 数组赋值的话，一定要make数组，数组追加的话，可以不用
		preToCour[course[1]] = append(preToCour[course[1]], course[0])
		numPreCourses[course[0]]++
	}

	// 增加广度优先搜索的队列
	quene := []int{}
	for i := 0; i < numCourses; i++ {
		// 没有先修课程的
		if numPreCourses[i] == 0 {
			quene = append(quene, i)
		}
	}

	for len(quene) > 0 {
		c := quene[0] // 没有先修课程的或者先修课程已经修习完成的
		quene = quene[1:]
		finishCourse = append(finishCourse, c)
		for _, v := range preToCour[c] {
			numPreCourses[v]--
			// 该课程的先修课程已经学完，可以修习了该课程了，所以加入到quene队列中
			if numPreCourses[v] == 0 {
				quene = append(quene, v)
			}
		}
	}
	return len(finishCourse) == numCourses
}
