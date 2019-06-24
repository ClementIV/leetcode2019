

import java.util.Scanner;

/**
 * Created by Idan on 2019/5/25.
 */
public class ProblemB {

    public static void main (String args[]) {
        Scanner sc = new Scanner(System.in);

        int N = sc.nextInt(), M = sc.nextInt();

        int[][] grid = new int[N][M];

        for (int i=0; i<N; i++) {
            for (int j=0; j<M;j++) {
                grid[i][j] = sc.nextInt();
            }
        }

        boolean[][] visited = new boolean[N][M];
        int ans = 0;

        for (int i=0; i<N; i++) {
            for (int j=0; j<M; j++) {
                if (grid[i][j] == 1 && visited[i][j] == false) {
                    ans++;
                    bfs(grid, visited, i, j, N, M);
                }
            }
        }
        System.out.println(ans);
    }

    private static void bfs(int[][] grid, boolean[][] visited, int i, int j, int N,  int M) {

        if (i>=0 && i<N && j>=0 && j<M) {
            if (grid[i][j] == 1 && visited[i][j] == false) {
                visited[i][j] = true;
                bfs(grid, visited, i+1, j, N, M);
                bfs(grid, visited, i-1, j, N, M);
                bfs(grid, visited, i, j+1, N, M);
                bfs(grid, visited, i, j-1, N, M);
                bfs(grid, visited, i+1, j+1, N, M);
                bfs(grid, visited, i+1, j-1, N, M);
                bfs(grid, visited, i-1, j-1, N, M);
                bfs(grid, visited, i-1, j+1, N, M);
            }
        }
    }

}