import java.util.*;

public class Solution {
    /***
     * leetcode 编程赛 2 校园自行车分配
     */
    class Node implements  Comparable<Node>{
        int index;
        int dis;
        int bike;

        public Node(int index,int dis,int bike){
            this.index = index;
            this.dis  = dis;
            this.bike = bike;
        }
        public int compareTo(Node o1){
            return this.dis == o1.dis? this.index==o1.index?this.bike-o1.bike:this.index - o1.index : this.dis - o1.dis;
        }
    }
    public int[] assignBikes(int[][] workers, int[][] bikes) {
        int n = workers.length,m = bikes.length;
        int [] res = new int[n];
        for(int i=0;i<n;i++) res[i]=-1;

        boolean[] isUse = new boolean[m];

        PriorityQueue<Node> pq = new PriorityQueue<>();
        for(int i=0;i<n;i++){
            for(int j =0;j<m;j++){
                int dis = Math.abs(workers[i][0] - bikes[j][0])+Math.abs(workers[i][1] -bikes[j][1]);
                pq.add(new Node(i,dis,j));
            }
        }
        int k=0;
        while(k<n){
            Node t = pq.poll();
            if(!isUse[t.bike]&&res[t.index]==-1){
                res[t.index] = t.bike;
                isUse[t.bike] = true;
                k++;
            }

        }

        return res;

    }

    /***
     * 3.最小化舍入误差以满足目标
     */
    /**
     * leetcode 139 单词拆分
     * @param s
     * @param wordDict
     * @return
     */

}
