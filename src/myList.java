import java.util.Arrays;
import java.util.Comparator;

public class myList {
    //反转链表非递归
    public ListNode reverseList(ListNode head) {
        if(head==null){
            return null;
        }
        ListNode pre = null;
        ListNode cur = head;
        ListNode next = cur.next;
        while(cur!=null){
            next = cur.next;
            cur.next = pre;
            pre = cur;
            cur  = next;

        }
        return pre;
    }

    /**
     * 反转链表递归
     * @param head
     * @return
     */
    public ListNode reverseList2(ListNode head) {
        if(head==null){
            return null;
        }else if(head.next==null){return head;}
        ListNode curr  = reverseList2(head.next);
        ListNode l = curr.next;
        head.next = null;
        while(l.next!=null){ l = l.next;}
        l.next = head;
        return curr;

    }

    public int[][] flipAndInvertImage(int[][] A) {
        int n = A.length;
        for(int i=0;i<n;i++){
            int l = 0,r = n-1;
            while(l<r){
                A[i][l] ^=1;A[i][r]^=1;
                int t = A[i][l];A[i][l] = A[i][r];A[i][r]= t;
                l++;
                r--;
            }
            if(l==r) A[i][l] ^=1;
        }
        return A;
    }
    public class myPair{
        int index ;
        String s;
        String t;
        public myPair(int index,String s,String t){
            this.index = index;
            this.s = s;
            this.t = t;
        }
    }
    public String findReplaceString(String S, int[] indexes, String[] sources, String[] targets) {
        int n = indexes.length;
        myPair[] pairs= new myPair[n];
        for(int i=0;i<n;i++) pairs[i] = new myPair(indexes[i],sources[i],targets[i]);
        Arrays.sort(pairs, Comparator.comparingInt(o -> o.index));

        StringBuilder ans = new StringBuilder(S);
        int offset=0;
        for(int i=0;i<n;i++){
            int j=pairs[i].s.length(),start=pairs[i].index;
            if(S.substring(start,start+j).equals(pairs[i].s)){
               ans = ans.replace(start+offset,offset+start+j,pairs[i].t);
               offset += pairs[i].t.length() - j;
            }
        }
        return ans.toString();
    }

    public int largestOverlap(int[][] A, int[][] B) {
        int n = A.length;
        int max =0;
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                max = Math.max(overLap(A,B,i,j,n,n,0,0),max);
            }
        }
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                max = Math.max(overLap(A,B,n-1-i,n-1-i,i+1,j+1,0,n-1-j),max);
            }
        }
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                max = Math.max(overLap(A,B,0,j,i+1,n,n-1-i,0),max);
            }
        }
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                max = Math.max(overLap(A,B,0,0,i+1,j+1,n-1-i,n-1-j),max);
            }
        }
        return  max;
    }
    public int overLap(int[][] A,int[][] B,int i,int j,int n1,int n2,int bi,int bj){
        int ans =0;
        for(int k=i,t1=bi;k<n1;k++,t1++){
            for(int m=j,t2=bj;m<n2;m++,t2++){
                if(A[k][m]== B[t1][t2]&&A[k][m]==1){ans++;}
            }
        }
        return ans;
    }
    public int[] sumOfDistancesInTree(int N, int[][] edges) {
        int[] dis = new int[N];
        int[][] graph = new int[N][N];
        for(int i =0;i<N;i++){
            for(int j=0;j<N;j++){
                graph[i][j] = 99999999;
                if(i==j){ graph[i][j]=0;}
            }
        }
        buildTree(graph,edges);
        for(int k=1;k<=N;k++) {
            for (int i = 1; i <= N; i++){
                for (int j = 1; j <= N; j++){
                    if (graph[i][j] > graph[i][k] + graph[k][j]){
                        graph[i][j] = graph[i][k] + graph[k][j];
                    }
                }
            }
        }
        for(int i = 0;i<N;i++){
            int ans =0;
            for(int j=0;j<N;j++){
                ans+=graph[i][j];
            }
            dis[i] = ans;
        }

        return dis;
    }

    private void buildTree(int[][] graph, int[][] edges) {
        for(int i=0;i<edges.length;i++){
            graph[edges[i][0]][edges[i][1]]=1;
            graph[edges[i][1]][edges[i][0]]=1;
        }
    }

}
