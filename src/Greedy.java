import java.util.Arrays;
import java.util.PriorityQueue;
import java.util.Scanner;

public class Greedy {

    public static class Lake implements Comparable<Lake>{
        public int id;
        public int fish;
        public int dec;
        public Lake(int a,int b,int c){
            id = a;
            fish = b;
            dec = c;
        }

        @Override
        public int compareTo(Lake o) {
            if(this.fish==o.fish) return this.id - o.id;
            return o.fish - this.fish;
        }
    }

    public static void main(String[] args){
        Scanner  in = new Scanner(System.in);
        while(in.hasNext()){
            int n = in.nextInt();
            if(n==0) return ;
            int h = in.nextInt()*12;
            int[] f = new int[n],d = new int[n],t = new int[n];

            int[] maxFishTime = new int[n],tempFishTime = new int[n];
            //初始并求和
            for(int i=0;i<n;i++) f[i] =in.nextInt();
            for(int i=0;i<n;i++) d[i] = in.nextInt();
            for(int i=1;i<n;i++) t[i] = in.nextInt()+t[i-1];

            int maxFish = -1;

            for(int stopLake =0;stopLake<n;stopLake++){

                final int maxFishingTime = h - t[stopLake];

                PriorityQueue<Lake> minLake = new PriorityQueue<Lake>();
                for(int i=0;i<=stopLake;i++) minLake.add(new Lake(i,f[i],d[i]));

                int sumFish =0;
                //初始化临时钓鱼次数
                for(int i =0;i<n;i++) tempFishTime[i] = 0;

                for(int fishTimeh=0;fishTimeh<maxFishingTime;fishTimeh++){
                    //每次选择最多的鱼的湖
                    Lake tempLake = minLake.poll();
                    sumFish += tempLake.fish;

                    tempFishTime[tempLake.id]++;
                    //更新湖中鱼的信息
                    tempLake.fish -= tempLake.dec;
                    if(tempLake.fish<0) tempLake.fish =0;

                    minLake.add(tempLake);

                }
                if(sumFish>maxFish){
                    maxFishTime = Arrays.copyOf(tempFishTime,n);
                    maxFish = sumFish;
                }
            }

            for(int i=0;i<n-1;i++)
                System.out.print(maxFishTime[i]*5+", ");
            System.out.println(maxFishTime[n-1]*5);
            System.out.println("Number of fish expected: "+maxFish+"\n");
        }

    }
}
