public class myArray {
    /**
     * 两趟扫描法
     * dp[i] 前i个孩子所需的最小糖果量
     * 老师想给孩子们分发糖果，有 N 个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。
     *
     * 你需要按照以下要求，帮助老师给这些孩子分发糖果：
     *
     * 每个孩子至少分配到 1 个糖果。
     * 相邻的孩子中，评分高的孩子必须获得更多的糖果。
     * 那么这样下来，老师至少需要准备多少颗糖果呢？
     * @param ratings
     * @return
     */
    public int candy(int[] ratings) {
        int n = ratings.length;
        int [] candies = new int [n];
        candies[0] = 1;
        for(int i = 1; i < n; i++){
            if(ratings[i] > ratings[i - 1])
                candies[i] = candies[i - 1] + 1;
            else
                candies[i] = 1;
        }
        int ans = candies[n - 1];
        for(int i = n - 2; i >= 0; i--){
            if(ratings[i] > ratings[i + 1])
                candies[i] = Math.max(candies[i], candies[i + 1] + 1);
            ans += candies[i];
        }
        return ans;
    }


}
