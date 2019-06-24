

public class StreamChecker {
    String[] words;
    StringBuilder stringBuilder;
    public StreamChecker(String[] words) {
        this.words = words;
        this.stringBuilder = new StringBuilder();
    }
    public boolean query(char letter) {
       this.stringBuilder.append(letter);
       int n = stringBuilder.length();
       for(String s:words){
            int start =Math.max(0,n-s.length());
            if(s.equals(stringBuilder.substring(start)))return true;
       }
       return  false;
    }
}
