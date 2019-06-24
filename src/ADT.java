
import java.util.Scanner;

public class ADT {


    public static class Factor{
        public int cof;
        public int index;
        public Factor(int cof,int index){
            this.cof = cof;
            this.index = index;
            next = null;
        }
        public Factor next;
    }
    public static  Factor insert(Factor head,Factor factor){
        if(head==null) return factor;
        if(head.index<factor.index){
            factor.next = head;
            return factor;
        }
        Factor curr = head;
        while(curr!=null){
            if(curr.index==factor.index){
                curr.cof +=factor.cof;
                return head;
            }else if(curr.next!=null&&curr.index>factor.index&&curr.next.index<factor.index){
                factor.next = curr.next;
                curr.next = factor;
                return head;
            }else if(curr.next==null){
                curr.next = factor;
                return head;
            }
            curr = curr.next;
        }
        return head;
    }
    public static void main(String[] args){
        Scanner in = new Scanner(System.in);
        int n = in.nextInt();

        while(n-->0){
            int cof = in.nextInt(),index = in.nextInt();
            Factor list = null;
            while(index>=0){
                list=insert(list,new Factor(cof,index));
                cof = in.nextInt();index = in.nextInt();
            }
            cof = in.nextInt();index = in.nextInt();
            while(index>=0){
                list = insert(list,new Factor(cof,index));
                cof = in.nextInt();index = in.nextInt();
            }

            while(list!=null){
                if(list.cof!=0)
                    System.out.print("[ "+list.cof+" "+list.index+" ] ");
                list = list.next;
            }
            System.out.println();
        }

    }



}
