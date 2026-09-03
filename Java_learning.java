class Robot{

    //attributes
    static String model = "Rosbot3" ;
    String name ;
    double battery;

    //constructor
    Robot(String name){
        this.name = name ;
        this.battery = 100 ;
    }

    // methodes
    public void move(String direction, int distance){
        System.out.println("Moving " + direction );
        this.battery -= distance*2 ;
    } 

    public void recharge(double time){
        System.out.println("The Robot would recharge for " + time + " hours") ;

        this.battery += time*0.5 ;
    }

}





class Main {

    static int square(int a){
        return  a*a ;
    }

    static boolean isEven(int a){
        boolean b = false ;

        if(a % 2 == 0){
            b = true ; 
        }

        return  b ;
    }

    public static void main(String[] args){

        Robot r = new Robot("Terminator") ;

        System.out.println(r.name) ;

        r.move("north", 5);

        System.out.println(r.battery);

        r.recharge(10.5);

        System.out.println(r.battery);



    }
}