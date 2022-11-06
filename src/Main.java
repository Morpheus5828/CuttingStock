public class Main {
    public static void main(String[] args) {

        // initialise a price tab
        int[] price = new int[11];
        price[0] = 0;
        price[1] = 1;
        price[2] = 5;
        price[3] = 8;
        price[4] = 9;
        price[5] = 10;
        price[6] = 17;
        price[7] = 17;
        price[8] = 20;
        price[9] = 24;
        price[10] = 30;

        int stockLength = 4;

        new CuttingStock(price, stockLength);
    }

}