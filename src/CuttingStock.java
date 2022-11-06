public class CuttingStock {
    private int length;
    private int[] price;
    private int[] maxRevenue;
    private int[] optimalSize;

    public CuttingStock(int[] price, int length) {
        length = 7;
        this.price = price;
        this.length = length;
        this.optimalSize = new int[length+1];
        this.maxRevenue = new int[length+1];
        this.maxRevenue[0] = 0; // Stock with length 0 equal 0

        algo();
        //displayTab(maxRevenue);
        //displayTab(optimalSize);

        displayResult(price, length);
    }

    private void algo() {
        for(int j = 1; j <= length; j++) {
            int q = -1;
            for(int i = 1 ; i <= j; i++) {
                if(q < price[i] + maxRevenue[j - i]) {
                    q = price[i] + maxRevenue[j - i];
                    optimalSize[j] = i;
                }

            }
            maxRevenue[j] = q;
        }
    }

    private void displayResult(int[] price, int length) {
        while (length > 0) {
            System.out.println(optimalSize[length]);
            length -= optimalSize[length];
        }
    }

    private void displayTab(int[] tab) {
        System.out.print("Tab : \t");
        for(int i : tab)
            System.out.print(i  + "\t");
    }

}
