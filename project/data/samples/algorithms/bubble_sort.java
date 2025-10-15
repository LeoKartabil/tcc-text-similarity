public class BubbleSortExample {
    public static void main(String[] args) {
        int[] numeros = { 5, 2, 9, 1, 5, 6 };
        bubbleSort(numeros);

        System.out.println("Array ordenado:");
        for (int num : numeros) {
            System.out.print(num + " ");
        }
    }

    public static void bubbleSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    // troca os elementos
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
    }
}