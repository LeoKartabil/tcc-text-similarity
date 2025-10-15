using System;

class Programa
{
    static void Main()
    {
        int[] numeros = { 5, 2, 9, 1, 5, 6 };
        BubbleSort(numeros);

        Console.WriteLine("Array ordenado:");
        foreach (int num in numeros)
        {
            Console.Write(num + " ");
        }
    }

    static void BubbleSort(int[] arr)
    {
        int n = arr.Length;
        for (int i = 0; i < n - 1; i++)
        {
            for (int j = 0; j < n - i - 1; j++)
            {
                if (arr[j] > arr[j + 1])
                {
                    // troca os elementos
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
    }
}