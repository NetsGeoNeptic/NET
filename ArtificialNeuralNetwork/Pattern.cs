namespace ArtificialNeuralNetwork
{
    public class Pattern
    {
        public double[] actualInputs; // входные сигналы
        public double[] actualOutput; // выходные сигналы сигналы

        /** «акрываем конструктор по умолчанию */
        private Pattern()
        {
            // TODO закрытый конструктор		
        }

        //rjycnhernjh ghbybvftn dtotcndtyyst xbckf
        public Pattern(double[] actualInputs, double[] actualOutput)
        {
            this.actualInputs = actualInputs; // входные сигналы
            this.actualOutput = actualOutput; // выходные сигналы сигналы
        }

        //конструктор принимает целые числа
        public Pattern(int[] actualInputs, int[] actualOutput)
        {
            //приведение типов
            this.actualInputs = new double[actualInputs.Length];//инициализация размера
            for (int i = 0; i < actualInputs.Length; ++i)// входные сигналы
            {
                this.actualInputs[i] = (double)actualInputs[i];
            }

            this.actualOutput = new double[actualOutput.Length];//инициализация размера
            for (int i = 0; i < actualOutput.Length; ++i)// выходные сигналы сигналы
            {
                this.actualOutput[i] = (double)actualOutput[i];
            }
        }

        //конструктор принимает гибридные числа
        public Pattern(int[] actualInputs, double[] actualOutput)
        {
            //приведение типов
            this.actualInputs = new double[actualInputs.Length];//инициализация размера
            for (int i = 0; i < actualInputs.Length; ++i)// входные сигналы
            {
                this.actualInputs[i] = (double)actualInputs[i];
            }

            this.actualOutput = new double[actualOutput.Length];//инициализация размера
            for (int i = 0; i < actualOutput.Length; ++i)// выходные сигналы сигналы
            {
                this.actualOutput[i] = (double)actualOutput[i];
            }
        }

    }
}