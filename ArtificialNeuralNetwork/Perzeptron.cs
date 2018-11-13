using System;
using System.Linq;

namespace ArtificialNeuralNetwork
{
    public class Perzeptron
    {
        //public double[] enterSignal; // входные сигналы
        public double[] weights_delta; // Дельта весов
        public double[] weighSignal; // Весовые коифициенты		
        public double extraWeight; // доп Вес
        public double sum; //сумма
        public double activation; //выход аксон (Активация узлов слоя)
        public Random randObj;
        public int weighSignalLength;

        //==================================================
        // Блок обратного распространения ошибки
        //==================================================
        public double[] changeSignal; // Изменения весовых коифициентов
        public double errorSignal; // Содержит ошибку для узла, то есть желаемый результат

        //==================================================
        // Блок внутренних переменных
        //==================================================
        

        /** Закрываем конструктор по умолчанию */
        public Perzeptron()
        {
            // TODO закрытый конструктор 		
        }

        /**
         * @param enter веса входных сигналов
         */
        public Perzeptron(int enter)
        {
            randObj = new Random((int)DateTime.Now.Ticks); ;
            //enterSignal = new double[enter];// инициализируем входные сигналы
            changeSignal = new double[enter];// инициализируем зменения весовых коифициентов
            weighSignal = new double[enter];// инициализируем веса Перцептроны
            weights_delta = new double[enter];// инициализируем веса дельты весов
            for (int i = 0; i < weighSignal.Length; i++)
            {
                weighSignal[i] = rand();
            }
            extraWeight = rand();
            weighSignalLength = weighSignal.Length;
        }

        /**
         * Очищаем изменения весовых коифициентов
         */
        public void ClearChange()
        {
            extraWeight = rand();
            for (int i = 0; i < changeSignal.Length; i++)
            {
                changeSignal[i] = 0;
            }
        }

        /**
         * Получить размер весов
         * @return Колличество связей(весов) в нейроне
         */
        public int getWeighsSize()
        {
            return weighSignal.Length;
        }

        /**
         * @return возвращает случайное значение от -0.25 до 0.25 
         */
        private double rand()
        {
            double minX = -0.25D;
            double maxX = 0.25D;
            double finalX = RandomProvider.GetThreadRandom().NextDouble() * (maxX - minX) + minX;
            //Console.Write(String.Format(" {0:0.###} ", finalX));
            return finalX;
        }
    }/**End Perzeptron class*/


}