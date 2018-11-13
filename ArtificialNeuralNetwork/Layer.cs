using System;

namespace ArtificialNeuralNetwork
{
    public class Layer
    {
        public Perzeptron[] neurons; // нерйроны в слое

        /**
         * Закрываем конструктор по умолчанию 
         */
        public Layer()
        {
            // TODO закрытый конструктор
        }

        /**Инициализация слоя
         * @param numberOfNeurons число нейронов в слое
         * @param enter число входных сигналов в нейрон от предыдущего слоя
         */
        public Layer(int numberOfNeurons, int enter)
        {
            neurons = new Perzeptron[numberOfNeurons]; //Создаем слой с заданым количестыом нейроноы
            for (int i = 0; i < numberOfNeurons; i++)
            {
                neurons[i] = new Perzeptron(enter);
            }
        }

        /**
         * Очищаем предыдущие изменения в слое
         */
        public void ClearLayerChange()
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i].ClearChange();
            }
        }

        /**
         * Получить размер слоя
         * @return Колличество нейронов в слое
         */
        public int getSize()
        {
            return neurons.Length;
        }
    }/**End Layer class*/
}