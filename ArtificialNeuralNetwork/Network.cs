using System;
using System.Diagnostics;
using System.Linq;

namespace ArtificialNeuralNetwork
{
    public class Network
    {
        public double Eta = 0.01d;
        private double wsMax = 5.5d; // максимальный вес
        private double wsMin = -5.5d; // минимальный вес

        public int difference = 0;

        // Сумма ответов сети
        //--------------------
        public double sum;

        // Сумма ошибок
        //--------------------
        public int error;

        // Дизайн сети
        //--------------------
        public int[] _netDesign;

        // Слои нейронной сети
        //--------------------
        public Layer[] layers;
        
        // Закрываем конструктор по умолчанию
        //-----------------------------------
        public Network()
        {
            // TODO закрытый конструктор
        }

        /**Инициализация сети
         * @param inputLayer количество нейронов во входящем слое
         * @param hiddenLayer количество нейронов в скрытом слое
         * @param outputLayer количество нейронов в выходном слое
         */
        public Network(int inputLayer, int hiddenLayer, int outputLayer)
        {
            this.sum = 0;
            layers = new Layer[3]; //Создаем сеть с заданым количестыом слоев
            for (int i = 0; i < layers.Length; i++)
            {
                switch (i)
                {
                    case 0:
                        {
                            layers[i] = new Layer(inputLayer, inputLayer);
                            break;
                        }
                    case 1:
                        {
                            layers[i] = new Layer(hiddenLayer, inputLayer);
                            break;
                        }
                    case 2:
                        {
                            layers[i] = new Layer(outputLayer, hiddenLayer);
                            break;
                        }
                    default:
                        {
                            break;
                        }
                }// end switch
            }// end for
        }// end Network


        //==================================================
        //Инициализация сети через массив
        //==================================================

        /**Инициализация сети
         * @param layers массив целых положительных чисел, числа означают количество нейронов в слое
         */
        public Network(int[] layers)
        {
            this.sum = 0;
            this.layers = new Layer[layers.Length]; //Создаем сеть с заданым количестыом слоев    	
            for (int i = 0; i < layers.Length; i++)
            {
                if (layers[i] <= 0)
                {
                    layers[i] = 1;
                    Console.WriteLine("initialization error");
                }

                if (i == 0)
                {
                    this.layers[0] = new Layer(layers[0], layers[0]);
                    continue;
                }

                this.layers[i] = new Layer(layers[i], layers[i - 1]);
            }// end for
            this._netDesign = layers;//сохраняем дизайн сети (для чтения)
        }// end Network



        /**
         * Подготовка к обратному распостранению ощибки (сброс предыдущих изменений) 
         */
        public void ClearChangeValues()
        {
            for (int i = 0; i < layers.Length; i++)
            {
                layers[i].ClearLayerChange();
            }// end for
            
        }// end ClearChangeValues

        /**
         * Получить размер выходного слоя
         */
        public int getOutputLayerSize(Layer[] layers)
        {
            return layers[layers.Length - 1].getSize();
        }

        /** Вычисление ошибки в выходном слое
         * @param actualOutput ожидаемый результат
         */
        public void CalculateOutputLayerErrors(double[] actualOutput)
        {
            Perzeptron[] ol = layers[layers.Length - 1].neurons;// нейроны выходного слоя
            for (int i = 0; i < getOutputLayerSize(this.layers); i++)
            {
                ol[i].errorSignal = ol[i].activation - actualOutput[i];
            }
        }//end CalculateOutputLayerErrors  

        public void DeltaOutputLayerWeightChange()
        {
            //double Eta = 0.01;
            for (int i = this.layers.Length - 1; i >= 0; i--) //проходим по слоям нейросети [i]
            {
                var l_size = this.layers[i].getSize();
                for (int j = 0; j < l_size; j++) //проходим по неронам слоя [j]
                {
                    if (i == this.layers.Length - 1)
                    {
                        Perzeptron[] ol = layers[i].neurons;// используемый слой (current layer)
                        Perzeptron[] pl = layers[i - 1].neurons;
                        var curLayerNeuSize = layers[i].neurons[j].weighSignalLength;
                        for (int k = 0; k < curLayerNeuSize; k++)
                        {
                            ol[j].weights_delta[k] = ol[j].errorSignal * (ol[j].activation * (1 - ol[j].activation));
                            var weighSignal = ol[j].weighSignal[k] - (pl[k].activation * ol[j].weights_delta[k] * this.Eta);
                            if (weighSignal < wsMax && weighSignal > wsMin)
                            {
                                ol[j].weighSignal[k] = weighSignal;
                            }
                        }
                        continue;
                    }
                }
            }
        }//end DeltaOutputLayerWeightChange

        /**
         * Вычисление ошибок в скрытых слоях
         */
        public void CalculateHiddenLayersErrors(double[] actualInputs)
        {
            for (int i = this.layers.Length - 2; i >= 0; i--)
            {
                double sum; //временная переменная для расчета ошибок скрытого слоя
                var curLayerSize = this.layers[i].getSize();
                for (int j = 0; j < curLayerSize; j++)
                {
                    sum = 0;
                    Perzeptron[] cl = layers[i].neurons;// текущий слой (current layer)
                    Perzeptron[] pl = layers[i + 1].neurons;// предыдущий слой (previous layer)
                    var nextLayerSize = this.layers[i + 1].getSize();
                    for (int k = 0; k < nextLayerSize; k++)
                    {
                        sum += pl[k].weights_delta[j] * pl[k].weighSignal[j];
                    }
                    cl[j].errorSignal = sum;
                }
                WeightChange(actualInputs, i);
            }
        }//end CalculateHiddenLayersErrors

        public void WeightChange(double[] actualInputs, int i)
        {
            //double Eta = 0.1;
            var l_size = this.layers[i].getSize();
            for (int j = 0; j < l_size; j++) //проходим по неронам слоя [j]
            {
                if (i == this.layers.Length - 1)
                {
                    continue;
                }

                if (i == 0)
                {
                    Perzeptron[] ol = layers[i].neurons;// используемый слой (current layer)
                    var firstLayNeuSize = layers[i].neurons[j].weighSignalLength;
                    for (int k = 0; k < firstLayNeuSize; k++)
                    {
                        ol[j].weights_delta[k] = ol[j].errorSignal * (ol[j].activation * (1 - ol[j].activation));
                        var weighSignal = ol[j].weighSignal[k] - (actualInputs[k] * ol[j].weights_delta[k] * this.Eta);
                        if (weighSignal < wsMax && weighSignal > wsMin)
                        {
                            ol[j].weighSignal[k] = weighSignal;
                        }
                    }
                    continue;
                }

                Perzeptron[] cl = layers[i].neurons;// используемый слой (current layer)
                Perzeptron[] pl = layers[i - 1].neurons;// предыдущий слой (previous layer)

                var curLayNeuSize = layers[i].neurons[j].weighSignalLength;
                for (int k = 0; k < curLayNeuSize; k++)
                {
                    cl[j].weights_delta[k] = cl[j].errorSignal * (cl[j].activation * (1.0d - cl[j].activation)); //Производная функция
                    var weighSignal = cl[j].weighSignal[k] - (pl[k].activation * cl[j].weights_delta[k] * this.Eta);
                    if (weighSignal < wsMax && weighSignal > wsMin)
                    {
                        cl[j].weighSignal[k] = weighSignal;
                    }
                }
            }
        }//end WeightChange


        // Run the neural net
        public void RunNet(double[] actualInputs)
        {
            for (int i = 0; i < this.layers.Length; i++)
            {
                double sum; //временная переменная для расчета ошибок скрытого слоя
                var l_size = this.layers[i].getSize();
                for (int j = 0; j < l_size; j++)
                {
                    sum = 0;
                    Perzeptron[] cl = layers[i].neurons;// текущий слой (current layer)
                    if (i == 0)
                    {
                        for (int k = 0; k < cl[i].weighSignalLength; k++)
                        {
                            sum = sum + (actualInputs[k] * cl[j].weighSignal[k]);
                        }
                        //sum = sum + cl[j].extraWeight;//???
                        cl[j].sum = sum;
                        cl[j].activation = f(cl[j].sum);//???
                        continue;
                    }

                    Perzeptron[] pl = layers[i - 1].neurons;// предыдущий слой (previous layer)
                    var previousLayerSize = this.layers[i - 1].getSize();
                    for (int k = 0; k < previousLayerSize; k++)
                    {
                        sum = sum + pl[k].activation * cl[j].weighSignal[k];
                    }
                    //sum = sum + cl[j].extraWeight;//???
                    cl[j].sum = sum;
                    cl[j].activation = f(cl[j].sum);//???
                }
            }
        }//end run_net

        // Sigmoid function
        public double f(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        // Промежуточный результат Для теста
        public double GetNeyResult()
        {
            double result = 0;            
            int lastLayer = layers.Length - 1;
            Console.WriteLine("");
                      
            for (int i = 0; i < layers[lastLayer].getSize(); i++)
            {
                //System.out.print(String.format("%(.2f   ", layers[lastLayer].neurons[i].activation));
                Console.Write(String.Format(" {0:0.###} ", layers[lastLayer].neurons[i].activation));
                //Console.Write(layers[lastLayer].neurons[i].activation);
            }

            Console.WriteLine("");
            return result;
        }//end run_net

        // Заполняет Массив ответов
        public void GetNetAnswer(double[] Output)
        {
            int lastLayer = layers.Length - 1;
            for (int i = 0; i < layers[lastLayer].getSize(); i++)
            {
                Output[i] = layers[lastLayer].neurons[i].activation;
            }
        }//end GetNetAnswer

        //метод тренеровки нейросети
        //указываем входящие значения actualInputs
        //рекомендованые исходящие значения при таких входящих
        //-----------------------------------------------------
        public void TrainNet(double[] actualInputs, double[] actualOutput)
        {
            RunNet(actualInputs);
            ClearChangeValues();
            CalculateOutputLayerErrors(actualOutput);
            DeltaOutputLayerWeightChange();
            CalculateHiddenLayersErrors(actualInputs);
        }
    }/**End Network class*/

}